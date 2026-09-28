import random
import itertools
from typing import Optional
from .engine import SimulatorEngine
from .events import ViewerArrival, ViewerDeparture, MacroAnchorUpdate
from .state import SimulatorState
from .orchestrators import PPOAgentOrchestrator, MiniZincOrchestrator

class EventGenerator:
    """
    Independent generator of events.
    Responsible for generating Poisson arrivals for each PoP
    without creating self-sustaining infinite loops in handlers.
    """
    def __init__(self, engine: SimulatorEngine, pop_lambdas: list[float], until_time: float, event_counter: itertools.count, demand_counter: itertools.count, deterministic: bool = False, rng = None):
        self.engine = engine
        self.pop_lambdas = pop_lambdas
        self.until_time = until_time
        self.event_counter = event_counter
        self.demand_counter = demand_counter
        self.deterministic = deterministic
        self.rng = rng

    def bootstrap_arrivals(self, start_time: float = 0.0):
        """Schedule initial arrivals for each PoP."""
        for pop_id, lam in enumerate(self.pop_lambdas):
            if lam > 0:
                self.schedule_next_arrival(pop_id, current_time=start_time)

    def schedule_next_arrival(self, pop_id: int, current_time: float):
        lam = self.pop_lambdas[pop_id]
        if lam <= 0:
            return
            
        if self.deterministic:
            # Deterministic mode to remove variance and match MILP exactly
            inter_arrival = 1.0 / lam
        else:
            # Standard Poisson process
            if self.rng:
                inter_arrival = float(self.rng.exponential(1.0 / lam))
            else:
                inter_arrival = random.expovariate(lam)
            
        arrival_time = current_time + inter_arrival
        
        # Don't schedule beyond simulation horizon to avoid queue bloating
        if arrival_time <= self.until_time:
            demand_id = next(self.demand_counter)
            event_id = next(self.event_counter)
            
            arrival = ViewerArrival(
                time=arrival_time, 
                event_id=event_id, 
                demand_id=demand_id, 
                pop_id=pop_id
            )
            self.engine.schedule(arrival)


def setup_and_run_simulation(
    case_path: Optional[str] = None,
    sim_hours: float = 24.0,
    macro_interval: float = 0.25, # e.g. 15 minutes = 0.25 hours
    seed: int = 42
):
    random.seed(seed)
    
    # 1. Initialize Core
    engine = SimulatorEngine()
    state = SimulatorState(case_meta_path=case_path)
    
    # 2. Counters
    event_counter = itertools.count()
    demand_counter = itertools.count()
    
    # Example rates (Normally read from case_path if available)
    pop_lambdas = [5.0 for _ in range(state.num_pops)]
    service_mu = 1.0 # 1 hour average watching time
    
    # 3. Initialize Orchestrators
    ppo_agent = PPOAgentOrchestrator()
    mzn_solver = MiniZincOrchestrator("minizinc/model_erlang.mzn")
    
    # 4. Event Generator
    generator = EventGenerator(engine, pop_lambdas, sim_hours, event_counter, demand_counter)
    
    # 5. Event Handlers
    def handle_arrival(event: ViewerArrival):
        # Determine cluster choice
        cluster_id = ppo_agent.decide_cluster(state, event.pop_id)
        
        # Validate QoS and logic (Simulation State rejects invalid allocations)
        success = state.allocate_viewer(event.demand_id, cluster_id, event.pop_id)
        
        if success:
            # Schedule Departure
            service_time = random.expovariate(service_mu)
            departure = ViewerDeparture(
                time=event.time + service_time,
                event_id=next(event_counter),
                demand_id=event.demand_id,
                cluster_id=cluster_id
            )
            # Only schedule departure if it falls within the simulation window
            if departure.time <= sim_hours:
                engine.schedule(departure)
        else:
            state.total_unserved += 1
            
        # Schedule next arrival from the same POP
        generator.schedule_next_arrival(event.pop_id, event.time)

    def handle_departure(event: ViewerDeparture):
        state.remove_viewer(event.demand_id)
        
    def handle_macro_update(event: MacroAnchorUpdate):
        new_anchor = mzn_solver.compute_anchor(state)
        state.update_anchor(new_anchor)
        
        # Schedule next macro update
        next_time = event.time + macro_interval
        if next_time <= sim_hours:
            next_macro = MacroAnchorUpdate(time=next_time, event_id=next(event_counter))
            engine.schedule(next_macro)

    # Register handlers
    engine.register_handler(ViewerArrival, handle_arrival)
    engine.register_handler(ViewerDeparture, handle_departure)
    engine.register_handler(MacroAnchorUpdate, handle_macro_update)
    
    # 6. Bootstrap events
    engine.schedule(MacroAnchorUpdate(time=0.0, event_id=next(event_counter)))
    generator.bootstrap_arrivals()
        
    # 7. Execute
    engine.run(until_time=sim_hours)
    
    # Final Telemetry
    print(f"Simulation completed. Served: {state.total_served}, Unserved: {state.total_unserved}")

if __name__ == "__main__":
    setup_and_run_simulation()
