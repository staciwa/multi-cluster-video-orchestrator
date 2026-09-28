import heapq
import time
from typing import List, Callable, Dict, Any
from .events import BaseEvent

class SimulatorEngine:
    """
    Discrete Event Simulator engine.
    Priority queue manages the time-ordered sequence of events.
    """
    def __init__(self):
        self.current_time: float = 0.0
        self.events_queue: List[BaseEvent] = []
        # Handlers mapping event classes to their processing functions
        self.handlers: Dict[type, Callable[[BaseEvent], None]] = {}
        
        # Telemetry / Statistics
        self.processed_events: int = 0
        
    def register_handler(self, event_type: type, handler: Callable[[BaseEvent], None]):
        """Register a callback function for a specific event type."""
        self.handlers[event_type] = handler

    def schedule(self, event: BaseEvent):
        """Push an event into the priority queue."""
        if event.time < self.current_time:
            raise ValueError(f"Cannot schedule event in the past (Event time: {event.time}, Current time: {self.current_time})")
        heapq.heappush(self.events_queue, event)

    def run(self, until_time: float):
        """
        Run the simulation until the specified simulation time.
        """
        print(f"[Engine] Starting simulation up to t={until_time}")
        start_real_time = time.time()
        
        while self.events_queue:
            # Peek at the next event
            next_event = self.events_queue[0]
            
            if next_event.time > until_time:
                break
                
            # Pop the event and advance simulation time
            event = heapq.heappop(self.events_queue)
            self.current_time = event.time
            self.processed_events += 1
            
            # Dispatch to the appropriate handler
            handler = self.handlers.get(type(event))
            if handler:
                handler(event)
            else:
                print(f"[Engine] Warning: No handler registered for {type(event).__name__}")
                
        real_duration = time.time() - start_real_time
        print(f"[Engine] Simulation stopped at t={self.current_time:.4f}")
        print(f"[Engine] Processed {self.processed_events} events in {real_duration:.4f} real seconds.")

