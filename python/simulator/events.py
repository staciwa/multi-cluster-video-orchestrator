from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class BaseEvent:
    """
    Base class for all discrete events.
    Ordered primarily by time. 'event_id' is used as a tie-breaker 
    to avoid comparing complex payloads if times are identical.
    """
    time: float
    event_id: int
    
    # Exclude payload from ordering
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __lt__(self, other):
        if self.time == other.time:
            return self.event_id < other.event_id
        return self.time < other.time


@dataclass
class ViewerArrival(BaseEvent):
    """
    Micro-scale event: A new viewer connects and requests a video stream.
    """
    demand_id: int = field(compare=False, default=-1)
    pop_id: int = field(compare=False, default=-1)


@dataclass
class ViewerDeparture(BaseEvent):
    """
    Micro-scale event: A viewer disconnects from the stream.
    """
    demand_id: int = field(compare=False, default=-1)
    cluster_id: int = field(compare=False, default=-1)
    pop_id: int = field(compare=False, default=-1)


@dataclass
class MacroAnchorUpdate(BaseEvent):
    """
    Macro-scale event: Periodic trigger to wake up MiniZinc.
    The solver will read the current load and topology to compute 
    the optimal global baseline (Anchor) for the PPO agent.
    """
    pass