import time
from collections import namedtuple
from collections import OrderedDict

UpDownEventHandler = namedtuple('UpDownEventHandler', ('handler', 'debouncing_time'))

class UpDownEvent:
    def __init__(self, id, count_up, count_down, started_at, ended_at) -> None:
        self.id = id
        self.count_up = count_up
        self.count_down = count_down
        self.started_at = started_at
        self.ended_at = ended_at
    
    def __repr__(self):
        return f'UpDownEvent({self.id}, {self.count_up}, {self.count_down}, {self.started_at}, {self.ended_at})'

class UpDownEvents:
    def __init__(self) -> None:
        self.next_id = 0
        self.open_event = None
        self.events: OrderedDict[int, UpDownEvent] = OrderedDict()

    def register_new_event(self) -> None:
        if self.open_event is None:
            tmsp = round(time.time() * 1000)
            self.events[self.next_id] = UpDownEvent(self.next_id, 0, 0, tmsp, None)
            self.open_event = self.next_id
            self.next_id += 1 

    def update_open_event(self, count_up: int, count_down: int) -> None:
        if self.open_event is not None:
            self.events[self.open_event].count_up += count_up
            self.events[self.open_event].count_down += count_down
    
    def close_open_event(self) -> UpDownEvent:
        if self.open_event is not None:
            tmsp = round(time.time() * 1000)
            id = self.open_event
            self.open_event = None
            self.events[id].ended_at = tmsp
            return self.events[id]
        return None

    def has_open_event(self) -> bool:
        return self.open_event is not None

    def get_open_event(self) -> UpDownEvent:
        if self.open_event is not None:
            return self.events[self.open_event]
        return None

    