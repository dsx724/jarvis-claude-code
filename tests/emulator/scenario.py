"""ScenarioDriver: loads YAML scenarios, coordinates mocks, records events."""

import os
import threading

import yaml


class ScenarioDriver:
    """Coordinates mock behavior according to a YAML scenario definition.

    Tracks the current interaction index, provides scripted data to mocks,
    and records events for post-run validation.
    """

    def __init__(self, scenario_path):
        with open(scenario_path) as f:
            self.scenario = yaml.safe_load(f)
        self.name = self.scenario.get("name", os.path.basename(scenario_path))
        self.time_scale = self.scenario.get("time_scale", 1.0)
        self.interactions = self.scenario.get("interactions", [])
        self.termination = self.scenario.get("termination", "exit")

        self._interaction_index = 0
        self._lock = threading.Lock()
        self._events = []  # List of (event_type, data) tuples

    def current_interaction(self):
        """Return the current interaction dict, or None if exhausted."""
        with self._lock:
            if self._interaction_index < len(self.interactions):
                return self.interactions[self._interaction_index]
            return None

    def advance(self):
        """Move to the next interaction."""
        with self._lock:
            self._interaction_index += 1

    def is_exhausted(self):
        """Return True if all interactions have been consumed."""
        with self._lock:
            return self._interaction_index >= len(self.interactions)

    def record_event(self, event_type, data=None):
        """Record an event for post-run validation."""
        with self._lock:
            self._events.append({
                "type": event_type,
                "interaction_index": self._interaction_index,
                "data": data or {},
            })

    @property
    def events(self):
        """Return a copy of recorded events."""
        with self._lock:
            return list(self._events)

    def events_for_interaction(self, index):
        """Return events recorded during a specific interaction."""
        return [e for e in self.events if e["interaction_index"] == index]
