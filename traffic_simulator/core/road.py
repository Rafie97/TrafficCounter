from scipy.spatial import distance
from collections import deque

class Road:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.vehicles = deque()
        self.init_properties()

    def init_properties(self):
        self.length = distance.euclidean(self.start, self.end)
        self.angle_sin = (self.end[1] - self.start[1]) / self.length
        self.angle_cos = (self.end[0] - self.start[0]) / self.length
        self.has_traffic_signal = False

    def set_traffic_signal(self, signal, group):
        self.traffic_signal = signal
        self.has_traffic_signal = True
        self.traffic_signal_group = group

    @property
    def traffic_signal_state(self):
        if self.has_traffic_signal:
            i = self.traffic_signal_group
            return self.traffic_signal.current_cycle[i]
        return True
    
    def update(self, dt):
        n = len(self.vehicles)
        if n > 0:
            # Update first vehicle
            self.vehicles[0].update(None, dt)
            # Update other vehicles
            for i in range(1, n):
                self.vehicles[i].update(self.vehicles[i-1], dt)
            
            # Check for traffic signal
            if self.traffic_signal_state:
                # Check for collision
                for i in range(1, n):
                    if self.vehicles[i].x - self.vehicles[i].l < self.vehicles[i-1].x:
                        self.vehicles[i].stopped = True
                        self.vehicles[i].a = 0
                    else:
                        self.vehicles[i].stopped = False

            else:
                # Signal is red
                if self.vehicles[0].x >= self.length - self.traffic_signal_slow_distance:
                    self.vehicles[0].stopped = True
                    self.vehicles[0].a = 0
                    self.vehicles[0].v = 0
        