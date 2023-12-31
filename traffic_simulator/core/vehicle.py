import uuid
import numpy as np

class Vehicle:
    def __init__(self, config={}):
        # Set default config
        self.set_default_config()

        # Update config
        for attr, val in config.items():
            setattr(self, attr, val)

        self.init_properties()

    def init_properties(self):
        self.sqrt_ab = 2*np.sqrt(self.a_max*self.b_max)
    
    def set_default_config(self):
        self.id = uuid.uuid4()

        self.l = 4
        self.s0 = 4
        self.T = 1
        self.v_max = 16.6
        self.a_max = 1.44
        self.b_max = 4.61

        self.path = []
        self.current_road_index = 0

        self.x = 0
        self.v = 0
        self.a = 0

        self.stopped = False

    def update(self, lead, dt):
        # Update position and velocity
        if self.v + self.a*dt < 0:
            self.x -= (1/2)*self.v**2/self.a
            self.v = 0
        else:
            self.v += self.a*dt
            self.x += self.v*dt + (1/2)*self.a*dt**2

        alpha = 0
        if lead:
            delta_x = lead.x - self.x - lead.l
            delta_v = self.v - lead.v
            alpha = (self.s0 + max(0, self.T*self.v+ delta_v*self.v/self.sqrt_ab)) / delta_x

        self.a = self.a_max*(1 - (self.v/self.v_max)**4 - alpha**2) 

        if self.stopped:
            self.a = -self.b_max*self.v/self.v_max
        
