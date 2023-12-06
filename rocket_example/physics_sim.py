import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

import quaternion

from utils import *

class State():
    def __init__(self, u):
        self.u = u
        self.t = 0
        self.m_rb = 18.88 # [kg]
        self.m_tot = self.m_rb + mass(self.t)
        self.roll = 0
    
    def _update_time_dependent_variables(self, dt):
        self.t = self.t + dt
        self.m_tot = self.m_rb + mass(self.t)
    
    def update_state(self, dt, forces=np.zeros(3),roll_torque=0):
        self._update_time_dependent_variables(dt)

        pos = self.u[:3]
        vel = self.u[3:6]
        acc = self.u[6:9]
        q = quaternion.from_float_array(self.u[9:13])
        q /= np.sqrt(q * q.conjugate())
        w = self.u[13:16]
        
        

        # # decide if rocket is still on rail
        # if np.linalg.norm(pos) < 3:
        #     #position changes in one degree of freedom
        #     pos += np.dot(np.array([0,0,1]), vel) * dt
        # else:
        pos += vel * dt
        pos += acc * dt**2 / 2

        vel += acc * dt

        acc = forces / self.m_tot * dt

        # update roll
        self.roll += roll_torque * dt

        up_vec = np.array([0,0,1])
        
        if np.linalg.norm(vel) > 1e-5:
          up_vec = vel / np.linalg.norm(vel)

        # Yes this is a constant, this is wrong
        fwrd_vec = np.array([0,1,0])
        right_vec = -np.cross(up_vec, fwrd_vec)
        right_vec = right_vec / np.linalg.norm(right_vec)

        fwrd_vec = np.cross(up_vec, right_vec)

        # Create the rotation matrix from rocket to world frame
        R = np.vstack((right_vec, fwrd_vec, up_vec))

        # Create the rotation matrix from world to rocket frame
        world_to_rocket = np.linalg.inv(R)

        q_prime = quaternion.from_rotation_matrix(world_to_rocket)
        q_prime /= np.sqrt(q_prime * q_prime.conjugate())

        delta_q = q_prime / q

        # compute the angular velocity in the world frame
        w = quaternion.as_vector_part(delta_q / dt)

        # update u
        self.u = np.hstack([pos, vel, acc, quaternion.as_float_array(q_prime), w])
