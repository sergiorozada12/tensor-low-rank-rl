import numpy as np

from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize


class CustomPendulumEnv(PendulumEnv):
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        reward = 1 - angle_normalize(th) ** 2 + .1 * thdot ** 2 + (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        done = True if ((newth > np.pi / 4) | (newth < -np.pi / 4)) else False
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.state = [np.random.rand()/100, np.random.rand()/100]
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot])