from numpy import pi
import numpy as np

from gym.envs.classic_control.acrobot import AcrobotEnv, wrap, bound, rk4


class CustomAcrobotEnv(AcrobotEnv):
    def step(self, a):
        s = self.state
        torque = a

        s_augmented = np.append(s, torque)
        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        terminal = self._terminal()
        reward = -0.1 if not terminal else 100.
        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([s[0], s[1], s[2], s[3]])

    def reset(self):
        s, _ = super().reset()
        return s
