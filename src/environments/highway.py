import gym
import highway_env


class CustomHighwayEnv:
    def __new__(cls):
        env = gym.make("highway-v0")
        env.env.config['lanes_count'] = 3

        env.configure({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 3,
                "features": ["x", "y", "vx"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted"
            }
        })

        return env
