import gym
import highway_env


class CustomHighwayEnv:
    """
    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    """
    def __new__(cls):
        env = gym.make("highway-v0")

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
                "normalize": False,
                "order": "sorted",
            },
            "lanes_count": 3,
            "vehicles_count": 10,
            "simulation_frequency": 5,
            "duration": 50,
            "collision_reward": -1.0,    # -1 The reward received when colliding with a vehicle.
            "right_lane_reward": 0,  # 0.1 The reward received when driving on the right-most lanes, zero for other lanes.
            "high_speed_reward": 0.4,  # 0.4 The reward driving at full speed, zero for lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [10, 30],
            "normalize_reward": False, # Does not work
            "disable_collision_checks":  True,
            "initial_lane_id": 0,
            "vehicle_density": 1,
            "other_vehicles_type": 'highway_env.vehicle.behavior.IDMVehicle',
            "ego_spacing": 2,
        })

        return env
