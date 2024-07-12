import numpy as np
import json


class AgentParams:
    def __init__(self, env_info) -> None:

        self._x_init = np.array([0.65, 0., env_info['robot']['ee_desired_height'] + 0.2])
        self._x_home = np.array([0.65, 0., env_info['robot']['ee_desired_height']])
        self._max_hit_velocity = 1.2
        self._joint_anchor_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])

        self.switch_tactics_min_steps: int = 15
        self.max_prediction_time: float = 1.0
        self.max_plan_steps: int = 5
        self.static_vel_threshold: float = 0.4
        self.transversal_vel_threshold: float = 0.1
        self.default_linear_vel: float = 0.6
        self.hit_range: np.ndarray = np.array([0.8, 1.3])
        self.defend_range: np.ndarray = np.array([0.8, 1.0])
        self.defend_width: np.ndarray = np.array(0.45)
        self.prepare_range: np.ndarray = np.array([0.8, 1.3])
        self.static_count_threshold: int = 3
        self.puck_approaching_count_threshold: int = 3
        self.puck_transversal_moving_count_threshold: int = 3

    def parse_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        
        for key, value in data.items():
            if hasattr(self, key):
                # If the attribute exists and is an ndarray, convert the value to an ndarray
                if isinstance(getattr(self, key), np.ndarray):
                    setattr(self, key, np.array(value))
            else:
                setattr(self, key, value)
        

    @property
    def x_init(self):
        return self._x_init

    @property
    def x_home(self):
        return self._x_home

    @property
    def max_hit_velocity(self):
        return self._max_hit_velocity

    @property
    def joint_anchor_pos(self):
        return self._joint_anchor_pos
