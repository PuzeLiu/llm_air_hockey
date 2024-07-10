import numpy as np
from air_hockey_challenge.environments.iiwas import AirHockeySingle 
from air_hockey_challenge.environments.position_control_wrapper import PositionControlIIWA
from air_hockey_challenge.utils.kinematics import inverse_kinematics


class AirHockeyHit(AirHockeySingle):
    def __init__(self, gamma=0.99, horizon=500, viewer_params=..., **kwargs):
        super().__init__(gamma, horizon, viewer_params, **kwargs)
        self.puck_init_pos = np.array([-0.71, 0.0])
        self.puck_init_vel = np.array([0.0, 0.0])
        self.hit_range = np.array([[-0.65, -0.25], [-0.4, 0.4]])  # Table Frame

    def setup(self, obs):
        # Initial position of the puck
        # puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]
        puck_pos = self.puck_init_pos

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        self._write_data("puck_x_vel", self.puck_init_vel[0])
        self._write_data("puck_y_vel", self.puck_init_vel[1])

        # if self.moving_init:
        #     lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        #     angle = np.random.uniform(-np.pi / 2 - 0.1, np.pi / 2 + 0.1)
        #     puck_vel = np.zeros(3)
        #     puck_vel[0] = -np.cos(angle) * lin_vel
        #     puck_vel[1] = np.sin(angle) * lin_vel
        #     puck_vel[2] = np.random.uniform(-2, 2, 1)

        #     self._write_data("puck_x_vel", puck_vel[0])
        #     self._write_data("puck_y_vel", puck_vel[1])
        #     self._write_data("puck_yaw_vel", puck_vel[2])

        super().setup(obs)

    def reward(self, obs, action, next_obs, absorbing):
        return 0
    
    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        # Stop if the puck bounces back on the opponents wall
        if puck_pos[0] > 0 and puck_vel[0] < 0:
            return True
        return super(AirHockeyHit, self).is_absorbing(obs)



class IiwaPositionHit(PositionControlIIWA, AirHockeyHit):
    def __init__(self, interpolation_order, opponent_agent=None, opponent_interp_order=-1, *args, **kwargs):
        super().__init__(interpolation_order=interpolation_order, *args, **kwargs)

        # Use default agent when none is provided
        if opponent_agent is None:
            self._opponent_agent_gen = self._default_opponent_action_gen()
            self._opponent_agent = lambda obs: next(self._opponent_agent_gen)

    def setup(self, obs):
        super().setup(obs)
        self._opponent_agent_gen = self._default_opponent_action_gen()

    def _default_opponent_action_gen(self):
        vel = 3
        t = np.pi / 2
        cart_offset = np.array([0.65, 0])
        prev_joint_pos = self.init_state

        while True:
            t += vel * self.dt
            cart_pos = np.array([0.1, 0.16]) * np.array([np.sin(t) * np.cos(t), np.cos(t)]) + cart_offset

            success, joint_pos = inverse_kinematics(self.env_info['robot']['robot_model'],
                                                    self.env_info['robot']['robot_data'],
                                                    np.concatenate(
                                                        [cart_pos, [0.1 + self.env_info['robot']['universal_height']]]),
                                                    initial_q=prev_joint_pos)
            assert success

            joint_vel = (joint_pos - prev_joint_pos) / self.dt

            prev_joint_pos = joint_pos

            yield np.vstack([joint_pos, joint_vel])
