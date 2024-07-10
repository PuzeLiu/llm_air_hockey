import numpy as np
from baseline.baseline_agent.tactics import SystemState, Tactic, TACTICS
from baseline.baseline_agent.trajectory_generator import TrajectoryGenerator


class SmashInstruct(Tactic):
    def __init__(self, env_info, agent_params, state: SystemState, trajectory_generator: TrajectoryGenerator):
        super().__init__(env_info, agent_params, state, trajectory_generator)
        self.hit_vel_mag = self.agent_params['max_hit_velocity']
        self.q_anchor_pos = self.agent_params['joint_anchor_pos']
        self.instructions = {}

    def _update_tactic_impl(self):
        if not self.can_smash() or self.state.tactic_finish:
            self.switch_count += 1
        else:
            self.switch_count = 0

        if self.switch_count > 4 and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def set_instruction(self, instructions):
        self.instructions = instructions

    def clear_instruction(self):
        self.instructions = {}

    def ready(self):
        if self.state.is_new_tactic:
            self.plan_new_trajectory = True
            self.state.tactic_finish = False
            self.state.predicted_time = self.agent_params['max_prediction_time']
            self.replan_time = 0
            self.hit_vel_mag = self.agent_params['max_hit_velocity']
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params['joint_anchor_pos']
            return True
        else:
            if len(self.state.trajectory_buffer) == 0 and self.can_smash():
                return True
            return False

    def apply(self):
        self.state.is_new_tactic = False

        goal_pos = np.array([2.49, 0.0])
        puck_pos_2d = self.state.predicted_state[:2]

        if 'hit_angle' in self.instructions.keys():
            hit_angle = np.clip(self.instructions['hit_angle'], -np.pi/2, np.pi/2)
            hit_dir_2d = np.array([np.cos(hit_angle), np.sin(hit_angle)])
        else:
            hit_dir_2d = goal_pos - puck_pos_2d
        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)

        if 'hit_velocity' in self.instructions.keys():
            hit_vel_mag = np.clip(self.instructions['hit_velocity'], 0, 1.0)
        else:
            hit_vel_mag = self.hit_vel_mag
        
        hit_vel_2d = hit_dir_2d * hit_vel_mag

        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (self.env_info['mallet']['radius'])
        self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params['max_plan_steps'])
            elif self.generator.bezier_planner.t_final >= self.replan_time:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                                            self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.maximum(0, self.state.predicted_time)
                return

            self.state.predicted_time += 2 * self.generator.dt
            self.hit_vel_mag *= 0.9
            hit_vel_2d = hit_dir_2d * self.hit_vel_mag
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True