import numpy as np

from .agent_base import AgentBase
# from baseline.baseline_agent.tactics import *
from .tactic_smash_instruct import SmashInstruct
from .agent_params import AgentParams
from .system_state import SystemState
from .trajectory_generator import TrajectoryGenerator
from .tactics import Init, Ready, Prepare, Defend, Repel, TACTICS


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """
    if "hit" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1)
    if "defend" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="defend")
    if "prepare" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="prepare")

    return BaselineAgent(env_info, **kwargs)


class BaselineAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, only_tactic=None, **kwargs):
        super(BaselineAgent, self).__init__(env_info, agent_id, **kwargs)

        self.agent_params = AgentParams(env_info)

        self.state = SystemState(self.env_info, agent_id, self.agent_params)
        self.traj_generator = TrajectoryGenerator(self.env_info, self.agent_params, self.state)

        self.tactics_processor = [Init(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  Ready(self.env_info, self.agent_params, self.state, self.traj_generator,
                                        only_tactic=only_tactic),
                                  Prepare(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  Defend(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  Repel(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  SmashInstruct(self.env_info, self.agent_params, self.state, self.traj_generator)]

    def reset(self):
        self.state.reset()
        self.tactics_processor[TACTICS.SMASH.value].clear_instruction()

    def set_instruction(self, instructions):
        self.tactics_processor[TACTICS.SMASH.value].set_instruction(instructions)

    def draw_action(self, obs):
        self.state.update_observation(self.get_joint_pos(obs), self.get_joint_vel(obs), self.get_puck_pos(obs))

        while True:
            self.tactics_processor[self.state.tactic_current.value].update_tactic()
            activeTactic = self.tactics_processor[self.state.tactic_current.value]

            if activeTactic.ready():
                activeTactic.apply()
            if len(self.state.trajectory_buffer) > 0:
                break
            else:
                # print("iterate")
                pass

        self.state.q_cmd, self.state.dq_cmd = self.state.trajectory_buffer[0]
        self.state.trajectory_buffer = self.state.trajectory_buffer[1:]

        self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(self.state.q_cmd, self.state.dq_cmd)
        return np.vstack([self.state.q_cmd, self.state.dq_cmd])
