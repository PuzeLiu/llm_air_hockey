import copy
import numpy as np
import torch
import json
import mujoco
from enum import Enum
import scipy
import scipy.linalg
from scipy import sparse
from scipy.interpolate import CubicSpline
import nlopt
import osqp
import time
from mushroom_rl.core.agent import Agent
from air_hockey_challenge.environments.iiwas import AirHockeySingle
from air_hockey_challenge.environments.position_control_wrapper import PositionControlIIWA


### kinematics.py ###
def forward_kinematics(mj_model, mj_data, q, link="ee"):
    """
    Compute the forward kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (np.array):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]

    Returns
    -------
    position: numpy.ndarray, (3,)
        Position of the link in robot's base frame
    orientation: numpy.ndarray, (3, 3)
        Orientation of the link in robot's base frame
    """

    return _mujoco_fk(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


def inverse_kinematics(mj_model, mj_data, desired_position, desired_rotation=None, initial_q=None, link="ee"):
    """
    Compute the inverse kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is
        the case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        desired_position (numpy.ndarray, (3,)):
            The desired position of the selected link.
        desired_rotation (optional, numpy.array, (3,3)):
            The desired rotation of the selected link.
        initial_q (numpy.ndarray, None):
            The initial configuration of the algorithm, if set to None it will take the initial configuration of the
            mj_data.
        link (str, "ee"):
            Link for which the inverse kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]
    """
    q_init = np.zeros(mj_model.nq)
    if initial_q is None:
        q_init = mj_data.qpos
    else:
        q_init[:initial_q.size] = initial_q

    q_l = mj_model.jnt_range[:, 0]
    q_h = mj_model.jnt_range[:, 1]
    lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
    upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

    desired_quat = None
    if desired_rotation is not None:
        desired_quat = np.zeros(4)
        mujoco.mju_mat2Quat(desired_quat, desired_rotation.reshape(-1, 1))

    return _mujoco_clik(desired_position, desired_quat, q_init, link_to_xml_name(mj_model, link), mj_model,
                        mj_data, lower_limit, upper_limit)


def jacobian(mj_model, mj_data, q, link="ee"):
    """
    Compute the Jacobian of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (numpy.ndarray):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]

    Returns
    -------
    numpy.ndarray, (6, num_joints):
        The Jacobian matrix for the robot kinematics.
    """
    return _mujoco_jac(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


def link_to_xml_name(mj_model, link):
    try:
        mj_model.body('iiwa_1/base')
        link_to_frame_idx = {
            "1": "iiwa_1/link_1",
            "2": "iiwa_1/link_2",
            "3": "iiwa_1/link_3",
            "4": "iiwa_1/link_4",
            "5": "iiwa_1/link_5",
            "6": "iiwa_1/link_6",
            "7": "iiwa_1/link_7",
            "ee": "iiwa_1/striker_joint_link",
        }
    except:
        link_to_frame_idx = {
            "1": "planar_robot_1/body_1",
            "2": "planar_robot_1/body_2",
            "3": "planar_robot_1/body_3",
            "ee": "planar_robot_1/body_ee",
        }
    return link_to_frame_idx[link]


def _mujoco_fk(q, name, model, data):
    data.qpos[:len(q)] = q
    mujoco.mj_fwdPosition(model, data)
    return data.body(name).xpos.copy(), data.body(name).xmat.reshape(3, 3).copy()


def _mujoco_jac(q, name, model, data):
    data.qpos[:len(q)] = q
    dtype = data.qpos.dtype
    jac = np.empty((6, model.nv), dtype=dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)
    return jac


def _mujoco_clik(desired_pos, desired_quat, initial_q, name, model, data, lower_limit, upper_limit):
    IT_MAX = 1000
    eps = 1e-4
    damp = 1e-3
    progress_thresh = 20.0
    max_update_norm = 0.1
    rot_weight = 1
    i = 0

    dtype = data.qpos.dtype

    data.qpos = initial_q

    neg_x_quat = np.empty(4, dtype=dtype)
    error_x_quat = np.empty(4, dtype=dtype)

    if desired_pos is not None and desired_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if desired_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif desired_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("Desired Position and desired rotation is None, cannot compute inverse kinematics")

    while True:
        # forward kinematics
        mujoco.mj_fwdPosition(model, data)

        x_pos = data.body(name).xpos
        x_quat = data.body(name).xquat

        error_norm = 0
        if desired_pos is not None:
            err_pos[:] = desired_pos - x_pos
            error_norm += np.linalg.norm(err_pos)

        if desired_quat is not None:
            mujoco.mju_negQuat(neg_x_quat, x_quat)
            mujoco.mju_mulQuat(error_x_quat, desired_quat, neg_x_quat)
            mujoco.mju_quat2Vel(err_rot, error_x_quat, 1)
            error_norm += np.linalg.norm(err_rot) * rot_weight

        if error_norm < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)

        hess_approx = jac.T.dot(jac)
        joint_delta = jac.T.dot(err)

        hess_approx += np.eye(hess_approx.shape[0]) * damp
        update_joints = np.linalg.solve(hess_approx, joint_delta)

        update_norm = np.linalg.norm(update_joints)

        # Check whether we are still making enough progress, and halt if not.
        progress_criterion = error_norm / update_norm
        if progress_criterion > progress_thresh:
            success = False
            break

        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        mujoco.mj_integratePos(model, data.qpos, update_joints, 1)
        data.qpos = np.clip(data.qpos, lower_limit, upper_limit)
        i += 1
    q_cur = data.qpos.copy()

    return success, q_cur


### agent_base.py ###
class AgentBase(Agent):
    def __init__(self, env_info, agent_id=1, **kwargs):
        """
        Initialization of the Agent.

        Args:
            env_info [dict]:
                A dictionary contains information about the environment;
            agent_id [int, default 1]:
                1 by default, agent_id will be used for the tournament;
            kwargs [dict]:
                A dictionary contains agent related information.

        """
        super().__init__(env_info['rl_info'], None)
        self.env_info = env_info
        self.agent_id = agent_id
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])

        self._add_save_attr(
            env_info='none',
            agent_id='none',
            robot_model='none',
            robot_data='none',
        )

    def reset(self):
        """
        Reset the agent

        Important:
            To be implemented

        """
        raise NotImplementedError

    def draw_action(self, observation):
        """ Draw an action, i.e., desired joint position and velocity, at every time step.

        Args:
            observation (ndarray): Observed state including puck's position/velocity, joint position/velocity,
                opponent's end-effector position (if applicable).

        Returns:
            numpy.ndarray, (2, num_joints): The desired [Positions, Velocities] of the next step

        Important:
            To be implemented

        """

        raise NotImplementedError

    def episode_start(self):
        self.reset()

    @classmethod
    def load_agent(cls, path, env_info, agent_id=1):
        """ Load the Agent

        Args:
            path (Path, str): Path to the object
            env_info (dict): A dictionary parsed from the AirHockeyChallengeWrapper
            agent_id (int, default 1): will be specified for two agents game

        Returns:
            Returns the loaded agent

        """
        agent = cls.load(path)

        agent.env_info = env_info
        agent.agent_id = agent_id
        agent.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        agent.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        return agent

    def get_puck_state(self, obs):
        """
        Get the puck's position and velocity from the observation

        Args
        ----
        obs: numpy.ndarray
            observed state.

        Returns
        -------
        joint_pos: numpy.ndarray, (3,)
            [x, y, theta] position of the puck w.r.t robot's base frame
        joint_vel: numpy.ndarray, (3,)
            [vx, vy, dtheta] position of the puck w.r.t robot's base frame

        """
        return self.get_puck_pos(obs), self.get_puck_vel(obs)

    def get_joint_state(self, obs):
        """
        Get the joint positions and velocities from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        joint_pos: numpy.ndarray
            joint positions of the robot;
        joint_vel: numpy.ndarray
            joint velocities of the robot.

        """
        return self.get_joint_pos(obs), self.get_joint_vel(obs)

    def get_puck_pos(self, obs):
        """
        Get the Puck's position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's position of the robot

        """
        return obs[self.env_info['puck_pos_ids']]

    def get_puck_vel(self, obs):
        """
        Get the Puck's velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's velocity of the robot

        """
        return obs[self.env_info['puck_vel_ids']]

    def get_joint_pos(self, obs):
        """
        Get the joint position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint position of the robot

        """
        return obs[self.env_info['joint_pos_ids']]

    def get_joint_vel(self, obs):
        """
        Get the joint velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint velocity of the robot

        """
        return obs[self.env_info['joint_vel_ids']]

    def get_ee_pose(self, obs):
        """
        Get the End-Effector's Position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            opponent's end-effector's position

        """
        return forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(obs))


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


### kalman_filter.py ###
B_PARAMS = np.array([8.40683102e-01, 0, 7.71445220e-04])
N_PARAMS = np.array([0., -0.79, 0.])
THETA_PARAMS = np.array([-6.48073315, 6.32545305, 0.8386719])
DAMPING = np.array([0.2125, 0.2562])
LIN_COV = np.diag([8.90797655e-07, 5.49874493e-07, 2.54163138e-04, 3.80228296e-04, 7.19007035e-02, 1.58019149e+00])
COL_COV = \
    np.array([[0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.],
              [0., 0., 2.09562546e-01, 3.46276805e-02, 0., -1.03489604e+00],
              [0., 0., 3.46276805e-02, 9.41218351e-02, 0., -1.67029496e+00],
              [0., 0., 0., 0., 0., 0.],
              [0., 0., -1.03489604e+00, -1.67029496e+00, 0., 1.78037877e+02]])

OBS_COV = np.diag([5.0650402e-07, 8.3995428e-07, 1.6572967e-03])


class SystemModel:
    def __init__(self, env_info, agent_id):
        self.puck_radius = env_info['puck']['radius']
        self.mallet_radius = env_info['mallet']['radius']
        self.dt = env_info['dt']

        self.table = AirHockeyTable(env_info['table']['length'], env_info['table']['width'],
                                    env_info['table']['goal_width'], env_info['puck']['radius'],
                                    abs(env_info['robot']['base_frame'][agent_id - 1][0, 3]), env_info['dt'])
        self.F = np.eye(6)
        self.F_linear = np.eye(6)
        self.F_linear[0, 2] = self.F_linear[1, 3] = self.F_linear[4, 5] = self.dt
        self.F_linear[2, 2] = 1 - self.dt * DAMPING[0]
        self.F_linear[3, 3] = 1 - self.dt * DAMPING[1]
        self.Q_collision = np.zeros((6, 6))
        self.has_collision = False
        self.outside_boundary = False
        self.score = False

    def f(self, x):
        self.has_collision, self.outside_boundary, self.score, F, Q = self.table.check_collision(x)
        if self.has_collision:
            # Collision Dynamics
            self.F = F
            self.Q_collision = Q
        elif self.outside_boundary or self.score:
            # Stop Moving
            self.F = np.eye(6)
            self.Q_collision = np.zeros((6, 6))
        else:
            # Normal Prediction
            self.F = self.F_linear
        x = self.F @ x
        x[4] = (x[4] + np.pi) % (np.pi * 2) - np.pi
        return x


class AirHockeyTable:
    def __init__(self, length, width, goal_width, puck_radius, x_offset, dt):
        self.table_length = length
        self.table_width = width
        self.goal_width = goal_width
        self.puck_radius = puck_radius
        self.x_offset = x_offset
        self.dt = dt

        pos_offset = np.array([x_offset, 0])
        p1 = np.array([-length / 2 + puck_radius, -width / 2 + puck_radius]) + pos_offset
        p2 = np.array([length / 2 - puck_radius, -width / 2 + puck_radius]) + pos_offset
        p3 = np.array([length / 2 - puck_radius, width / 2 - puck_radius]) + pos_offset
        p4 = np.array([-length / 2 + puck_radius, width / 2 - puck_radius]) + pos_offset

        self.boundary = np.array([[p1, p2],
                                  [p2, p3],
                                  [p3, p4],
                                  [p4, p1]])

        self.local_rim_transform = np.zeros((4, 6, 6))
        self.local_rim_transform_inv = np.zeros((4, 6, 6))
        transform_tmp = np.eye(6)
        self.local_rim_transform[0] = transform_tmp.copy()
        self.local_rim_transform_inv[0] = transform_tmp.T.copy()

        transform_tmp = np.zeros((6, 6))
        transform_tmp[0, 1] = transform_tmp[2, 3] = transform_tmp[4, 4] = transform_tmp[5, 5] = 1
        transform_tmp[1, 0] = transform_tmp[3, 2] = -1
        self.local_rim_transform[1] = transform_tmp.copy()
        self.local_rim_transform_inv[1] = transform_tmp.T.copy()

        transform_tmp = np.eye(6)
        transform_tmp[0, 0] = transform_tmp[1, 1] = transform_tmp[2, 2] = transform_tmp[3, 3] = -1
        self.local_rim_transform[2] = transform_tmp.copy()
        self.local_rim_transform_inv[2] = transform_tmp.T.copy()

        transform_tmp = np.zeros((6, 6))
        transform_tmp[1, 0] = transform_tmp[3, 2] = transform_tmp[4, 4] = transform_tmp[5, 5] = 1
        transform_tmp[0, 1] = transform_tmp[2, 3] = -1
        self.local_rim_transform[3] = transform_tmp.copy()
        self.local_rim_transform_inv[3] = transform_tmp.T.copy()

        self._F_precollision = np.eye(6)
        self._F_postcollision = np.eye(6)
        self._jac_local_collision = np.eye(6)
        self._jac_local_collision[2, [2, 3, 5]] = B_PARAMS[0:3]
        self._jac_local_collision[3, [2, 3, 5]] = N_PARAMS[0:3]
        self._jac_local_collision[5, [2, 3, 5]] = THETA_PARAMS[0:3]

    def check_collision(self, state):
        score = False
        outside_boundary = False
        collision = False

        u = state[2:4] * self.dt
        if np.abs(state[1]) < self.goal_width / 2:
            if state[0] + u[0] < -self.boundary[0, 0, 0] or state[0] + u[0] > self.boundary[0, 1, 0]:
                score = True
        elif np.any(state[:2] < self.boundary[0, 0]) or np.any(state[:2] > self.boundary[1, 1]):
            outside_boundary = True

        if not score and not outside_boundary:
            F, Q_collision, collision = self._check_collision_impl(state, u)

        else:
            F = np.eye(4)
            Q_collision = np.zeros((4, 6))
        return collision, outside_boundary, score, F, Q_collision

    def _cross_2d(self, u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    def _check_collision_impl(self, state, u):
        F = np.eye(4)
        Q_collision = np.zeros((4, 6))
        v = self.boundary[:, 1] - self.boundary[:, 0]
        w = self.boundary[:, 0] - state[:2]
        denominator = self._cross_2d(v, u)
        s = self._cross_2d(v, w) / (denominator + 1e-6)
        r = self._cross_2d(u, w) / (denominator + 1e-6)
        collide_idx = np.where(np.logical_and(np.logical_and(1e-6 < s, s < 1 - 1e-6),
                               np.logical_and(1e-6 < r, r < 1 - 1e-6)))[0]
        collision = False

        if len(collide_idx) > 0:
            collision = True
            collide_rim_idx = collide_idx[0]
            s_i = s[collide_rim_idx]
            self._F_precollision[0][2] = self._F_precollision[1][3] = self._F_precollision[4][5] = s_i * self.dt
            self._F_postcollision[0][2] = self._F_postcollision[1][3] = self._F_postcollision[4][5] = (
                1 - s_i) * self.dt
            state_local = self.local_rim_transform[collide_rim_idx] @ state
            # Compute the slide direction
            slide_dir = 1 if state_local[2] + state_local[5] * self.puck_radius >= 0 else -1

            jac_local_collision = self._jac_local_collision.copy()
            jac_local_collision[2, 3] *= slide_dir
            jac_local_collision[5, 3] *= slide_dir

            F_collision = self.local_rim_transform_inv[collide_rim_idx] @ jac_local_collision @ self.local_rim_transform[collide_rim_idx]
            F = self._F_postcollision @ F_collision @ self._F_precollision
            Q_collision = self.local_rim_transform_inv[collide_rim_idx] @ COL_COV @ self.local_rim_transform_inv[collide_rim_idx].T
        return F, Q_collision, collision


class PuckTracker:
    def __init__(self, env_info, agent_id=1):
        self.system = SystemModel(env_info, agent_id)
        self.Q = LIN_COV

        self.R = OBS_COV
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 4] = 1

        self.state = None
        self.P = None

    def reset(self, puck_pos):
        self.P = np.eye(6)
        self.state = np.zeros(6)
        self.state[[0, 1, 4]] = puck_pos

    def predict(self, state, P):
        predicted_state = self.system.f(state)
        if self.system.has_collision:
            Q = self.system.Q_collision
        elif self.system.outside_boundary or self.system.score:
            Q = self.system.Q_collision
        else:
            Q = self.Q
        P = self.system.F @ P @ self.system.F.T + Q
        return predicted_state, P

    def update(self, measurement, predicted_state, P):
        xy_innovation = measurement[:2] - predicted_state[:2]
        theta_innovation = (measurement[2] - predicted_state[4] + np.pi) % (np.pi * 2) - np.pi
        y = np.concatenate([xy_innovation, [theta_innovation]])
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        state = predicted_state + K @ y
        P = (np.eye(6) - K @ self.H) @ P
        return state, P

    def step(self, measurement):
        predicted_state, P = self.predict(self.state, self.P)
        # self.state = self.predicted_state
        self.state, self.P = self.update(measurement, predicted_state, P)

    def get_prediction(self, t, defend_line=0.):
        P_current = self.P.copy()
        state_current = self.state.copy()
        predict_time = 0

        for i in range(round(t / self.system.dt)):
            state_next, P_next = self.predict(state_current, P_current)
            if state_next[0] < defend_line:
                break
            if np.linalg.norm(state_current[2:4]) < 1e-2 and np.linalg.norm(state_next[2:4]) < 1e-2:
                predict_time = t
                break
            predict_time += self.system.dt
            state_current = state_next
            P_current = P_next
        return state_current, P_current, predict_time


### system_state.py ###
class TACTICS(Enum):
    __order__ = "INIT READY PREPARE DEFEND REPEL SMASH N_TACTICS"
    INIT = 0
    READY = 1
    PREPARE = 2
    DEFEND = 3
    REPEL = 4
    SMASH = 5
    N_TACTICS = 6


class SystemState:
    def __init__(self, env_info, agent_id, agent_params: AgentParams):
        self.env_info = env_info
        self.agent_id = agent_id
        self.agent_params = agent_params
        self.puck_tracker = PuckTracker(self.env_info, agent_id)
        self.robot_model = copy.deepcopy(self.env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self.env_info['robot']['robot_data'])

        self.restart = True

        self.q_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.q_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.dq_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.dq_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.x_cmd = np.zeros(3)
        self.x_actual = np.zeros(3)

        self.v_cmd = np.zeros(3)
        self.v_actual = np.zeros(3)

        self.predicted_state = np.zeros(6)
        self.predicted_cov = np.eye(6)
        self.predicted_time = 0.
        self.estimated_state = np.zeros(6)

        self.tactic_current = TACTICS.READY
        self.is_new_tactic = True
        self.tactic_finish = True
        self.has_generated_stop_traj = False
        self.switch_tactics_count = self.agent_params.switch_tactics_min_steps
        self.puck_static_count = 0
        self.puck_approaching_count = 0
        self.puck_transversal_moving_count = 0

        self.smash_finish = False

        self.trajectory_buffer = list()

    def reset(self):
        self.restart = True

        self.q_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.q_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.dq_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.dq_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.x_cmd = np.zeros(3)
        self.x_actual = np.zeros(3)

        self.v_cmd = np.zeros(3)
        self.v_actual = np.zeros(3)

        self.predicted_state = np.zeros(6)
        self.predicted_cov = np.eye(6)
        self.predicted_time = 0.
        self.estimated_state = np.zeros(6)

        self.tactic_current = TACTICS.READY
        self.is_new_tactic = True
        self.tactic_finish = True
        self.has_generated_stop_traj = False
        self.plan_new_trajectory = True
        self.switch_tactics_count = self.agent_params.switch_tactics_min_steps
        self.puck_static_count = 0
        self.puck_approaching_count = 0
        self.puck_transversal_moving_count = 0

        self.smash_finish = False

        self.trajectory_buffer = list()

    def is_puck_static(self):
        return self.puck_static_count > self.agent_params.static_count_threshold

    def is_puck_approaching(self):
        return self.puck_approaching_count > self.agent_params.puck_approaching_count_threshold

    def is_puck_transversal_moving(self):
        return self.puck_transversal_moving_count > self.agent_params.puck_transversal_moving_count_threshold

    def update_observation(self, joint_pos_cur, joint_vel_cur, puck_state):
        if self.restart:
            self.puck_tracker.reset(puck_state)
            self.q_cmd = joint_pos_cur
            self.dq_cmd = joint_vel_cur
            self.x_cmd, self.v_cmd = self.update_ee_pos_vel(self.q_cmd, self.dq_cmd)
            self.restart = False

        self.q_actual = joint_pos_cur
        self.dq_actual = joint_vel_cur
        self.x_actual, self.v_actual = self.update_ee_pos_vel(self.q_actual, self.dq_actual)

        self.puck_tracker.step(puck_state)
        self.estimated_state = self.puck_tracker.state.copy()

        if np.linalg.norm(self.puck_tracker.state[2:4]) < self.agent_params.static_vel_threshold:
            self.puck_static_count += 1
            self.puck_approaching_count = 0
            self.puck_transversal_moving_count = 0
        else:
            self.puck_static_count = 0
            puck_dir = self.puck_tracker.state[2:4] / np.linalg.norm(self.puck_tracker.state[2:4])
            if np.abs(np.dot(puck_dir, np.array([1., 0]))) < 0.15:
                self.puck_transversal_moving_count += 1
                self.puck_approaching_count = 0
            else:
                if self.puck_tracker.state[2] < 0 and self.puck_tracker.state[0] > self.agent_params.defend_range[0]:
                    self.puck_approaching_count += 1

    def update_prediction(self, prediction_time, stop_line=0.):
        self.predicted_state, self.predicted_cov, self.predicted_time = \
            self.puck_tracker.get_prediction(prediction_time, stop_line)

    def update_ee_pos_vel(self, joint_pos, joint_vel):
        x_ee, _ = forward_kinematics(self.robot_model, self.robot_data, joint_pos)
        v_ee = jacobian(self.robot_model, self.robot_data, joint_pos)[:3,
                                                                      :self.env_info['robot']['n_joints']] @ joint_vel
        return x_ee, v_ee


### cubic_linear_planner.py ###
class CubicLinearPlanner:
    def __init__(self, n_joints, step_size):
        self.n_joints = n_joints
        self.step_size = step_size

    def plan(self, start_pos, start_vel, end_pos, end_vel, t_total):
        t_total = self._round_time(t_total)
        coef = np.array([[1, 0, 0, 0], [1, t_total, t_total ** 2, t_total ** 3],
                         [0, 1, 0, 0], [0, 1, 2 * t_total, 3 * t_total ** 2]])
        results = np.vstack([start_pos, end_pos, start_vel, end_vel])

        A = scipy.linalg.block_diag(*[coef] * start_pos.shape[-1])
        y = results.reshape(-1, order='F')

        weights = np.linalg.solve(A, y).reshape(start_pos.shape[-1], 4)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        t = np.linspace(self.step_size, t_total, int(t_total / self.step_size))

        x = weights[:, 0:1] + weights[:, 1:2] * t + weights[:, 2:3] * t ** 2 + weights[:, 3:4] * t ** 3
        dx = weights_d[:, 0:1] + weights_d[:, 1:2] * t + weights_d[:, 2:3] * t ** 2
        ddx = weights_dd[:, 0:1] + weights_dd[:, 1:2] * t
        return np.hstack([x.T, dx.T, ddx.T])

    def _round_time(self, time):
        return (round(time / self.step_size)) * self.step_size


### bezier_planner_new.py ###
class BezierPlanner:
    def __init__(self, boundary, step_size):
        self.boundary = boundary
        self.step_size = step_size
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None

        self.z0 = None
        self.z1 = None
        self.z2 = None
        self.z3 = None

        self.t_final = 0

    def compute_control_point(self, p_start, v_start, p_stop, v_stop, t_plan=None):
        h_01 = np.inf
        h_23 = np.inf
        for b in self.boundary:
            b_s = b[0]
            b_f = b[1]

            if np.linalg.norm(v_start) > 1e-3:
                A_1 = np.vstack([v_start, b_s - b_f]).T
                b_1 = b_s - p_start
                if np.linalg.det(A_1) != 0:
                    h_1 = np.linalg.solve(A_1, b_1)[0]
                    if h_1 > 0:
                        h_01 = np.minimum(h_1, h_01)
            else:
                h_01 = 0

            if np.linalg.norm(v_stop) > 1e-3:
                A_2 = np.vstack([-v_stop, b_s - b_f]).T
                b_2 = b_s - p_stop
                if np.linalg.det(A_2) != 0:
                    h_2 = np.linalg.solve(A_2, b_2)[0]
                    if h_2 > 0:
                        h_23 = np.minimum(h_2, h_23)
            else:
                h_23 = 0

        self.p0 = p_start.copy()
        self.p3 = p_stop.copy()

        dz_start = 0
        dz_stop = 0
        if h_01 == 0 and h_23 == 0:
            self.p1 = (p_start + v_start * h_01)
            self.p2 = (p_stop - v_stop * h_23)
        elif h_01 == 0:
            l_start = p_stop - v_stop * h_23
            min_length = np.minimum(0.15 / h_23, 1)
            self.p2 = self.get_closest_point_from_line_to_point(l_start, p_stop, p_start,
                                                                [0, 1 - min_length])
            self.p1 = p_start
            dz_stop = np.linalg.norm(v_stop) / np.linalg.norm(self.p3 - self.p2) / 3
        elif h_23 == 0:
            l_end = p_start + v_start * h_01
            min_length = np.minimum(0.15 / h_01, 1)
            self.p1 = self.get_closest_point_from_line_to_point(p_start, l_end, p_stop, [min_length, 1])
            self.p2 = p_stop
            dz_start = np.linalg.norm(v_start) / np.linalg.norm(self.p1 - self.p0) / 3
        else:
            l1_end = p_start + v_start * h_01
            l2_start = p_stop - v_stop * h_23
            min_length1 = np.minimum(0.15 / h_01, 1)
            min_length2 = np.minimum(0.15 / h_23, 1)
            self.p1, self.p2 = self.get_closest_point_between_line_segments(p_start, l1_end, l2_start, p_stop,
                                                                            np.array([[min_length1, 1],
                                                                                      [0, 1 - min_length2]]))
            self.p1 = self.p1
            self.p2 = self.p2
            dz_start = np.linalg.norm(v_start) / np.linalg.norm(self.p1 - self.p0) / 3
            dz_stop = np.linalg.norm(v_stop) / np.linalg.norm(self.p3 - self.p2) / 3

        if t_plan is None:
            if abs(dz_start + dz_stop) > 1e-3:
                self.t_final = self._round_time(1 / (dz_start + dz_stop))
            else:
                self.t_final = self._round_time(np.linalg.norm(p_stop - p_start) / 1)  # Predefined Cart Velocity
        else:
            self.t_final = self._round_time(t_plan)

        self.compute_time_bezier(dz_start, dz_stop, self.t_final)

    def compute_time_bezier(self, dz_start, dz_stop, t_plan):
        self.z0 = np.array([0, 0])
        self.z3 = np.array([t_plan, 1])
        if dz_start == 0 and dz_stop == 0:
            self.z1 = np.array([t_plan / 3, 1 / 3])
            self.z2 = np.array([t_plan / 3 * 2, 2 / 3])
        else:
            t_min = np.minimum(1 / (dz_start + dz_stop), self.t_final)
            if t_min == self.t_final:
                a = 0.5
                b = 0.5
            else:
                if dz_start == 0:
                    a = 0.1
                    b = 1 - a
                elif dz_stop == 0:
                    a = 0.9
                    b = 1 - a
                else:
                    a = 0.5
                    b = 0.5

            self.z1 = np.array([a * t_min, dz_start * a * t_min])
            self.z2 = np.array([self.t_final - b * t_min, 1 - dz_stop * b * t_min])

    def get_point(self, t):
        z, dz_dt, ddz_ddt = self.get_time_bezier_root(t)

        z2 = z ** 2
        z3 = z ** 3
        nz_1 = 1 - z
        nz_2 = nz_1 * nz_1
        nz_3 = nz_2 * nz_1

        p = nz_3 * self.p0 + 3 * nz_2 * z * self.p1 + 3 * nz_1 * z2 * self.p2 + z3 * self.p3
        dp_dz = 3 * nz_2 * (self.p1 - self.p0) + 6 * nz_1 * z * (self.p2 - self.p1) + 3 * z2 * (self.p3 - self.p2)
        ddp_ddz = 6 * nz_1 * (self.p2 - 2 * self.p1 + self.p0) + 6 * z * (self.p3 - 2 * self.p2 + self.p1)
        return p, dp_dz * dz_dt, ddp_ddz * dz_dt ** 2 + dp_dz * ddz_ddt

    def get_time_bezier_root(self, t):
        if np.isscalar(t):
            cubic_polynomial = np.polynomial.polynomial.Polynomial([self.z0[0] - t,
                                                                    -3 * self.z0[0] + 3 * self.z1[0],
                                                                    3 * self.z0[0] - 6 * self.z1[0] + 3 * self.z2[0],
                                                                    -self.z0[0] + 3 * self.z1[0] - 3 * self.z2[0] +
                                                                    self.z3[0]])

        tau_orig = cubic_polynomial.roots()
        tau = tau_orig.real[np.logical_and(np.logical_and(tau_orig >= -1e-6, tau_orig <= 1. + 1e-6),
                                           np.logical_not(np.iscomplex(tau_orig)))]
        tau = tau[0]
        z = (1 - tau) ** 3 * self.z0 + 3 * (1 - tau) ** 2 * tau * self.z1 + 3 * (1 - tau) * (tau ** 2) * self.z2 + (
            tau ** 3) * self.z3
        dz_dtau = 3 * (1 - tau) ** 2 * (self.z1 - self.z0) + 6 * (1 - tau) * tau * (
            self.z2 - self.z1) + 3 * tau ** 2 * (self.z3 - self.z2)
        ddz_ddtau = 6 * (1 - tau) * (self.z2 - 2 * self.z1 + self.z0) + 6 * tau * (self.z3 - 2 * self.z2 + self.z1)

        z_t = z[1]
        dz_dt = dz_dtau[1] / dz_dtau[0]
        ddz_ddt = ddz_ddtau[1] / (dz_dtau[0]) ** 2 - dz_dtau[1] * ddz_ddtau[0] / (dz_dtau[0] ** 3)
        return z_t, dz_dt, ddz_ddt

    def update_bezier_curve(self, t_start, p_stop, v_stop, t_final):
        z, dz_dt, _ = self.get_time_bezier_root(t_start)
        dp_dz = 3 * (1 - z) ** 2 * (self.p1 - self.p0) + 6 * (1 - z) * z * (self.p2 - self.p1) + 3 * z ** 2 * (
            self.p3 - self.p2)

        h_23 = np.inf
        for b in self.boundary:
            if np.linalg.norm(v_stop) > 1e-3:
                A = np.vstack([-v_stop, b[0] - b[1]]).T
                b = b[0] - p_stop
                if np.linalg.det(A) != 0:
                    h = np.linalg.solve(A, b)[0]
                    if h > 0:
                        h_23 = np.minimum(h, h_23)
            else:
                h_23 = 0
        if h_23 == 0:
            p2_new = p_stop
        else:
            l2_start = p_stop - v_stop * h_23
            min_length2 = np.minimum(0.1 / h_23, 1)
            _, p2_new = self.get_closest_point_between_line_segments(self.p0, self.p1, l2_start,
                                                                     p_stop, np.array([[0, 1], [0, 1 - min_length2]]))
        p3_new = p_stop

        p_new = np.array([[-(z - 1) ** 3, 3 * (z - 1) ** 2 * z, -3 * (z - 1) * z ** 2, z ** 3],
                          [0, (z - 1) ** 2, -2 * (z - 1) * z, z ** 2],
                          [0, 0, 1 - z, z],
                          [0, 0, 0, 1]]) @ np.vstack([self.p0, self.p1, self.p2, self.p3])

        self.p0 = p_new[0].copy()
        self.p1 = p_new[1].copy()
        self.p2 = (1 - z) * p2_new + z * p3_new
        self.p3 = p3_new

        self.t_final = self._round_time(t_final)
        if np.linalg.norm(self.p1 - self.p0) > 1e-3:
            dz_start = np.linalg.norm(dp_dz * dz_dt) / np.linalg.norm(self.p1 - self.p0) / 3
        else:
            dz_start = 0
        if h_23 == 0:
            dz_stop = 0
        else:
            dz_stop = np.linalg.norm(v_stop) / np.linalg.norm(self.p3 - self.p2) / 3
        self.compute_time_bezier(dz_start, dz_stop, self.t_final)

    def _round_time(self, time):
        return (round(time / self.step_size)) * self.step_size

    @staticmethod
    def get_closest_point_from_line_to_point(l0, l1, p, range=None):
        v = l1 - l0
        u = l0 - p
        t = - np.dot(v, u) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        if range is not None:
            t = np.clip(t, range[0], range[1])
        return (1 - t) * l0 + t * l1

    @staticmethod
    def get_closest_point_between_line_segments(l1s, l1e, l2s, l2e, range=None):
        v = l1s - l2s  # d13
        u = l2e - l2s  # d43
        w = l1e - l1s  # d21
        A = np.array([[w @ w, -u @ w],
                      [w @ u, -u @ u]])
        b = -np.array([[v @ w], [v @ u]])
        if np.linalg.det(A) != 0:
            mu = np.linalg.solve(A, b)
            mu = np.clip(mu, 0, 1)
            if range is not None:
                mu = np.clip(mu, range[:, 0:1], range[:, 1:2])
            return l1s + mu[0] * (l1e - l1s), l2s + mu[1] * (l2e - l2s)
        elif np.linalg.norm(u) == 0:
            range_1 = None if range is None else range[0]
            p = BezierPlanner.get_closest_point_from_line_to_point(l1s, l1e, l2e, range_1)
            return p, l2s
        elif np.linalg.norm(w) == 0:
            range_1 = None if range is None else range[1]
            p = BezierPlanner.get_closest_point_from_line_to_point(l2s, l2e, l1s, range_1)
            return l1s, p
        else:
            return l1s + 0.5 * (l1e - l1s), l2s + 0.5 * (l2e - l2s)


### optimizer.py ###

class TrajectoryOptimizer:
    def __init__(self, env_info):
        self.env_info = env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.n_joints = self.env_info['robot']['n_joints']
        if self.n_joints == 3:
            self.anchor_weights = np.ones(3)
        else:
            self.anchor_weights = np.array([10., 1., 10., 1., 10., 10., 1.])

    def optimize_trajectory(self, cart_traj, q_start, dq_start, q_anchor):
        joint_trajectory = np.tile(np.concatenate([q_start]), (cart_traj.shape[0], 1))
        if len(cart_traj) > 0:
            q_cur = q_start.copy()
            dq_cur = dq_start.copy()

            for i, des_point in enumerate(cart_traj):
                if q_anchor is None:
                    dq_anchor = 0
                else:
                    dq_anchor = (q_anchor - q_cur)

                success, dq_next = self._solve_aqp(des_point[:3], q_cur, dq_anchor)

                if not success:
                    return success, []
                else:
                    q_cur += (dq_cur + dq_next) / 2 * self.env_info['dt']
                    # q_cur += dq_next * self.env_info['dt']
                    dq_cur = dq_next
                    joint_trajectory[i] = q_cur.copy()
            return True, joint_trajectory
        else:
            return False, []

    def _solve_aqp(self, x_des, q_cur, dq_anchor):
        x_cur = forward_kinematics(self.robot_model, self.robot_data, q_cur)[0]
        jac = jacobian(self.robot_model, self.robot_data, q_cur)[:3, :self.n_joints]
        N_J = scipy.linalg.null_space(jac)
        b = np.linalg.lstsq(jac, (x_des - x_cur) / self.env_info['dt'], rcond=None)[0]

        P = (N_J.T @ np.diag(self.anchor_weights) @ N_J) / 2
        q = (b - dq_anchor).T @ np.diag(self.anchor_weights) @ N_J
        A = N_J.copy()
        u = np.minimum(self.env_info['robot']['joint_vel_limit'][1] * 0.9,
                       (self.env_info['robot']['joint_pos_limit'][1] * 0.92 - q_cur) / self.env_info['dt']) - b
        l = np.maximum(self.env_info['robot']['joint_vel_limit'][0] * 0.9,
                       (self.env_info['robot']['joint_pos_limit'][0] * 0.92 - q_cur) / self.env_info['dt']) - b

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

        result = solver.solve()
        if result.info.status == 'solved':
            return True, N_J @ result.x + b
        else:
            return False, b

    def solve_hit_config(self, x_des, v_des, q_0):
        reg = 1e-6
        dim = q_0.shape[0]
        opt = nlopt.opt(nlopt.LD_SLSQP, dim)

        def _nlopt_f(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(_nlopt_f, q)
            f = v_des @ jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        def _nlopt_h(q, grad):
            if grad.size > 0:
                grad[...] = 2 * (forward_kinematics(self.robot_model, self.robot_data, q)[0] - x_des) @ \
                    jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return np.linalg.norm(forward_kinematics(self.robot_model, self.robot_data, q)[0] - x_des) ** 2 - 1e-4

        opt.set_max_objective(_nlopt_f)
        opt.set_lower_bounds(self.env_info['robot']['joint_pos_limit'][0])
        opt.set_upper_bounds(self.env_info['robot']['joint_pos_limit'][1])
        opt.add_inequality_constraint(_nlopt_h)
        opt.set_ftol_abs(1e-6)
        opt.set_xtol_abs(1e-8)
        opt.set_maxtime(5e-3)

        success, x = inverse_kinematics(self.robot_model, self.robot_data, x_des, initial_q=q_0)
        if not success:
            raise NotImplementedError("Need to check")
        xopt = opt.optimize(x[:dim])
        return opt.last_optimize_result() > 0, xopt

    def solve_hit_config_ik_null(self, x_des, v_des, q_0, max_time=5e-3):
        t_start = time.time()
        reg = 0e-6
        dim = q_0.shape[0]
        IT_MAX = 1000
        eps = 1e-4
        damp = 1e-3
        progress_thresh = 20.0
        max_update_norm = 0.1
        i = 0
        TIME_MAX = max_time
        success = False

        dtype = self.robot_data.qpos.dtype

        self.robot_data.qpos = q_0

        q_l = self.robot_model.jnt_range[:, 0]
        q_h = self.robot_model.jnt_range[:, 1]
        lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
        upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

        name = link_to_xml_name(self.robot_model, 'ee')

        def objective(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(objective, q)
            f = v_des @ jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        null_opt_stop_criterion = False
        while True:
            # forward kinematics
            mujoco.mj_fwdPosition(self.robot_model, self.robot_data)

            x_pos = self.robot_data.body(name).xpos

            err_pos = x_des - x_pos
            error_norm = np.linalg.norm(err_pos)

            f_grad = numerical_grad(objective, self.robot_data.qpos.copy())
            f_grad_norm = np.linalg.norm(f_grad)
            if f_grad_norm > max_update_norm:
                f_grad = f_grad / f_grad_norm

            if error_norm < eps:
                success = True
            if time.time() - t_start > TIME_MAX or i >= IT_MAX or null_opt_stop_criterion:
                break

            jac_pos = np.empty((3, self.robot_model.nv), dtype=dtype)
            mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, self.robot_model.body(name).id)

            update_joints = jac_pos.T @ np.linalg.inv(jac_pos @ jac_pos.T + damp * np.eye(3)) @ err_pos

            # Add Null space Projection
            null_dq = (np.eye(self.robot_model.nv) - np.linalg.pinv(jac_pos) @ jac_pos) @ f_grad
            null_opt_stop_criterion = np.linalg.norm(null_dq) < 1e-4
            update_joints += null_dq

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = error_norm / update_norm
            if progress_criterion > progress_thresh:
                success = False
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            mujoco.mj_integratePos(self.robot_model, self.robot_data.qpos, update_joints, 1)
            self.robot_data.qpos = np.clip(self.robot_data.qpos, lower_limit, upper_limit)
            i += 1
        q_cur = self.robot_data.qpos.copy()

        return success, q_cur


def numerical_grad(fun, q):
    eps = np.sqrt(np.finfo(np.float64).eps)
    grad = np.zeros_like(q)
    for i in range(q.shape[0]):
        q_pos = q.copy()
        q_neg = q.copy()
        q_pos[i] += eps
        q_neg[i] -= eps
        grad[i] = (fun(q_pos, np.array([])) - fun(q_neg, np.array([]))) / 2 / eps
    return grad


### trajectory_generator.py ###
class TrajectoryGenerator:
    def __init__(self, env_info, agent_params: AgentParams, system_state: SystemState):
        self.env_info = env_info
        self.dt = 1 / self.env_info['robot']['control_frequency']
        self.agent_params = agent_params
        self.state = system_state
        self.bezier_planner = self._init_bezier_planner()
        self.cubic_linear_planner = CubicLinearPlanner(self.env_info['robot']['n_joints'], self.dt)
        self.optimizer = TrajectoryOptimizer(self.env_info)

    def generate_stop_trajectory(self):
        q_plan = self.state.q_cmd + self.state.dq_cmd * 0.04
        joint_pos_traj = self.plan_cubic_linear_motion(self.state.q_cmd, self.state.dq_cmd, q_plan,
                                                       np.zeros_like(q_plan), 0.10)[:, :q_plan.shape[0]]

        t = np.linspace(0, joint_pos_traj.shape[0], joint_pos_traj.shape[0] + 1) * 0.02
        f = CubicSpline(t, np.vstack([self.state.q_cmd, joint_pos_traj]), axis=0, bc_type=((1, self.state.dq_cmd),
                                                                                           (2, np.zeros_like(q_plan))))
        df = f.derivative(1)
        self.state.trajectory_buffer = np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
        return True

    def _init_bezier_planner(self):
        self.bound_points = np.array([[-(self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius']),
                                       -(self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-(self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius']),
                                       (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-0.1, (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-0.1, -(self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])]
                                      ])
        self.bound_points = self.bound_points + np.tile([1.51, 0.], (4, 1))
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        return BezierPlanner(table_bounds, self.dt)

    def plan_cubic_linear_motion(self, start_pos, start_vel, end_pos, end_vel, t_total=None):
        if t_total is None:
            t_total = np.linalg.norm(end_pos - start_pos) / self.agent_params.default_linear_vel

        return self.cubic_linear_planner.plan(start_pos, start_vel, end_pos, end_vel, t_total)

    def generate_bezier_trajectory(self, max_steps=-1):
        if max_steps > 0:
            t_plan = np.minimum(self.bezier_planner.t_final, max_steps * self.dt)
        else:
            t_plan = self.bezier_planner.t_final
        res = np.array([self.bezier_planner.get_point(t_i) for t_i in np.arange(self.dt, t_plan + 1e-6, self.dt)])
        p = res[:, 0]
        dp = res[:, 1]
        ddp = res[:, 2]

        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.env_info['robot']["ee_desired_height"]])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])
        return np.hstack([p, dp, ddp])

    def optimize_trajectory(self, cart_traj, q_start, dq_start, q_anchor):
        success, joint_pos_traj = self.optimizer.optimize_trajectory(cart_traj, q_start, dq_start,
                                                                     q_anchor)
        if len(joint_pos_traj) > 1:
            t = np.linspace(0, joint_pos_traj.shape[0], joint_pos_traj.shape[0] + 1) * 0.02
            f = CubicSpline(t, np.vstack([q_start, joint_pos_traj]), axis=0, bc_type=((1, dq_start),
                                                                                      (2, np.zeros_like(dq_start))))
            df = f.derivative(1)
            return success, np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
        else:
            return success, []

    def solve_anchor_pos(self, hit_pos_2d, hit_dir_2d, q_0):
        hit_pos = np.concatenate([hit_pos_2d, [self.env_info['robot']["ee_desired_height"]]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = self.optimizer.solve_hit_config(hit_pos, hit_dir, q_0)
        if not success:
            q_star = q_0
        return q_star

    def solve_anchor_pos_ik_null(self, hit_pos_2d, hit_dir_2d, q_0):
        hit_pos = np.concatenate([hit_pos_2d, [self.env_info['robot']["ee_desired_height"]]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = self.optimizer.solve_hit_config_ik_null(hit_pos, hit_dir, q_0)
        if not success:
            q_star = q_0
        return q_star


### tactics.py ###

class Tactic:
    def __init__(self, env_info, agent_params: AgentParams, state: SystemState, trajectory_generator: TrajectoryGenerator):
        self.env_info = env_info
        self.agent_params = agent_params
        self.state = state
        self.generator = trajectory_generator

        self.state.tactic_finish = True
        self.plan_new_trajectory = True
        self.replan_time = 0
        self.switch_count = 0

    def ready(self):
        pass

    def apply(self):
        pass

    def update_tactic(self):
        self._update_prediction()
        if self.state.switch_tactics_count > self.agent_params.switch_tactics_min_steps or \
                self.state.tactic_finish:
            self._update_tactic_impl()
        else:
            self.state.switch_tactics_count += 1

    def _update_prediction(self):
        if self.state.estimated_state[0] < self.agent_params.defend_range[0]:
            self.state.update_prediction(self.state.predicted_time)
        else:
            self.state.update_prediction(self.state.predicted_time, self.agent_params.defend_range[0])

    def _update_tactic_impl(self):
        pass

    def _set_tactic(self, tactic):
        if tactic != self.state.tactic_current:
            self.state.is_new_tactic = True
            self.state.switch_tactics_count = 0
            self.state.tactic_current = tactic
            self.state.has_generated_stop_traj = False

    def can_smash(self):
        if self.state.is_puck_static():
            if self.agent_params.hit_range[0] < self.state.predicted_state[0] < self.agent_params.hit_range[1] \
                    and np.abs(self.state.predicted_state[1]) < self.env_info['table']['width'] / 2 - \
                    self.env_info['puck']['radius'] - 2 * self.env_info['mallet']['radius']:
                return True
        return False

    def should_defend(self):
        if self.state.is_puck_approaching():
            if self.agent_params.defend_range[0] <= self.state.predicted_state[0] <= \
                    self.agent_params.defend_range[1] \
                    and np.abs(self.state.predicted_state[1]) <= self.agent_params.defend_width and \
                    self.state.predicted_time >= self.agent_params.max_plan_steps * self.generator.dt:
                return True
            elif self.state.predicted_time < self.agent_params.max_prediction_time and \
                    self.state.predicted_state[0] > self.agent_params.defend_range[1]:
                self.state.predicted_time += ((self.agent_params.defend_range[1] - self.state.predicted_state[0]) /
                                              self.state.predicted_state[2])
                self.state.predicted_time = np.clip(self.state.predicted_time, 0,
                                                    self.agent_params.max_prediction_time)
        return False

    def is_puck_stuck(self):
        if self.state.is_puck_static():
            if self.state.predicted_state[0] < self.agent_params.hit_range[0]:
                return True
            elif self.state.predicted_state[0] < self.agent_params.hit_range[1] \
                    and np.abs(self.state.predicted_state[1]) > self.env_info['table']['width'] / 2 - \
                    self.env_info['puck']['radius'] - 2 * self.env_info['mallet']['radius']:
                return True
        return False


class Init(Tactic):
    def _update_tactic_impl(self):
        pass

    def ready(self):
        if self.state.is_new_tactic:
            return True
        else:
            return False

    def apply(self):
        if self.state.is_new_tactic:
            if np.linalg.norm(self.state.dq_cmd) > 0.01:
                if not self.state.has_generated_stop_traj:
                    self.generator.generate_stop_trajectory()
                    self.state.has_generated_stop_traj = True
            else:
                self.state.trajectory_buffer = []

            if len(self.state.trajectory_buffer) == 0:
                t_init = 2.0
                for i in range(10):
                    success = self._plan_init_trajectory(t_init)
                    if success:
                        self.state.is_new_tactic = False
                        break
                    t_init *= 1.2

    def _plan_init_trajectory(self, t_final):
        cart_traj = self.generator.plan_cubic_linear_motion(self.state.x_cmd, self.state.v_cmd,
                                                            self.agent_params.x_init, np.zeros(3), t_final)
        opt_success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
            cart_traj, self.state.q_cmd, self.state.dq_cmd, self.agent_params.joint_anchor_pos)
        return opt_success


class Ready(Tactic):
    def __init__(self, env_info, agent_params, state: SystemState, trajectory_generator: TrajectoryGenerator,
                 only_tactic=None):
        super(Ready, self).__init__(env_info, agent_params, state, trajectory_generator)

        self.only_tactic = only_tactic

    def _update_tactic_impl(self):
        if self.only_tactic is None:
            if self.can_smash():
                self._set_tactic(TACTICS.SMASH)
            elif self.should_defend():
                self._set_tactic(TACTICS.DEFEND)
            elif self.is_puck_stuck():
                self._set_tactic(TACTICS.PREPARE)

        elif self.only_tactic == "hit":
            if self.can_smash():
                self._set_tactic(TACTICS.SMASH)

        elif self.only_tactic == "defend":
            if self.should_defend():
                self._set_tactic(TACTICS.DEFEND)

        elif self.only_tactic == "prepare":
            if self.is_puck_stuck():
                self._set_tactic(TACTICS.PREPARE)

    def ready(self):
        if self.state.is_new_tactic:
            self.state.tactic_finish = False
            self.plan_new_trajectory = True
            self.replan_time = 0
            self.switch_count = 0
            self.state.predicted_time = self.agent_params.max_prediction_time
            self.t_stop = np.maximum(np.linalg.norm(self.agent_params.x_home - self.state.x_cmd) /
                                     self.agent_params.default_linear_vel, 1.0)
            return True
        else:
            if len(self.state.trajectory_buffer) == 0:
                self.t_stop = np.maximum(np.linalg.norm(self.agent_params.x_home - self.state.x_cmd) /
                                         self.agent_params.default_linear_vel, 1.0)
                return True
        return False

    def apply(self):
        self.state.is_new_tactic = False

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    self.agent_params.x_home[:2], np.zeros(2),
                                                                    self.t_stop)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, self.agent_params.x_home[:2],
                                                                      np.zeros(2), self.t_stop)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params.max_plan_steps * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params.max_plan_steps)
            elif self.generator.bezier_planner.t_final >= self.replan_time:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.agent_params.joint_anchor_pos)

            if success:
                self.replan_time = self.agent_params.max_plan_steps * self.generator.dt
                self.plan_new_trajectory = False
                self.t_stop = self.generator.bezier_planner.t_final - \
                    self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.maximum(0, self.state.predicted_time)
                return
            self.t_stop *= 1.5
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class Prepare(Tactic):
    def _update_tactic_impl(self):
        if not self.is_puck_stuck():
            self.switch_count += 1
        else:
            self.switch_count = 0

        if (self.switch_count > 4 or self.state.tactic_finish) and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def ready(self):
        if self.state.is_new_tactic:
            self.state.tactic_finish = False
            self.plan_new_trajectory = True
            self.state.predicted_time = self.agent_params.max_prediction_time
            self.replan_time = 0
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params.joint_anchor_pos
            return True
        else:
            if len(self.state.trajectory_buffer) == 0:
                return True
            return False

    def apply(self):
        self.state.is_new_tactic = False

        puck_pos_2d = self.state.predicted_state[:2]

        if puck_pos_2d[0] < self.agent_params.prepare_range[0]:
            hit_dir_2d = np.array([-1, np.sign(puck_pos_2d[1] + 1e-6) * 0.2])
            hit_vel_mag = 0.2
        elif abs(puck_pos_2d[0]) > np.mean(self.agent_params.prepare_range):
            hit_dir_2d = np.array([-0.5, np.sign(puck_pos_2d[1] + 1e-6)])
            hit_vel_mag = 0.2
        else:
            hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])
            hit_vel_mag = 0.2

        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (self.env_info['mallet']['radius'] +
                                                     self.env_info['puck']['radius'])
        hit_vel_2d = hit_dir_2d * hit_vel_mag

        # self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        for i in range(10):
            if self.plan_new_trajectory:
                self.state.predicted_time = np.maximum(np.linalg.norm(self.state.x_cmd[:2] - puck_pos_2d) /
                                                       self.agent_params.default_linear_vel, 1.0)
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params.max_plan_steps * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params.max_plan_steps)
            elif self.generator.bezier_planner.t_final >= 2 * self.generator.dt:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                break

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params.max_plan_steps * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                    self.state.trajectory_buffer.shape[0] * self.generator.dt
                return

            self.state.predicted_time += self.agent_params.max_plan_steps * self.generator.dt
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class Defend(Tactic):
    def _update_tactic_impl(self):
        self.state.update_prediction(self.state.predicted_time, self.agent_params.defend_range[0])
        if not self.should_defend():
            self.switch_count += 1
        else:
            self.switch_count = 0

        if (self.switch_count > 4 or self.state.tactic_finish) and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def ready(self):
        if self.state.is_new_tactic:
            self.state.tactic_finish = False
            self.plan_new_trajectory = True
            self.state.predicted_time = self.agent_params.max_prediction_time
            self.replan_time = 0
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params.joint_anchor_pos
            return True
        else:
            if len(self.state.trajectory_buffer) == 0:
                return True
            return False

    def apply(self):
        self.state.is_new_tactic = False

        puck_pos_2d = self.state.predicted_state[:2]

        hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])

        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (self.env_info['mallet']['radius'])
        hit_vel_2d = hit_dir_2d * 0.05

        # self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params.max_plan_steps * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params.max_plan_steps)
            elif self.generator.bezier_planner.t_final >= 2 * self.generator.dt:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                break

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params.max_plan_steps * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                    self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.clip(self.state.predicted_time, 0,
                                                    self.agent_params.max_prediction_time)
                return

            self.state.predicted_time += self.agent_params.max_plan_steps * self.generator.dt
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class Repel(Tactic):
    def _update_tactic_impl(self):
        pass

    def ready(self):
        pass

    def apply(self):
        pass


class Smash(Tactic):
    def __init__(self, env_info, agent_params, state: SystemState, trajectory_generator: TrajectoryGenerator):
        super().__init__(env_info, agent_params, state, trajectory_generator)
        self.hit_vel_mag = self.agent_params.max_hit_velocity
        self.q_anchor_pos = self.agent_params.joint_anchor_pos

    def _update_tactic_impl(self):
        if not self.can_smash() or self.state.tactic_finish:
            self.switch_count += 1
        else:
            self.switch_count = 0

        if self.switch_count > 4 and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def ready(self):
        if self.state.is_new_tactic:
            self.plan_new_trajectory = True
            self.state.tactic_finish = False
            self.state.predicted_time = self.agent_params.max_prediction_time
            self.replan_time = 0
            self.hit_vel_mag = self.agent_params.max_hit_velocity
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params.joint_anchor_pos
            return True
        else:
            if len(self.state.trajectory_buffer) == 0 and self.can_smash():
                return True
            return False

    def apply(self):
        self.state.is_new_tactic = False

        goal_pos = np.array([2.49, 0.0])
        puck_pos_2d = self.state.predicted_state[:2]

        hit_dir_2d = goal_pos - puck_pos_2d
        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_vel_2d = hit_dir_2d * self.hit_vel_mag

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

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params.max_plan_steps * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params.max_plan_steps)
            elif self.generator.bezier_planner.t_final >= self.replan_time:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params.max_plan_steps * self.generator.dt
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


### tactic_smash_instruct.py ###
class SmashInstruct(Tactic):
    def __init__(self, env_info, agent_params, state: SystemState, trajectory_generator: TrajectoryGenerator):
        super().__init__(env_info, agent_params, state, trajectory_generator)
        self.hit_vel_mag = self.agent_params.max_hit_velocity
        self.q_anchor_pos = self.agent_params.joint_anchor_pos
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
            self.state.predicted_time = self.agent_params.max_prediction_time
            self.replan_time = 0
            self.hit_vel_mag = self.agent_params.max_hit_velocity
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params.joint_anchor_pos
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

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params.max_plan_steps * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params.max_plan_steps)
            elif self.generator.bezier_planner.t_final >= self.replan_time:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params.max_plan_steps * self.generator.dt
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


### baseline_agent_instruct.py ### 
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

### environment/air_hockey_hit.py ### 


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


### evaluate.py ###
def compute_angle_and_velocity(puck_init_pos, puck_init_vel, goal_pos):
    hit_dir = goal_pos - puck_init_pos
    angel = np.arctan2(hit_dir[1], hit_dir[0])
    velocity_scale = 1.
    return angel, velocity_scale


def main():
    viewer_params = {'camera_params': {'static': dict(distance=3.0, elevation=-45.0, azimuth=90.0,
                                                      lookat=(0., 0., 0.))},
                     'width': 1440, 'height': 810,
                     'default_camera_mode': 'static',
                     'hide_menu_on_startup': True}

    env = IiwaPositionHit(interpolation_order=-1, viewer_params=viewer_params, horizon=200)
    agent = BaselineAgent(env_info=env.env_info, agent_id=1, only_tactic='hit', max_hit_velocity=1.0)

    n_episodes = 100
    env.reset()

    return_history = []

    for i in range(n_episodes):
        done = False
        
        env.puck_init_pos = np.random.rand(2) * (env.hit_range[:, 1] - env.hit_range[:, 0]) + env.hit_range[:, 0]
        env.puck_init_vel = np.random.rand(2) * 0.2
        obs = env.reset()
        puck_pos = obs[:2] - np.array([1.51, 0.])
        puck_vel = obs[3:5]
        goal_pos = np.array([0.938, 0.0])

        angle, velocity_scale = compute_angle_and_velocity(puck_init_pos=puck_pos, puck_init_vel=puck_vel, goal_pos=goal_pos)
        agent.set_instruction({'hit_angle': angle, 'hit_velocity': velocity_scale})

        n_steps = 0
        hit_puck_vel = 0.
        while n_steps < env.info.horizon:
            action = agent.draw_action(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            n_steps += 1
            if np.linalg.norm(hit_puck_vel) == 0. and obs[0] >= 1.51:
                hit_puck_vel = obs[3:5]
            if done:
                break

        puck_final_pos = obs[:2] - np.array([1.51, 0.])
        dist_to_goal = np.linalg.norm(puck_final_pos - goal_pos)

        return_history.append(dist_to_goal)

    env.close()
    return return_history


if __name__ == "__main__":
    main()
