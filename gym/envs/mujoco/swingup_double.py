import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SwingupDoubleEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    """
    ### Description

    This environment is based on the inverted double pendulum environment.

    ### Action Space

    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
    numerical force applied to the cart (with magnitude representing the amount of force and
    sign representing the direction)
    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |

    ### Observation Space

    The state space consists of positional values of different body parts of the pendulum system,
    followed by the velocities of those individual parts (their derivatives) with all the
    positions ordered before all the velocities.
    The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:
    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0   | position of the cart along the linear surface                        | -Inf                 | Inf                | slider | slide | position (m) |
    | 1   | sine of the angle between the cart and the first pole                | -Inf                 | Inf                | sin(hinge) | hinge | unitless |
    | 2   | sine of the angle between the two poles                              | -Inf                 | Inf                | sin(hinge2) | hinge | unitless |
    | 3   | cosine of the angle between the cart and the first pole              | -Inf                 | Inf                | cos(hinge) | hinge | unitless |
    | 4   | cosine of the angle between the two poles                            | -Inf                 | Inf                | cos(hinge2) | hinge | unitless |
    | 5   | velocity of the cart                                                 | -Inf                 | Inf                | slider | slide | velocity (m/s) |
    | 6   | angular velocity of the angle between the cart and the first pole    | -Inf                 | Inf                | hinge | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the angle between the two poles                  | -Inf                 | Inf                | hinge2 | hinge | angular velocity (rad/s) |
    | 8   | constraint force - 1                                                 | -Inf                 | Inf                |  |  | Force (N) |
    | 9   | constraint force - 2                                                 | -Inf                 | Inf                |  |  | Force (N) |
    | 10  | constraint force - 3                                                 | -Inf                 | Inf                |  |  | Force (N) |

    ### Rewards

    The reward consists of two parts:
    - *alive_bonus*: A reward of +5 is awarded for each timestep that the environment
    hasn't been terminated.
    - *distance_penalty*: This reward is a measure of how far the *tip* of the second pendulum
    (the only free end) moves, and it is calculated as
    *0.01 * x<sup>2</sup> + (y - 2)<sup>2</sup>*, where *x* is the x-coordinate of the tip
    and *y* is the y-coordinate of the tip of the second pole.
    - *velocity_penalty*: A negative reward for penalising the agent if it moves too
    fast *0.001 *  v<sub>1</sub><sup>2</sup> + 0.005 * v<sub>2</sub> <sup>2</sup>*
    The total reward returned is ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*

    ### Starting State

    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
    of [-0.1, 0.1] added to the positional values (cart position and pole angles) and standard
    normal force with a standard deviation of 0.1 added to the velocity values for stochasticity.

    ### Episode Termination

    The episode terminates when any of the following happens:
    1. The episode duration reaches 1000 timesteps.
    2. Any of the state space values is no longer finite.
    3. The y_coordinate of the tip of the second pole *is less than or equal* to 1, having reached a y_coordinate of more than 1.05.
    The maximum standing height of the system is 1.196 m when all the parts are perpendicularly vertical on top of each other).

    env = gym.make('SwingupDouble-v1')

    """


    def __init__(self):
        self._is_up = False

        mujoco_env.MujocoEnv.__init__(self, "swingup_double.xml", 1) # skips n frames (calls sim.step() n times before control)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        o = self._get_obs()
        r = self._get_reward()
        d = self._is_done()
        return o, r, d, {'time': self.sim.data.time}

    def _get_reward(self):
        x_site, _, y_site = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x_site ** 2 + (y_site - 2.) ** 2
        if y_site > 1.05: self._is_up = True

        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2

        alive_bonus = 5

        reward = alive_bonus - dist_penalty - vel_penalty
        return reward

    def _get_obs(self):
        return np.concatenate([
                self.sim.data.qpos[:1],  # cart x pos
                np.sin(self.sim.data.qpos[1:]),  # link angles
                np.cos(self.sim.data.qpos[1:]),
                self.sim.data.qvel,
                self.sim.data.qfrc_constraint,
            ]).ravel()

    def _is_done(self):
        if np.abs(self.sim.data.qpos[0]) > 1.99: return True
        if self.sim.data.site_xpos[0][2] <= 1. and self._is_up: return True
        else: return False

    def reset_model(self):
        # self._elapsed_steps = 0
        self._is_up = False
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1,
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.7
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]
