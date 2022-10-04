import collections
from gym import spaces
from flow.envs.base import Env
from flow.core import rewards

import numpy as np
ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # whether the toll booth should be active
    "disable_tb": True,
    # whether the ramp meter is active
    "disable_ramp_metering": True,
}

class SingalEnv(Env):
    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.num_rl = env_params.additional_params["num_rl"]

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []
        super().__init__(env_params, sim_params, network, simulator)

    def action_space(self):
        return spaces.Discrete(3)

    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(5 * self.num_rl, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            # return a reward of 0 if a collision occurred
            if kwargs["fail"]:
                return 0

            # reward high system-level velocities
            cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

            # penalize small time headways
            cost2 = 0
            t_min = 1  # smallest acceptable time headway
            for rl_id in self.rl_veh:
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(rl_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(rl_id) /
                        self.k.vehicle.get_speed(rl_id), 0)
                    cost2 += min((t_headway - t_min) / t_min, 0)

            # weights for cost1, cost2, and cost3, respectively
            eta1, eta2 = 1.00, 0.10

            return max(eta1 * cost1 + eta2 * cost2, 0)