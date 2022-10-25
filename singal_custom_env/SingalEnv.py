"""
Environments for training vehicles to reduce congestion in a merge.

This environment was used in:
TODO(ak): add paper after it has been published.
"""

from flow.envs.base import Env
from flow.core import rewards

from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 25,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
}


class SingalEnv(Env):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.num_rl = env_params.additional_params["num_rl"]

        self.rl_queue = collections.deque()

        self.rl_veh = []

        self.leader = []
        self.follower = []

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-1,
            high=1,
            shape=(self.num_rl, ),
            dtype=np.int8)

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(5 * self.num_rl, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        for i, rl_id in enumerate(self.rl_veh):
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_lane_change(rl_id, int(rl_actions[i]))

    def get_state(self, rl_id=None, **kwargs):
        self.leader = []
        self.follower = []

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = self.k.network.length()

        observation = [0 for _ in range(5 * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            this_speed = self.k.vehicle.get_speed(rl_id)
            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None]:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                    - self.k.vehicle.get_x_by_id(rl_id) \
                    - self.k.vehicle.get_length(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            observation[5 * i + 0] = this_speed / max_speed
            observation[5 * i + 1] = (lead_speed - this_speed) / max_speed
            observation[5 * i + 2] = lead_head / max_length
            observation[5 * i + 3] = (this_speed - follow_speed) / max_speed
            observation[5 * i + 4] = follow_head / max_length

        return observation

    def compute_reward(self, rl_actions, **kwargs):
        # """See class definition."""
        # if self.env_params.evaluate:
        #     return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        # else:
        #     # return a reward of 0 if a collision occurred
        #     if kwargs["fail"]:
        #         return 0

        #     # reward high system-level velocities
            # cost1 = rewards.desired_velocity(self, fail=kwargs["fail"])

        #     # penalize small time headways
        #     cost2 = 0
        #     t_min = 1  # smallest acceptable time headway
        #     for rl_id in self.rl_veh:
        #         lead_id = self.k.vehicle.get_leader(rl_id)
        #         if lead_id not in ["", None] \
        #                 and self.k.vehicle.get_speed(rl_id) > 0:
        #             t_headway = max(
        #                 self.k.vehicle.get_headway(rl_id) /
        #                 self.k.vehicle.get_speed(rl_id), 0)
        #             cost2 += min((t_headway - t_min) / t_min, 0)

        #     # weights for cost1, cost2, and cost3, respectively
        #     eta1, eta2 = 1.00, 0.10

        #     return max(eta1 * cost1 + eta2 * cost2, 0)
        delayed_sum = 0
        for rl_id in self.rl_veh:
            delayed_sum += rewards.avg_delay_specified_vehicles(self, rl_id)

        if self.env_params.evaluate:
            with open("result_log.txt", "w") as f:
                f.write(rewards.average_velocity(), delayed_sum)
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))

        avg_velocity = rewards.average_velocity(self)
        reward_for_minimizing_delay = rewards.min_delay(self)
        standstill = rewards.penalize_standstill(self)
        weighted_reward =  avg_velocity*10 - delayed_sum/10+reward_for_minimizing_delay*10+standstill*10
        
        print(avg_velocity, delayed_sum, reward_for_minimizing_delay, standstill)
        print("weighted reward",weighted_reward)

        return weighted_reward

    def additional_command(self):
        """See parent class.

        This method performs to auxiliary tasks:

        * Define which vehicles are observed for visualization purposes.
        * Maintains the "rl_veh" and "rl_queue" variables to ensure the RL
          vehicles that are represented in the state space does not change
          until one of the vehicles in the state space leaves the network.
          Then, the next vehicle in the queue is added to the state space and
          provided with actions from the policy.
        """
        # add rl vehicles that just entered the network into the rl queue
        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)

    def reset(self):
        """See parent class.

        In addition, a few variables that are specific to this class are
        emptied before they are used by the new rollout.
        """
        self.leader = []
        self.follower = []
        return super().reset()
