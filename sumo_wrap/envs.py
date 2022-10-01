import gym

import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
import traci
from sumo_rl.environment.env import SumoEnvironment

from model import DQN
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.env_checker import check_env
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gym.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

from sumo_rl.environment.traffic_signal import TrafficSignal

class SumoEnvironment2(SumoEnvironment):
    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False  # there are no 'terminal' states in this environment
        truncated = dones['__all__']  # episode ends when sim_step >= max_steps
        info = self._compute_info()

        if self.single_agent:
            print( observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated,info)
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, info
        else:
            return observations, rewards, dones, info

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time, 
                                                  self.yellow_time, 
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  self.reward_fn,
                                                  self.sumo) for ts in self.ts_ids}
        self.vehicles = dict()

        if self.single_agent:
            # print("from env...", self._compute_observations()[self.ts_ids[0]])
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

env = SumoEnvironment2(net_file='nets/big-intersection/big-intersection.net.xml',
                        single_agent=True,
                        route_file='nets/big-intersection/routes.rou.xml',
                        out_csv_name='outputs/big-intersection/dqn',
                        use_gui=False,
                        num_seconds=5400,
                        yellow_time=4,
                        min_green=5,
                        max_green=60)
check_env(env)