import errno

from time import strftime
from flow.utils.registry import env_constructor
import json
import os

class TrainSingalJC():
    def __init__(self, num_cpus=1, num_steps=50, rollout_size=50, save_path=None, exp_tag=None, multiagent=False):
        self.flow_params = scenario.flow_params
        if exp_tag is None:
            exp_tag = self.flow_params['exp_tag']
        result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))
        #FIXME : Is this params need to be in train?
        self.num_cpus, self.num_steps, self.rollout_size, self.exp_tag, self.multiagent = num_cpus, num_steps, rollout_size, exp_tag, multiagent
        self.save_path, self.exp_tag = save_path, exp_tag

    def train(self, save=True):
        from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines import PPO2

        if self.num_cpus == 1:
            constructor = env_constructor(params=self.flow_params, version=0)()
            # The algorithms require a vectorized environment to run
            env = DummyVecEnv([lambda: constructor])
        else:
            env = SubprocVecEnv([env_constructor(params=self.flow_params, version=i)
                                 for i in range(self.num_cpus)])

        train_model = PPO2('MlpPolicy', env, verbose=1, n_steps=self.rollout_size)
        train_model.learn(total_timesteps=self.num_steps)
        if save:
            save_model(train_model, self.save_path, self.exp_tag)
        return train_model

    @staticmethod
    def ensure_dir(path):
        """Ensure that the directory specified exists, and if not, create it."""
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        return path

    @staticmethod
    def save_model(model, path, exp_tag):
        if path is None:
            path = os.path.realpath(os.path.expanduser('~/baseline_results'))
        ensure_dir(path)
        save_path = os.path.join(path, result_name)
        os.mkdir(os.path.join(path, exp_tag))
        model.save(save_path)
        with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
            json.dump(flow_params, outfile,
                      cls=FlowParamsEncoder, sort_keys=True, indent=4)


if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--num_cpus', type=int, default=1, help='How many CPUs to use')
    args.add_argument('--num_steps', type=int, default=5000, help='How many total steps to perform learning over')
    args.add_argument('--rollout_size', type=int, default=1000, help='How many steps are in a training batch.')
    args.add_argument('--save_path', type=str, default=None,help='Directory with save')
    Trainer = TrainSingalJC(args.num_cpus, args.num_steps, args.rollout_size, args.save_path)
