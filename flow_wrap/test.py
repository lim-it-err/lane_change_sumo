from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, SAC
from flow.utils.rllib import get_flow_params


env = DummyVecEnv([lambda: constructor])

class TestSingalJC():
    def __init__(self, algorithm, model_path, param_path, render=True):
        #FIXME: Allow multiple algorithm
        self.model = PPO2.load(model_path)
        self.flow_params = get_flow_params(param_path)
        self.flow_params['sim'].render = render
        self.eval_env = DummyVecEnv([lambda: env_constructor(params=self.flow_params, version=0)()])

    def __call__(self):
        obs = self.eval_env.reset()
        reward =0
        for _ in range(flow_params['env'].horizon):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = eval_env.step(action)
            reward += rewards
        print('the final reward is {}'.format(reward))

    def __del__(self):
        pass

if __name__ == "__main__":
    model_path = "~/baseline_results/stabilizing_open_network_merges/2022-10-02-19:20:54.zip"
    param_path = "~/baseline_results/stabilizing_open_network_merges/2022-10-02-19:20:54.json"

    TestSingalJC(PPO2, model_path, param_path)