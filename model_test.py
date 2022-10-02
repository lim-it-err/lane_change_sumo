from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, SAC
from flow.utils.rllib import get_flow_params
from flow.utils.registry import env_constructor

class TestSingalJC():
    def __init__(self, algorithm, model_path, param_path, render=True):
        #FIXME: Allow multiple algorithm
        self.model = PPO2.load(model_path)
        self.flow_params = get_flow_params(param_path)
        self.flow_params['sim'].render = render
        self.eval_env = DummyVecEnv([lambda: env_constructor(params=self.flow_params, version=0)()])

    def test(self):
        obs = self.eval_env.reset()
        reward =0
        for _ in range(self.flow_params['env'].horizon):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.eval_env.step(action)
            reward += rewards
        print('the final reward is {}'.format(reward))

    def __del__(self):
        pass

if __name__ == "__main__":
    model_path = "/home/ubuntu/baseline_results/DesiredVelocity/2022-10-02-19:38:22.zip"
    param_path = "/home/ubuntu/baseline_results/DesiredVelocity/2022-10-02-19:38:22.json"

    test = TestSingalJC(PPO2, model_path, param_path).test()
