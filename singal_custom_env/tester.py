from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2
from flow.utils.rllib import get_flow_params
from flow.utils.registry import env_constructor

class TestBottleNeck():
    def __init__(self, algorithm, model_path, param_path, render=True):
        #FIXME: Allow multiple algorithm
        self.model = PPO2.load(model_path)
        self.flow_params = get_flow_params(param_path)
        self.flow_params['sim'].render = render
        self.eval_env = DummyVecEnv([lambda: env_constructor(params=self.flow_params, version=0)()])

    def test(self):
        obs = self.eval_env.reset()
        reward =0
        for i in range(self.flow_params['env'].horizon):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.eval_env.step(action)
            reward += rewards
        print('the final reward is {}'.format(reward))


if __name__ == "__main__":
    model_path = "/home/scar1080/lane_change_sumo/singal_custom_env/result/SingalEnv/2022-10-25-10:51:51.zip"
    param_path = "/home/scar1080/lane_change_sumo/singal_custom_env/result/SingalEnv/2022-10-25-10:51:51.json"

    test = TestBottleNeck(PPO2, model_path, param_path).test()
