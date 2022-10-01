import gym
import sumo_rl
env = gym.make('sumo-rl-v0',
                net_file='nets/big-intersection/big-intersection.net.xml',
                route_file='nets/big-intersection/routes.rou.xml',
                out_csv_name='outputs/4x4grid/test',
                use_gui=True,
                num_seconds=100000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated