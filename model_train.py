class TrainSingalJC():
    def __init__(self, num_cpus=1, num_steps=50, rollout_size=50, multiagent=False):



if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--num_cpus', type=int, default=1, help='How many CPUs to use')
    args.add_argument('--num_steps', type=int, default=5000, help='How many total steps to perform learning over')
    args.add_argument('--rollout_size', type=int, default=1000, help='How many steps are in a training batch.')
    args.add_argument('--checkpoint_path', type=str, default=None,help='Directory with checkpoint to restore training from.')
    Trainer = TrainSingalJC(args.num_cpus, args.num_steps, args.checkpoint_path, False)
