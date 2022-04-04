from argparse import ArgumentParser

from algorithms import *
from utils import *


def main():
    parser = ArgumentParser()

	# experiment and  environment
	parser.add_argument('--experiment_name', default="default", type=str)
	parser.add_argument('--environment_name', default="couinrun")

	# saving options
	parser.add_argument('--log', default=True, type=bool)
	parser.add_argument('--graph', default=True, type=bool)

	# training params
	parser.add_argument('--random_seeds', default=list(range(10)), type=list)
	parser.add_argument('--n_episodes', default=20, type=int)
	parser.add_argument('--n_steps', default=100000, type=int)
	parser.add_argument('--batch_sz', default=64, type=int)
	parser.add_argument('--gamma', default=0.999, type=float)
	parser.add_argument('--critic_epochs', default=20, type=int)
	parser.add_argument('--n_envs', default=1, type=int)

	# model params
	parser.add_argument('--actor_lr', default=2e-1, type=float)
	parser.add_argument('--critic_lr', default=2e-1, type=float)
	parser.add_argument('--epsilon', default=0.3, type=float)

	params = parser.parse_args()

if __name__ == '__main__':
    main()