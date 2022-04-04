from tqdm import tqdm
import torch
import copy
import gym3
from matplotlib import pyplot as plt
from procgen import ProcgenGym3Env

class LinearEvoActor(torch.nn.Module):

	def __init__(self, input_dim, output_dim, hidden_dim, *args, **kwargs):
		super(LinearEvoActor, self).__init__()

		self.define_network()

		# remove gradient calculations
		self.remove_grad()

	def define_network(self):

		self.weights = torch.nn.ModuleList()
		
		self.weights.append(torch.nn.Linear(1024, 512))
		self.weights.append(torch.nn.Linear(512, 64))
		self.weights.append(torch.nn.Linear(64, 15))

		self.relu = torch.nn.ReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax(dim=0)

		self.remove_grad()

	def remove_grad(self):
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		x = torch.tensor(x, dtype=torch.float32).view(1, -1)
		x = self.fc1(x)
		x = torch.nn.functional.relu(x)
		x = self.fc2(x)
		x = torch.nn.functional.relu(x)
		x = self.fc3(x)
		return x

	def act(self, x):
		with torch.no_grad():
			x = self.forward(x).max(1)[1].view(1, 1)
			out = x.detach().numpy()
		return out

class ConvEvoActor(torch.nn.Module):

	def __init__(self, *args, **kwargs):
		super(ConvEvoActor, self).__init__()

		# initialize layers
		self.define_network()

	def define_network(self):

		# store in weight dict
		self.weights = torch.nn.ModuleList() 

		self.weights.append(torch.nn.Conv2d(3, 32, kernel_size=4, stride=2))
		self.weights.append(torch.nn.Conv2d(32, 64, kernel_size=4, stride=2))
		self.weights.append(torch.nn.Conv2d(64, 128, kernel_size=4, stride=2))
		self.weights.append(torch.nn.Conv2d(128, 256, kernel_size=4, stride=2))

		self.weights.append(torch.nn.Linear(1024, 512))
		self.weights.append(torch.nn.Linear(512, 64))
		self.weights.append(torch.nn.Linear(64, 15))

		self.relu = torch.nn.ReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax(dim=0)

		self.remove_grad()

	def remove_grad(self):
		for i in range(len(self.weights)):
			self.weights[i].requires_grad = False

	def forward(self, x):

		
		# need to add a dimension to beginning of tensor
		out = torch.Tensor(x).reshape(-1, 3, 64, 64).float()

		for i in range(len(self.weights)):
			out = self.weights[i](out) # convolution
			out = self.leaky_relu(out)	# activation
			# reshape for linear layer
			if i == 3:
				out = out.reshape(-1, 2*2*256)
		
		out = self.relu(out)
		out = self.softmax(out)

		return out

	def act(self, x):
		with torch.no_grad():
			out = self.forward(x)#.max(1)[1].view(1, 1)
			action_probabilities = torch.distributions.Categorical(out)
			actions = action_probabilities.sample()
			actions = actions.detach().numpy()
		return actions

	def apply_mutation(self, mutation):
		for i in range(len(self.weights)):
				self.weights[i].weight = torch.nn.parameter.Parameter(self.weights[i].weight + mutation[i])

	def remove_mutation(self, mutation):
		for i in range(len(self.weights)):
				self.weights[i].weight = torch.nn.parameter.Parameter(self.weights[i].weight - mutation[i])

class Buffer(object):

	def __init__():
		pass

	def store(self, action, state, reward, done):
		raise NotImplementedError


def generate_mutations(actor, std=0.1, population=20):
	'''
	generate a population of mutated actors
	'''
	mutations = []
	for p in range(population):
		weights = torch.nn.ParameterList()
		for layer in actor.weights:
			# sample normal distribution tensor with same shape as weight
			w = torch.randn(layer.weight.shape).float()*std
			w = torch.nn.parameter.Parameter(w)
			w.requires_grad = False

			weights.append(w)

		mutations.append(weights)

	return mutations
'''
	for i in range(n_iters):
		dw = random_noise(actor.parameters()) # normally sampled weights
		child = copy.deepcopy(actor) 
		while environment is running:
'''

def genetic_algorithm(env, actor, iters=10, num_steps=1000, population=1, lr=0.001, std=0.1, *args, **kwargs):
	'''
	need to be able to generate random noise for weights of network
	need to be able to map the returns of the episode to the 
	what is the best way to compute the return?
	'''

	plt.ion()
	avg_rewards = []
	# loop through each training iteration 
	for i in range(iters):


		mutations = generate_mutations(actor, std, population)
		for i in tqdm(range(len(mutations))):

			# run mutated agent to collect returns for mutation weighting
			mutations[i] = compute_returns(env, actor, mutations[i], num_steps=num_steps, lr=lr)

		# update the actor with all mutations wieghted by their returns
		update_weights(actor, mutations, lr=lr, std=std)

		# see how well the actor it doing
		avg_reward = test(env, actor, num_steps=10000)
		avg_rewards.append(avg_reward)

		print('iteration: ', i, 'avg_reward: ', avg_reward)

		plt.title("Reward per Epoch")
		plt.xlabel("Epoch")
		plt.ylabel("Reward")
		plt.plot(avg_rewards, label="average reward")
		plt.legend(loc="upper left")
		plt.draw()
		"""
		if epoch%10 == 0:
			plt.savefig('reward_img/epoch{}.png'.format(epoch))
		"""
		plt.pause(0.0001)
		plt.clf()

def test(env, actor, num_steps=10000):
	'''
	test the actor on the environment
	'''
	total_reward = 0
	episodes = 0
	for i in tqdm(range(num_steps)):
		reward, state, first = env.observe()
		total_reward += reward
		action = actor.act(state['rgb'])
		env.act(action)
		if first:
			episodes += 1
	return total_reward/episodes

def compute_returns(env, actor, mutation, num_steps=100, lr=0.1, *args, **kwargs):

	total_reward = 0
	for step in range(num_steps):
		reward, state, first = env.observe()
		total_reward += reward
		action = actor.act(state['rgb'])
		env.act(action)
		reward, state, first = env.observe()

	# multiply each mutation weight by reward
	for i in range(len(mutation)):
		mutation[i] = torch.nn.parameter.Parameter(mutation[i]*total_reward)

	return mutation

def update_weights(actor, mutations, lr, std):

	num_mutations = len(mutations)

	# update actor weights
	for i in range(len(actor.weights)):
		dws = []
		for j in range(num_mutations):
			dws.append(mutations[j][i])
		sum_dw = torch.stack(dws, dim=0).sum(dim=0)
		w = actor.weights[i].weight + lr*sum_dw/(num_mutations*std)
		actor.weights[i].weight = torch.nn.parameter.Parameter(w)

def main():

	'''
	env = gym.make('CartPole-v0')
	actor = LinearEvoActor(4, 2, 64)
	#genetic_algorithm(env, actor, iters=100, population=20, lr=0.001)
	for param in actor.parameters():
		print(param.shape)
	'''
	

	# for each mutation, add and then remove mutations
	
	'''
	for mutation in mutations:
		actor.apply_mutation(mutation)
		actor.remove_mutation(mutation)

	'''

	env = ProcgenGym3Env(
		num=1,
		env_name="coinrun",
		render_mode="rgb_array",
		num_levels=1,
		start_level=2,
		)

	actor = ConvEvoActor()

	# get mutations
	mutations = generate_mutations(actor, std=0.1, population=20)



	env = gym3.ViewerWrapper(env, info_key="rgb")

	genetic_algorithm(env, actor, iters=100, steps=1000, population=10, lr=0.001, std=0.1)
	
		

if __name__ == '__main__':
	main()