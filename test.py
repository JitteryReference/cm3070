from tqdm import tqdm
import torch
import copy
import gym3
from matplotlib import pyplot as plt
from procgen import ProcgenGym3Env
import gym

class LinearEvoActor(torch.nn.Module):

	def __init__(self, *args, **kwargs):
		super(LinearEvoActor, self).__init__()

		self.define_network()

		# remove gradient calculations
		self.remove_grad()

	def define_network(self):

		self.weights = torch.nn.ModuleList()
		
		self.weights.append(torch.nn.Linear(4, 16))
		#self.weights.append(torch.nn.Linear(64, 64))
		self.weights.append(torch.nn.Linear(16, 2))

		self.relu = torch.nn.ReLU()
		self.leaky_relu = torch.nn.LeakyReLU()
		self.sigmoid = torch.nn.Sigmoid()
		self.tanh = torch.nn.Tanh()
		self.softmax = torch.nn.Softmax()

		self.remove_grad()

	def remove_grad(self):
		for i in range(len(self.weights)):
			self.weights[i].requires_grad = False

	def forward(self, x):
		# need to add a dimension to beginning of tensor
		out = torch.Tensor(x).view(1, -1)
		for i in range(len(self.weights)):
			out = self.weights[i](out) # convolution
			out = self.tanh(out) # activation
		out = self.relu(out)		# activation		
		out = self.softmax(out)
		return out

	def act(self, x):
		with torch.no_grad():
			out = self.forward(x)
			#print(out)
			action_probabilities = torch.distributions.Categorical(out)
			#print(action_probabilities)
			actions = action_probabilities.sample()
			actions = actions.detach().numpy()	
		return actions[0]

	def max_act(self, x):
		with torch.no_grad():
			out = self.forward(x)
			# get the index of the max value
			action = out.argmax().detach().numpy()
		return action

	def apply_mutation(self, mutation):
		for i in range(len(self.weights)):
				self.weights[i].weight = torch.nn.parameter.Parameter(self.weights[i].weight + mutation['weights'][i])
				self.weights[i].bias = torch.nn.parameter.Parameter(self.weights[i].bias + mutation['biases'][i])

	def remove_mutation(self, mutation):
		for i in range(len(self.weights)):
				self.weights[i].weight = torch.nn.parameter.Parameter(self.weights[i].weight - mutation['weights'][i])
				self.weights[i].bias = torch.nn.parameter.Parameter(self.weights[i].bias - mutation['biases'][i])


def generate_mutations(actor, std=0.1, population=20):
	'''
	generate a population of mutated actors
	'''
	mutations = []
	for p in range(population):
		weights = torch.nn.ParameterList()
		biases = torch.nn.ParameterList()
		for layer in actor.weights:
			# sample normal distribution tensor with same shape as weight
			w = torch.randn(layer.weight.shape).float()*std
			w = torch.nn.parameter.Parameter(w)
			w.requires_grad = False
			weights.append(w)

			b = torch.randn(layer.bias.shape).float()*std
			b = torch.nn.parameter.Parameter(b)
			b.requires_grad = False
			biases.append(b)

		mutation = {'weights': weights, 'biases': biases}
		mutations.append(mutation)

	return mutations
'''
	for i in range(n_iters):
		dw = random_noise(actor.parameters()) # normally sampled weights
		child = copy.deepcopy(actor) 
		while environment is running:

'''

def genetic_algorithm(env, actor, iters=10, num_steps=1000, population=1, lr=0.001, std=0.1, render=False, *args, **kwargs):
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
			mutations[i] = compute_returns(env, actor, mutations[i], num_steps=num_steps, lr=lr, render=render)

		# update the actor with all mutations wieghted by their returns
		update_weights(actor, mutations, lr=lr, std=std)

		# see how well the actor it doing
		avg_reward = test(env, actor, num_steps=200)
		avg_rewards.append(avg_reward)

		print('iteration: ', i, 'avg_reward: ', avg_reward)

		plt.title("Reward per Epoch")
		plt.xlabel("Epoch")
		plt.ylabel("Reward")
		plt.plot(avg_rewards, label="average reward")
		plt.legend(loc="upper left")
		plt.draw()
		plt.pause(0.0001)
		plt.clf()
	"""
	if epoch%10 == 0:
		plt.savefig('reward_img/epoch{}.png'.format(epoch))
	"""
	

def test(env, actor, num_steps=200):
	'''
	test the actor on the environment
	'''
	total_reward = 0
	episodes = 0

	state = env.reset()
	for step in range(num_steps):
		action = actor.max_act(state)
		state, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			state = env.reset()
			episodes += 1
			break
	
	return total_reward/episodes

def compute_returns(env, actor, mutation, num_steps=100, lr=0.1, render=False, *args, **kwargs):

	total_reward = 0
	state = env.reset()
	num_episodes = 0
	for step in range(num_steps):
		action = actor.act(state)
		#print(action)
		state, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			state = env.reset()
			num_episodes += 1
		if render:
			env.render()
	# multiply each mutation weight by reward
	for i in range(len(mutation['weights'])):
		mutation['weights'][i] = torch.nn.parameter.Parameter(mutation['weights'][i]*total_reward**2/num_episodes)
		mutation['biases'][i] = torch.nn.parameter.Parameter(mutation['biases'][i]*total_reward**2/num_episodes)

	return mutation

def update_weights(actor, mutations, lr, std):

	num_mutations = len(mutations)

	# update actor weights
	for i in range(len(actor.weights)):
		dws = []
		dbs = []
		for j in range(num_mutations):
			dws.append(mutations[j]['weights'][i])
			dbs.append(mutations[j]['biases'][i])
		sum_dw = torch.stack(dws, dim=0).sum(dim=0)
		sum_db = torch.stack(dbs, dim=0).sum(dim=0)		
		w = actor.weights[i].weight + lr*sum_dw/(num_mutations*std)
		b = actor.weights[i].bias + lr*sum_db/(num_mutations*std)
		actor.weights[i].weight = torch.nn.parameter.Parameter(w)
		actor.weights[i].bias = torch.nn.parameter.Parameter(b)

def main():

	env = gym.make('CartPole-v0')
	actor = LinearEvoActor()

	print(actor.weights[0].bias.shape)
	print(actor.weights[0].weight.shape)

	genetic_algorithm(env, actor, iters=100, steps=1000, population=5, lr=0.0001, std=0.002, render=False)
	
		

if __name__ == '__main__':
	main()