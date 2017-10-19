import numpy as np
import tensorflow as tf

from memory import Memory

class NeuralNetwork():
	def __init__(self, n_features, n_actions, hidden_layers, lr=0.001):
		self.n_features = n_features
		self.n_actions = n_actions
		self.lr = lr

		# tensors
		self.inputs__ = None
		self.actions__ = None
		self.targets__ = None
		self.output = None
		self.predict = None
		self.loss = None
		self.opt = None

		# session
		self.sess = tf.Session()

		# build the network
		self._build(hidden_layers)

	def _build(self, hidden_layer_sizes):
		self.inputs__ = tf.placeholder(tf.float32, [None, self.n_features], name='inputs')
		self.actions__ = tf.placeholder(tf.int32, [None], name='actions')
		one_hot_actions = tf.one_hot(self.actions__, self.n_actions)

		self.targets__ = tf.placeholder(tf.float32, [None], name='targets')

		# build hidden layers
		next_layer_inputs = self.inputs__
		for layer_size in hidden_layer_sizes:
			next_layer_inputs = tf.contrib.layers.fully_connected(next_layer_inputs, layer_size)

		# build linear output layers
		self.output = tf.contrib.layers.fully_connected(next_layer_inputs, self.n_actions, activation_fn=None)

		# only retain the action-value we want
		output_filtered = tf.multiply(self.output, one_hot_actions)
		self.predict = tf.reduce_sum(output_filtered, axis=1)

		# use `predict` and `target__` to calculate loss
		self.loss = tf.reduce_mean(tf.square(self.targets__ - self.predict))
		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def output(self, state):
		feed = {self.inputs__: state.reshape(1, *state.shape)}
		qtable_row = self.sess.run(self.output, feed_dict=feed)	
		return qtable_row

class DQN():
	def __init__(self, n_features, n_actions, lr=0.001, gamma=0.99):
		self._n_features = n_features
		self._n_actions = n_actions

		self.nn = NeuralNetwork(n_features, n_actions, [10, 10], lr)

		# memory of episodes
		self.memory = Memory(1000)

		# `action` selection algorithm parameters
		self.explore_start = 0.9
		self.explore_stop = 0.1
		self.decay_rate = 0.0001

		# training
		self.episodes = 0

		# hyperparameters
		self.lr = lr
		self.gamma = gamma

	def action_values(self, state, action=None):	
		output = self.nn.output(state)
		if action != None:
			output = ouput[action]
		return output

	def next_action(self, state):
		explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*self.episodes)
		if np.random.rand() < explore_p:  # should go to explore
			action = np.random.choice(self.n_actions)
		else:
			output = self.action_values(state)
			action = np.argmax(output)
		return action

	def fill_memory(self, episode):
		self.memory.add(episode)

	def train_an_episode(self, episode):
		states = np.array([step[0] for step in episode])
		action = np.array([step[1] for step in episode])
		reward = np.array([step[2] for step in episode])
		next_states = np.array([step[3] for step in episode])

		action_values = self.action_values(next_states)

		# the last one is `terminal` point, mark it using 0
		action_values[-1] = (0, ) * self.n_actions

		targets = rewards + self.gamma * np.max(action_values, axis=1)

		# training ...
		self.episodes += 1
		feed = {self.inputs__: states, self.actions__: actions, self.targets__: targets}
		loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed)
			
		return loss

	def learn_from_experience(self, batch_size):
		batch = self.memory.sample(batch_size)
		for episode in batch:
			self.train_an_episode(episode)


if __name__ == "__main__":
	dqn = DQN(10, 2)
