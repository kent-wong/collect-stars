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
			next_layer_inputs = tf.contrib.layers.fully_connected(next_layer_inputs, layer_size, 
							weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
							activation_fn=tf.nn.relu)

		# build linear output layers
		self.output = tf.contrib.layers.fully_connected(next_layer_inputs, self.n_actions, 
							weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
							activation_fn=None)

		# only retain the action-value we want
		output_filtered = tf.multiply(self.output, one_hot_actions)
		self.predict = tf.reduce_sum(output_filtered, axis=1)

		# use `predict` and `target__` to calculate loss
		self.loss = tf.reduce_mean(tf.square(self.targets__ - self.predict))
		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

		# session
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def nn_output(self, states):
		feed = {self.inputs__: states}
		qtable_row = self.sess.run(self.output, feed_dict=feed)	
		return qtable_row

class DQN():
	def __init__(self, n_features, n_actions, hidden_layers, lr=0.001, gamma=0.99, experience_limit=None):
		self.n_features = n_features
		self.n_actions = n_actions

		self.nn = NeuralNetwork(n_features, n_actions, hidden_layers, lr)

		# memory of episodes
		self.experience = Memory(maxlen=experience_limit)

		# `action` selection algorithm parameters
		self.explore_start = 1.0
		self.explore_stop = 0.2
		self.decay_rate = 0.0001

		# hyperparameters
		self.lr = lr
		self.gamma = gamma

	def action_values(self, states, action=None):	
		output = self.nn.nn_output(states)
		if action != None:
			output = output[action]
		return output

	def best_action(self, state):
		state = np.array(state)
		matrix_form = state.reshape((1, *state.shape))
		output = self.action_values(matrix_form)[0]
		action = np.argmax(output)

		return action

	def next_action(self, state, n_episodes=0):
		explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*n_episodes)
		if np.random.rand() < explore_p:  # should go to explore
			action = np.random.choice(self.n_actions)
		else:
			action = self.best_action(state)
		return action

	def fill_experience(self, exp):
		self.experience.add(exp)

	def clear_experience(self):
		self.experience.clear()

	def train_batch_states(self, batch_size):
		batch = self.experience.sample(batch_size)

		states = np.array([step[0] for step in batch])
		actions = np.array([step[1] for step in batch])
		rewards = np.array([step[2] for step in batch])
		next_states = np.array([step[3] for step in batch])
		ends = np.array([step[4] for step in batch])

		# query NN to get action-values
		action_values = self.action_values(next_states)

		# if it is `terminal` point, set its action-value to 0
		action_values[ends] = (0, ) * self.n_actions

		targets = rewards + self.gamma * np.max(action_values, axis=1)

		# training ...
		feed = {self.nn.inputs__: states, self.nn.actions__: actions, self.nn.targets__: targets}
		loss, _ = self.nn.sess.run([self.nn.loss, self.nn.opt], feed_dict=feed)
			
		return loss

	def train_an_episode(self, episode):
		states = np.array([step[0] for step in episode])
		actions = np.array([step[1] for step in episode])
		rewards = np.array([step[2] for step in episode])
		next_states = np.array([step[3] for step in episode])
		ends = np.array([step[4] for step in episode])

		action_values = self.action_values(next_states)

		# if it is `terminal` point, set its action-value to 0
		action_values[ends] = (0, ) * self.n_actions

		targets = rewards + self.gamma * np.max(action_values, axis=1)

		# training ...
		feed = {self.nn.inputs__: states, self.nn.actions__: actions, self.nn.targets__: targets}
		loss, _ = self.nn.sess.run([self.nn.loss, self.nn.opt], feed_dict=feed)
			
		return loss

	def train_batch(self, episodes):
		all_states = None
		all_actions = None
		all_targets = None

		for episode in episodes:
			states = np.array([step[0] for step in episode])
			actions = np.array([step[1] for step in episode])
			rewards = np.array([step[2] for step in episode])
			next_states = np.array([step[3] for step in episode])

			action_values = self.action_values(next_states)

			# the last one is `terminal` point, mark it using 0
			action_values[-1] = (0, ) * self.n_actions

			targets = rewards + self.gamma * np.max(action_values, axis=1)

			# agent is not moving in some steps(hit a wall), corresponding action should be zero-valued
			#stay = (states == next_states).all(axis=1)
			#targets[stay] = 0

			# wk_debug
			#for i in stay:
			#	if i == True:
			#		print(i, "end=")
			#print()

			#print("stay:", stay)
			#print("targets:", targets)

			if all_states is None:
				all_states = states
				all_actions = actions
				all_targets = targets
			else:
				# concatenate
				all_states = np.concatenate((all_states, states))
				all_actions = np.concatenate((all_actions, actions))
				all_targets = np.concatenate((all_targets, targets))

		# training batch ...
		feed = {self.nn.inputs__: all_states, self.nn.actions__: all_actions, self.nn.targets__: all_targets}
		losses, _ = self.nn.sess.run([self.nn.loss, self.nn.opt], feed_dict=feed)
			
		return losses

	def learn_from_experience(self, batch_size):
		losses = 0
		batch = self.experience.sample(batch_size)
		#for episode in batch:
			#losses += self.train_an_episode(episode)

		losses = self.train_batch(batch)
		return losses


if __name__ == "__main__":
	dqn = DQN(10, 2)
