"""A basic class for td-learning, eligibility(lambda) can be any number between [0, 1].
This class can be used both for off-policy(Q-leaning) and on-policy(SARSA) learning algorithm."""
class TD():
	def __init__(self, alpha, gamma, eligibility, value_func_callback, update_func_callback):
		# hyperparameters
		self.alpha = alpha
		self.gamma = gamma
		self.eligibility = eligibility

		# user of this class must provide value/update callback function
		self.value_func = value_func_callback
		self.update_func = update_func_callback

		self.eligible = {}

	def episode_start(self, start_state):
		self.eligible = {}

	def step(self, state, action, reward, state_next, action_next=None):
		# calculate predict value and target value
		# use value callback to get value of state/action from user of this class
		predict = self.value_func(state, action)
		target = self.value_func(state_next, action_next)
		target *= self.gamma
		target += reward

		# calculate the 'Temporal Difference' between two states
		delta = self.alpha * (target - predict)

		# if TD(0), won't bother to record eligibility
		# so we can just use this function to learn in case of TD(0)
		if self.eligibility == 0:
			self.update_func(state, action, delta)
		else:
			# propagate this difference back on the current episode
			self.eligible[(state, action)] = 1  # not using '+1', to normalize it
			for s_a, e in self.eligible.items():
				self.update_func(s_a[0], s_a[1], delta*e)
				self.eligible[s_a] *= self.eligibility
			

	def episode_end(self):	
		pass
