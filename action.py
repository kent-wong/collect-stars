import random

class ActionSpace():
	"""A class that describes all the actions that an agent can take in a reinforcement-learning environment"""

	def __init__(self, actions):
		"""
		Args:
			actions: A total set of actions for this action space.
				 Allowed types can be `set`, `list` or `tuple`.
				 If using a sequence-like type, duplicates will be removed by internally
				 converting it to `set` first.
		"""
		self._actions = tuple(set(actions))
		self._n_actions = len(self._actions)

	def sample(self):
		"""Uses an uniform distribution to randomly choose an action"""
		return random.choice(self._actions)

	def _action_at(self, index):
		return self._actions[index]

	def _action_index(self, action):
		return self._actions.index(action)	
		
	def dict_from_actions(self):
		"""Return action dictionary, key is action, value is the corresponding action index
		   in the internal tuple.
		"""
		action_dict = {v:i for i, v in enumerate(self._actions)}
		return action_dict

	@property
	def n_actions(self):
		return self._n_actions

	@property
	def actions(self):
		"""Return actions for iterable access"""
		return self._actions


if __name__ == "__main__":
	a = ActionSpace(('N', 'S', 'W', 'E'))
	print("n_actions:", a.n_actions)
	action_dict = a.dict_from_actions()
	print("action dict:", action_dict)
	
	for _ in range(a.n_actions):
		print(a.sample()) 
