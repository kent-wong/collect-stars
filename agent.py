class Agent():
	def __init__(self, born_at=(0, 0), born_facing='E'):
		# save parameters and don't change after init
		self._born_at = born_at
		self._born_facing = born_facing

		# current location and facing, reset after each episode
		self._at = born_at
		self._facing = born_facing

		# credit
		self._credit = 0

		# bag that stores objects this agent picked up
		self.bag = []

	def reset(self):
		self._at = self.born_at
		self._facing = self.facing

	def step_to(self, to_index, facing):
		self._at = to_index
		self._facing = facing

	def pickup(self, item):
		self.bag.append(item)

		self.credit *= 10
		self.credit += item.credit

	def drop(self):
		objs = self.bag
		self.bag = []
		self.credit = 0
		return objs

	@property
	def credit(self):
		return self._credit

	@credit.setter
	def credit(self, value):
		self._credit = value

	@property
	def at(self):
		return self._at

	@at.setter
	def at(self, where):
		self._at = where
	
	@property
	def facing(self):
		return self._facing

	@facing.setter
	def facing(self, facing):
		self._facing = facing

	@property
	def bag_of_objects(self):
		return self.bag

	@property
	def born_at(self):
		return self._born_at

	@born_at.setter
	def born_at(self, where):
		if where is None:
			where = (0, 0)  # default
		self._born_at = where
		self.reset()  # set current location to the born place and redraw agent

	@property
	def born_facing(self):
		return self._born_facing

	@born_facing.setter
	def born_facing(self, facing):
		if facing is None:
			facing = 'E'
		self._born_facing = facing
		self.reset()



