class Item():
	"""A class represents an item(image or text) of environment"""

	def __init__(self, name, index, credit=0, terminal=False, pickable=False, label=None):
		self._name = name
		self._index = index
		self._credit = credit
		self._terminal = terminal
		self._pickable = pickable
		self._label = label

	#def draw(self):
	#	if self.drawing_manager != None:
	#		self.drawing_manager.draw_object(self.obj_type, self.index_or_id)
	#		if self.label != None:
	#			self.drawing_manager.draw_text(self.index_or_id, {'C':self.label})

	#def remove(self):
	#	if self.drawing_manager != None:
	#		self.drawing_manager.delete_object(self.obj_type, self.index_or_id)
	#		self.drawing_manager.delete_text(self.index_or_id)

	@property
	def name(self):
		return self._name

	@property
	def credit(self):
		return self._credit

	@property
	def terminal(self):
		return self._terminal
	
	@property
	def pickable(self):
		return self._pickable

	@property
	def index(self):
		return self._index

	@index.setter
	def index(self, where):
		self._index = where

	@property
	def label(self):
		return self._label

	@label.setter
	def label(self, label):
		self._label = label

	def __str__(self):
		desc = "`" + self.name + "` at ({}, {})".format(self.index[0], self.index[1])
		if self.label != None:
			desc += " with label `{}`".format(self.label)

		if self.terminal:
			desc += ", terminal"
		if self.pickable:
			desc += ", pickable"

		return desc
