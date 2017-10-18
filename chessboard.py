"""A class representing the underlying chessboard of the environment"""
class Chessboard():
	def __init__(self, size):
		self._height, self._width = size[0], size[1]
		self.n_squares = self._height * self._width  # just for convenience
		self._squares = {}

	def _is_valid_square_index(self, index):
		h, w = index[0], index[1]
		return (h >= 0) and (h < self._height) and (w >= 0) and (w < self._width)

	def square_at(self, index):
		assert self._is_valid_square_index(index)
		return self._squares.get(index)

	def set_square(self, index, a_square):
		assert self._is_valid_square_index(index)
		self._squares[index] = a_square

	@property
	def height(self):
		return self._height

	@property
	def width(self):
		return self._width

	@property
	def squares(self):
		return self._squares


class Square():
	"""A class represents an object(image or text) that resides on a chessboard square"""

	def __init__(self, name, index, terminal=False, pickable=False, data=None):
		self._name = name
		self._index = index
		self._data = data
		self._terminal = terminal
		self._pickable = pickable
		self._label = None

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
	def data(self):
		return self._data

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
	def label(self, text):
		self._label = text
		#if text != None:
		#	self.drawing_manager.draw_text(self.index_or_id, {'C':text})

	def __str__(self):
		desc = "`" + self.name + "` at ({}, {})".format(self.index[0], self.index[1])
		if self.label != None:
			desc += " with label `{}`".format(self.label)

		if self.terminal:
			desc += ", terminal"
		if self.pickable:
			desc += ", pickable"

		return desc

if __name__ == "__main__":
	chess = Chessboard((8, 10))
	print("chessboard height and width: ({}, {}):".format(chess.height, chess.width))
	print("square at (3, 3):", chess.square_at((3, 3)))
	sq = Square('yellow_star', (3, 3), pickable=True)
	sq.label = "new star"
	chess.set_square(sq.index, sq)
	print("square at (3, 3):", chess.square_at((3, 3)))
