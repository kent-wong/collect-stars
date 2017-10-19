"""A class representing the underlying chessboard of the environment"""
class Chessboard():
	def __init__(self, size):
		self._rows, self._columns = size
		self.n_squares = self._rows * self._columns  # just for convenience
		self._items = {}

	def _is_valid_square_index(self, index):
		r, c = index
		return (r >= 0) and (r < self._rows) and (c >= 0) and (c < self._columns)

	def item_at(self, index):
		assert self._is_valid_square_index(index)
		return self._items.get(index)

	def put_item(self, index, item):
		assert self._is_valid_square_index(index)
		self._items[index] = item

	def delete_item(self, index):
		assert self._is_valid_square_index(index)
		try:
			del self._items[index]
		except KeyError:
			pass

	@property
	def n_rows(self):
		return self._rows

	@property
	def n_columns(self):
		return self._columns

	@property
	def all_items(self):
		return self._items

	@property
	def index_space(self):
		return [(row, column) for row in range(self.n_rows) for column in range(self.n_columns)]


if __name__ == "__main__":
	from env_item import Item

	chess = Chessboard((4, 6))
	print("chessboard rows and columns: ({}, {}):".format(chess.n_rows, chess.n_columns))
	print("item at (3, 3):", chess.item_at((3, 3)))
	item = Item('yellow_star', (3, 3), pickable=True)
	item.label = "new star"
	chess.put_item(item.index, item)
	print("item at (3, 3):", chess.item_at((3, 3)))
	print("index space:")
	print(chess.index_space)
