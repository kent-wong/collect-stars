from collections import deque
import numpy as np

class Memory():
	def __init__(self, maxlen=1000):
		self._memory = deque(maxlen=maxlen)

	def add(self, record):
		self._memory.append(record)

	def sample(self, n):
		assert n > 0

		if self.n_records < n:
			return None

		# 'deque' object has O(n) for indexed access except for both ends(that is O(1))
		# so first convert it to a list
		a_list = list(self._memory)

		if self.n_records < n:
			n = self.n_records

		choices = np.random.choice(range(self.n_records), n, replace=False)

		return [a_list[i] for i in choices]

	@property
	def n_records(self):
		return len(self._memory)

	@property
	def is_full(self):
		return self.n_records == self._memory.maxlen


if __name__ == "__main__":
	m = Memory()	
	for _ in range(100):
		m.add([[np.random.randint(1000)]*3])

	a = m.sample(10)
	print("random choice:")
	print(a)
	print("current records:", m.n_records)
	print("memory is full?", m.is_full)
	m.add(range(1000))
	print("current records:", m.n_records)
	print("memory is full?", m.is_full)
