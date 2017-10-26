import numpy as np
import time

from env import Env
from dqn import DQN
from chessboard import Chessboard

class DQNSolution():
	def __init__(self, gamma=0.9, lr=0.001):
		# hyperparameters
		self.lr = lr
		self.gamma = gamma

		# create environment
		self.env = Env((8, 8), (160, 90), default_rewards=0)
		self.layout()

		# create DQN
		self.dqn = DQN(3, self.env.action_space.n_actions, [64, 32], lr, gamma, experience_limit=20000)

	def layout(self):
		self.env.add_item('yellow_star', (3, 3), credit=100, pickable=True, label=1)
		self.env.add_item('yellow_star', (0, 7), credit=100, pickable=True, label=2)
		self.env.add_item('red_ball', (5, 6), terminal=True, label="Exit")

	def _centralize_range(self, start, stop=0):
		if stop == 0:
			stop = start
			start = 0
		assert stop >= start
		len = stop - start + 1
		middle = (len-1) / 2
		start = - middle
		stop = middle
		return start, stop

	def _regularize_range(self, start, stop=0):
		begin, end = self._centralize_range(start, stop)
		half = (end-begin) / 2
		result = [i/half for i in np.arange(begin, end+1)]
		return result

	def _regularize_location(self, loc):
		r, c = loc
		height, width = (self.env.map.n_rows, self.env.map.n_columns)
		rows = self._regularize_range(height-1)
		cols = self._regularize_range(width-1)
		return rows[r], cols[c]

	def _compose_state(self, loc, encode):
		agent_row, agent_col = self._regularize_location(loc)
		return [agent_row, agent_col, encode]

	def state(self):    
		map_size = (self.env.map.n_rows, self.env.map.n_columns)
		encode = 0
		for item in self.env.agent.bag_of_objects:
			assert item.label > 0 # `label` must start from 1
			encode += pow(2, item.label-1)

		state = self._compose_state(self.env.agent.at, encode)
		return state

	def show_agent_values(self):
		self._show_action_values_from_loc_state(self.env.agent.at, self.state())
			  
	def _show_action_values_from_loc_state(self, loc, state):
		state = np.array(state)
		matrix_form = state.reshape((1, *state.shape))
		action_values = self.dqn.action_values(matrix_form)[0]  

		text_dict = {}    
		for action_id, value in enumerate(action_values):
			action = self.env.action_space.action_from_id(action_id)
			value = np.round(value, 2)
			text_dict[action] = str(value)

		self.env.draw_values(loc, text_dict)

	def show_all_action_values(self, items_encode):
		self.env.show = True
		rows, cols = (self.env.map.n_rows, self.env.map.n_columns)
		locations = [(row, col) for row in range(rows) for col in range(cols)]
		for loc in locations:
			state = self._compose_state(loc, items_encode)
			self._show_action_values_from_loc_state(loc, state)
			
	def train(self, n_episodes, show=False, delay=0):
		env = self.env # just for convenience
		total_losses = 0
		for episode in range(1, n_episodes+1):
			env.reset()
			env.show = show

			# get initial state and location
			state = self.state()
			location = env.agent.at
			next_location = location

			this_episode = []
			hit_walls = 0
			end = False
			while end == False:
				# show agent's action-values
				if show:
					self.show_agent_values()

				action_id = self.dqn.next_action(state, episode)
				action = env.action_space.action_from_id(action_id)
				reward, next_location, end, _ = env.step(action)
				if show:
					time.sleep(delay)
				if end and len(env.pickable_items()) == 0:
					reward = 100

				next_state = self.state()
				one_step = (state, action_id, reward, next_state, end)
				#dqn.fill_experience(one_step)
				this_episode.append(one_step)

				if location == next_location:
					hit_walls += 1

				if env.steps >= 400:
					break

				# debug info
				#print("state:", state)
				#print("common state:", common_state(state))
				#print("step {}: {} ----> {} {} reward {}\n".format(env.steps, location, next_location, action, reward))
				#print(next_state)
				state = next_state
				location = next_location

			# train a whole episode
			loss = self.dqn.train_an_episode(this_episode)
			total_losses += loss
			if reward > 0: # train again
				self.dqn.train_an_episode(this_episode)


			#print("items remain in map:", len(env.pickable_items()))
			print("# {}: batch avg loss is {:.4f}, steps/hit walls:{}/{}, stars {}, reward is {}".format(
				episode, 
				loss,
				env.steps,
				hit_walls,
				len(env.agent.bag_of_objects),
				reward))
			#env.show = True
			#show_all_state2(env, 0)

	def test(self, action_func, delay=1):
		env = self.env
		env.reset()
		env.show = True
		end = False
		while end == False:
		    # debug
		    self.show_agent_values()

		    state = self.state()
		    action_id = action_func(state)
		    action = self.env.action_space.action_from_id(action_id)

		    reward, next_location, end, _ = env.step(action)
		    time.sleep(delay)


if __name__ == "__main__":
	solution =solution = DQNSolution(gamma=0.9, lr=0.001) 
	solution.train(1000, show=False, delay=0.1)

	#solution.test(solution.dqn.best_action, delay=0.5)
	solution.show_all_action_values(0)
