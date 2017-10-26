import numpy as np
import tkinter as tk
import time
import random
import sys

from chessboard import Chessboard
from env_item import Item
from drawing import DrawingManager
from action import ActionSpace
from agent import Agent

class Env():
	def __init__(self, dimension=(5, 5), square_size=(80, 80), default_rewards=0, agent_born_at=(0, 0), agent_born_facing='E'):
		# create a chessboard-style map for the environment
		self.map = Chessboard(dimension)
		self.default_rewards = default_rewards

		# set walls, user can still call .set_walls() to add new walls
		self.walls = []
		rows, columns = dimension
		for row in range(rows):
			self.walls.append((row, -1))
			self.walls.append((row, columns))
		for column in range(columns):
			self.walls.append((-1, column))
			self.walls.append((rows, column))

		# create a drawing manager, which creates a tkinter window in a new thread
		self.drawing_manager = DrawingManager(dimension, square_size).wait()

		# draw chessboard
		self.drawing_manager.draw_chessboard()

		# create an action space
		self.action_space = ActionSpace(('N', 'S', 'W', 'E'))

		# create agent and set its born location and facing direction
		self.agent = Agent(agent_born_at, agent_born_facing)

		# whether environment changes should be displayed on screen
		self._show = True

		# track total steps agent has walked
		self._steps = 0

		# reset environment
		self.reset()

	@property
	def show(self):
		return self._show

	@show.setter
	def show(self, onoff):
		self._show = onoff

	@property
	def steps(self):
		return self._steps

	def _draw_item(self, item):
		self.drawing_manager.draw_object(item.name, item.index)
		if item.label != None:
			self.drawing_manager.draw_text(item.index, {'C':item.label})

	def _remove_item(self, item):
		self.drawing_manager.delete_object(item.name, item.index)
		self.drawing_manager.delete_text(item.index)

	def _reset_agent_drawing(self):
		if self.show == True:
			self.drawing_manager.remove_agent()
			self.drawing_manager.draw_agent(self.agent.born_at, self.agent.born_facing)

			# delete previous traces so we don't mess up the environment
			self.drawing_manager.delete_trace()

	def _draw_agent_step(self, next_index, facing):
		if self.show == True:
			if self.agent.facing != facing:
				self.drawing_manager.rotate_agent(self.agent.at, facing)
			if self.agent.at != next_index:
				self.drawing_manager.move_agent(self.agent.at, next_index)
				self.drawing_manager.draw_trace(self.agent.at, facing)

	def add_item(self, name, index, credit=0, terminal=False, pickable=False, label=None):
		"""Add an object to the environment, currently only one object is allowed per chessboard square.
			Args:
				name: A object name string.
				index: A tuple `(row, column)` of chessboard square.
		"""
		item = Item(name, index, credit, terminal, pickable, label)
		self.map.put_item(item.index, item)

		if self.show:
			self._draw_item(item)

		return item

	def random_move(self, index):
		item = self.map.item_at(index)
		if item is None:
			return
		assert item.index == index

		action = self.action_space.sample()
		next_index = self._item_move_by_order(index, action)
		if item.index != next_index:
			# move the item in chessboard
			self.map.put_item(index, None)
			assert self.map.item_at(next_index) is None
			self.map.put_item(next_index, item)

			# redraw item
			self._remove_item(item)
			item.index = next_index
			self.redraw_item(item)

		return next_index

	def redraw_item(self, item):
		if self.show:
			self._draw_item(item)

	def remove_item(self, index):
		item = self.map.item_at(index)
		if item != None:
			self.map.delete_item(index)
			if self.show:
				self._remove_item(item)

		return item

	def pickable_items(self):
		items = []
		for item in self.map.all_items:
			if item.pickable == True:
				items.append(item)
		return items
			
	def let_agent_pickup(self, index):
		item = self.map.item_at(index)
		if item != None:
			# if this item is pickable, let agent pick it up
			if item.pickable == True:
				self.agent.pickup(item)
				self.remove_item(index)
		return item

	def let_agent_random_pickup(self, n=1):
		"""let agent randomly pickup at most `n` items"""
		items = self.pickable_items()
		if len(items) < n:
			n = len(items)

		picked = 0
		while picked < n:
			i = np.random.randint(len(items))
			self.let_agent_teleport(items[i].index)
			picked += 1
		return picked

	def let_agent_teleport(self, dest):
		if self.show:
			self.drawing_manager.remove_agent()
			self.drawing_manager.draw_agent(dest, 'E')
		self.agent.at = dest
		self.let_agent_pickup(dest)

	def let_agent_random_teleport(self):
		squares = self.map.squares_list
		while True:
			n = np.random.randint(len(squares))
			dest = squares[n]
			item = self.map.item_at(dest)
			if item is None or item.terminal != True:
				break
		return self.let_agent_teleport(dest)

	def draw_text(self, index, text_dict):
		if self.show:
			self.drawing_manager.draw_text(index, text_dict)

	def show_action_values(action_values):
		text_dict = {}    
		for action_id, value in enumerate(action_values):
			action = self.action_space.action_from_id(action_id)
			value = np.round(value, 2)
			text_dict[action] = str(value)

		self.draw_text(location, text_dict)

	def set_walls(self, walls):
		for wall in walls:
			if self._is_hit_wall(wall) == False:
				self.walls.append(wall)

	def _is_hit_wall(self, index):
		return index in self.walls

	def _move_by_order(self, start, actions):
		move = {"N":(-1, 0), "S":(1, 0), "W":(0, -1), "E":(0, 1)}
		destination = np.array(start)
		next_step = np.array(start)

		if isinstance(actions, (list, tuple)) == False:
			actions = [actions]
			
		for action in actions:
			next_step += np.array(move[action])
			if self._is_hit_wall((int(next_step[0]), int(next_step[1]))) == True:
				break
			else:
				destination += np.array(move[action])

		return (int(destination[0]), int(destination[1]))

	def _item_move_by_order(self, start, action):
		move = {"N":(-1, 0), "S":(1, 0), "W":(0, -1), "E":(0, 1)}
		next_step = np.array(start)

		next_step += np.array(move[action])
		#if self._is_hit_wall((int(next_step[0]), int(next_step[1]))) == True:
			#break

		if self._is_hit_wall(next_step) or self.map.item_at(next_step) is None:
			return start
		else:
			return next_step


	def step(self, action):
		agent_loc = self.agent.at
		next_index = self._move_by_order(agent_loc, action)

		#self.increase_access_counter(cur_index)

		item = self.map.item_at(agent_loc)
		if item != None:
			# if currently agent is on `terminal point`, then do nothing
			if item.terminal == True:
				return (0, agent_loc, True)
			else:
				reward = item.reward
		else:
			reward = self.default_rewards

		# move agent to next location
		self._draw_agent_step(next_index, action)
		self.agent.step_to(next_index, action)

		item = self.map.item_at(next_index)
		if item != None:
			# `terminal point` must be an item, empty square can't be terminal
			terminal = item.terminal

			# if this item is pickable, let agent pick it up
			if item.pickable == True:
				self.agent.pickup(item)
				self.remove_item(next_index)
		else:
			terminal = False

		bag_of_items = []
		if terminal == True:
			reward += self.agent.credit
			bag_of_items = self.agent.bag_of_objects

		self._steps += 1
		return (reward, next_index, terminal, bag_of_items)

	def reset(self):
		self.drawing_manager.delete_all_values()

		self.agent.reset()
		bag_of_items = self.agent.drop()
		for item in bag_of_items:
			self.map.put_item(item.index, item)
			self.redraw_item(item)

		self._reset_agent_drawing()
		self._steps = 0

	def _simple_state_generator(self, env):
		return self.agent.at

	def _simple_action_generator(self, state):
		return self.action_space.sample()

	def create_episodes(n_episodes, state_generator=None, action_generator=None):
		assert n_episodes > 0
		if state_generator is None:
			state_generator = _simple_state_generator
		if action_generator is None:
			action_generator = _simple_state_generator

		total_episodes = []
		for ep in n_episodes:
			this_episode = []
			self.reset()
			state = state_generator(self)

			end = False
			while end != False:
				action = action_generator(state)
				action_id = env.action_space.action_id(action)
				reward, next_index, end = self.step(action)
				next_state = state_generator(self)

				one_step = (state, action_id, reward, next_state, end)
				this_episode.append(one_step)
				
			total_episodes += this_episode
		return total_episodes


if __name__ == '__main__':
	# set the environment
	env = Env((8, 8), (130, 90), default_rewards=0)
	env.add_item('yellow_star', (3, 3), credit=100, pickable=True, label="1")
	env.add_item('yellow_star', (0, 7), credit=1000, pickable=True, label="2")
	env.add_item('red_ball', (5, 6), terminal=True, label="Exit")

	for _ in range(10):
		time.sleep(0.5)
		#env.let_agent_random_teleport()
		env.let_agent_random_pickup(2)
		time.sleep(0.5)

	for _ in range(10):
		action = env.action_space.sample()
		print(action)
		reward, next, end, _ = env.step(action)
		print(reward, next, end)
		time.sleep(0.2)
	env.reset()
