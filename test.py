from env import Env
import time

# set the environment
env = Env((8, 8), (130, 90), default_rewards=0)

def game_map_1(environment):
	environment.add_item('yellow_star', (3, 3), pickable=True)
	environment.add_item('yellow_star', (0, 7), pickable=True)
	environment.add_item('red_ball', (5, 6), terminal=True, label="Exit")

# select a game
game_map_1(env)
for _ in range(100):
	action = env.action_space.sample()
	print(action)
	reward, next, end = env.step(action)
	print(reward, next, end)
	time.sleep(0.2)
env.reset()
