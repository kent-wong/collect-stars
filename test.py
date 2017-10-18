# import algorithms
from q_learning import QLearning
from sarsa import Sarsa

# set the environment
env = Env((8, 8), (130, 90), default_rewards=0)

def layout0(env):
	star_credit = 1
	env.add_object('yellow_star', (3, 3), reward=star_credit, pickable=True)
	env.add_object('yellow_star', (7, 1), reward=star_credit, pickable=True)
	env.add_object('yellow_star', (0, 7), reward=star_credit, pickable=True)
	env.add_object('yellow_star', (5, 7), reward=star_credit, pickable=True)
	env.add_object('yellow_star', (6, 6), reward=star_credit, pickable=True)
	env.add_object('yellow_star', (5, 5), reward=star_credit, pickable=True)
	env.add_object('yellow_star', (4, 6), reward=star_credit, pickable=True)
	env.add_object('red_ball', (5, 6), value=0, terminal=True).label = "Exit"

def layout1(env):
	env.add_object('yellow_star', (3, 3), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (0, 7), reward=1000, pickable=True).label = "(1000)"
	env.add_object('red_ball', (5, 6), value=0, terminal=True).label = "Exit"
	
def layout2(env):
	env.agent.born_at = (3, 3)
	env.add_object('yellow_star', (0, 0), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (0, 7), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (7, 0), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (7, 7), reward=100, pickable=True).label = "(100)"
	env.add_object('red_ball', (3, 4), value=0, terminal=True).label = "Exit"
	
def layout3(env):
	env.agent.born_at = (0, 0)
	env.agent.credit = 100
	env.add_object('red_ball', (3, 4), value=0, terminal=True).label = "Exit"

def layout4(env):
	env.agent.born_at = (7, 0)
	env.add_object('yellow_star', (0, 0), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (1, 1), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (2, 2), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (3, 3), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (4, 4), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (5, 5), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (6, 6), reward=100, pickable=True).label = "(100)"
	env.add_object('yellow_star', (7, 7), reward=100, pickable=True).label = "(100)"
	env.add_object('red_ball', (0, 7), value=0, terminal=True).label = "Exit"
	
def layout5(env):
	env.agent.born_at = (5, 0)
	env.add_object('yellow_star', (0, 0), pickable=True)
	env.add_object('yellow_star', (7, 7), pickable=True)
	env.add_object('red_ball', (4, 3), terminal=True).label = "Exit"
	
# use a layout
layout4(env)

# hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.7
lambda_ = 0.7
n_episodes = 10000

rl_algorithm = QLearning(alpha, gamma, epsilon)
#rl_algorithm = Sarsa(alpha, gamma, lambda_, epsilon)

env.show = False
print("training ...")
env.train(rl_algorithm, n_episodes, delay_per_step=0)
env.show = True

#env.show_access_counters()

# wk_debug
print("pickup all:", env.agent.pickup_all)

print("agent is now walking ...")
env.test(rl_algorithm)
#env.test(rl_algorithm, 100, only_exploitation=False)

print("end ...")
#env.remove_object((3, 3))


