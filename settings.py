''' Environment setup '''
HEIGHT = 10
WIDTH = 10
N_FOOD = 6
N_MONEY = 3
TIME_TO_LIVE = 100

''' Pygame settings '''
FPS = 10
CELL_SIZE = 50

''' Reinforcement learning settings '''
N_DISCRETE_ACTIONS = 8  # Move for each direction
LOW_STATE = [0, 0, 0, 0, 0, 0, 0, 0, ]
HIGH_STATE = [4, 4, 4, 4, 4, 4, 4, 4, ]
