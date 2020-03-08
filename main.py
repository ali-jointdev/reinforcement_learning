import settings as s
from random import randrange, choices
import numpy as np
import gym
from gym import spaces
import pygame
import sys


class Environment(gym.Env):
    '''Custom Environment that follows OpenAI Gym interface
    '''
    def __init__(self):
        # World
        self.grid = None
        self.agent = None

        # Reinforcement learning
        super(Environment, self).__init__()
        low_state = np.array(s.LOW_STATE, dtype=np.uint8)
        high_state = np.array(s.HIGH_STATE, dtype=np.uint8)
        self.observation_space = spaces.Box(low_state, high_state)
        self.action_space = spaces.Discrete(s.N_DISCRETE_ACTIONS)

        # Pygame
        self.window = None
        self.step_button = None
        self.toggle_button = None
        self.reset_button = None

    def step(self, action):
        self.agent.take_action(action)
        obs = self.agent.observation()
        reward = self.agent.reward
        done = self.agent.done
        if not done:  # Generate new resources
            resource = choices([0, 1, 2], [30, 2, 1], k=1)[0]
            if resource != 0:
                pos = self._find_empty_cell()
                if pos is not None:
                    row, col = pos
                    self.grid[row][col] = resource
        return obs, reward, done, {}

    def reset(self):
        self.grid = [[0 for _ in range(s.WIDTH)] for _ in range(s.HEIGHT)]
        for row in range(s.HEIGHT):
            for col in range(s.WIDTH):
                if (row == 0 or row == s.HEIGHT-1 or col == 0
                        or col == s.WIDTH-1):
                    self.grid[row][col] = 3  # Boundary barriers
        for i in range(s.WIDTH-4):
            self.grid[s.HEIGHT//2][i] = 3  # Middle barriers
        row, col = self._find_empty_cell()
        self.agent = Agent(self, row, col)
        self.grid[row][col] = self.agent
        resources = [1 for _ in range(s.N_FOOD)]+[2 for _ in range(s.N_MONEY)]
        while len(resources) > 0:
            row, col = self._find_empty_cell()
            self.grid[row][col] = resources.pop()
        return self.agent.observation()

    def render(self):
        if self.window is None:
            self._start_window()
        for row in range(s.HEIGHT):
            for col in range(s.WIDTH):
                self._update_cell(row, col, self.grid[row][col])
        self._update_panel()
        pygame.display.update()

    def start(self, model):
        '''Non-API method to test the trained model

        Arguments:
            model: trained model
        '''
        step, done, paused = False, False, True
        obs = self.reset()
        while 1:
            self.render()
            pygame.time.Clock().tick(s.FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.step_button.collidepoint(event.pos):
                        step = True
                    if self.toggle_button.collidepoint(event.pos):
                        paused = not paused
                    if self.reset_button.collidepoint(event.pos):
                        done, paused = False, True
                        obs = self.reset()
            if not done:
                if not paused:
                    step = True
                if step:
                    action, _states = model.predict(obs)
                    obs, _reward, done, _info = self.step(action)
                    step = False

    def _find_empty_cell(self):
        for _ in range(s.HEIGHT*s.WIDTH):
            row = randrange(s.HEIGHT)
            col = randrange(s.WIDTH)
            if self.grid[row][col] == 0:
                return row, col

    def _start_window(self):
        pygame.init()
        window_width = s.CELL_SIZE*4+(s.CELL_SIZE+1)*s.WIDTH-1
        window_height = (s.CELL_SIZE+1)*s.HEIGHT-1
        self.window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption('Environment Window')
        box = pygame.Rect((0, 0), (s.CELL_SIZE*4, window_height))
        self.window.fill((100, 100, 100), box)
        self.step_button = pygame.draw.rect(self.window, (150, 150, 150),
                                            ((2,
                                              window_height-s.CELL_SIZE+2),
                                            (s.CELL_SIZE*4-4, s.CELL_SIZE-4)))
        self.reset_button = pygame.draw.rect(self.window, (150, 150, 150),
                                             ((2,
                                              window_height-s.CELL_SIZE*2+2),
                                              (s.CELL_SIZE*2-4,
                                               s.CELL_SIZE-4)
                                              )
                                             )
        self.toggle_button = pygame.draw.rect(self.window, (150, 150, 150),
                                              ((s.CELL_SIZE*2+2,
                                               window_height-s.CELL_SIZE*2+2),
                                               (s.CELL_SIZE*2-4,
                                                s.CELL_SIZE-4)
                                               )
                                              )
        font = pygame.font.Font(None, s.CELL_SIZE//2)
        text = font.render('Step', True, (255, 255, 255))
        self.window.blit(text, text.get_rect(center=(s.CELL_SIZE*2,
                         window_height-s.CELL_SIZE/2)))
        text = font.render('Reset', True, (255, 255, 255))
        self.window.blit(text, text.get_rect(center=(s.CELL_SIZE,
                         window_height-s.CELL_SIZE*1.5)))
        text = font.render('Toggle', True, (255, 255, 255))
        self.window.blit(text, text.get_rect(center=(s.CELL_SIZE*3,
                         window_height-s.CELL_SIZE*1.5)))

    def _update_cell(self, row, col, obj):
        pos = (s.CELL_SIZE*4+(s.CELL_SIZE+1)*col, (s.CELL_SIZE+1)*row)
        dims = (s.CELL_SIZE, s.CELL_SIZE)
        pygame.draw.rect(self.window, (240, 240, 240), [pos, dims])
        if obj == 0:
            return
        elif obj == 1:
            self.window.blit(pygame.transform.scale(
                                    pygame.image.load('food.png'), dims), pos)
        elif obj == 2:
            self.window.blit(pygame.transform.scale(
                             pygame.image.load('money.png'), dims), pos)
        elif obj == 3:
            self.window.blit(pygame.transform.scale(
                             pygame.image.load('wall.png'), dims), pos)
        else:
            self.window.blit(pygame.transform.scale(
                             pygame.image.load('agent.png'), dims), pos)

    def _update_panel(self):
        box = pygame.Rect((0, 0), (s.CELL_SIZE*4, s.CELL_SIZE*4))
        self.window.fill((100, 100, 100), box)
        font = pygame.font.Font(None, s.CELL_SIZE//2)
        text = font.render('Remaining steps: '+str(self.agent.rttl),
                           True, (255, 255, 255))
        self.window.blit(text, text.get_rect(top=10, left=10))
        text = font.render('Energy: '+str(self.agent.energy),
                           True, (255, 255, 255))
        self.window.blit(text, text.get_rect(top=10+s.CELL_SIZE/2, left=10))
        text = font.render('Wealth: '+str(self.agent.wealth),
                           True, (255, 255, 255))
        self.window.blit(text, text.get_rect(top=10+s.CELL_SIZE, left=10))


class Agent:
    def __init__(self, env, row, col):
        self.env = env
        self.row = row
        self.col = col
        self.rttl = s.TIME_TO_LIVE
        self.energy = 0
        self.wealth = 0
        self.reward = 0
        self.done = False

    def take_action(self, action):
        if self.rttl == 0:
            self.env.grid[self.row][self.col] = 0
            self.done = True
        else:
            self.rttl -= 1
            direction = action
            self._move(direction)

    def observation(self):
        surroundings = []
        for direction in range(8):
            row, col = self._direction_to_points(direction)
            surroundings.append(self.env.grid[row][col])
        return np.array(surroundings)

    def _move(self, direction):
        row, col = self._direction_to_points(direction)
        obj = self.env.grid[row][col]
        if obj != 3:
            self.env.grid[self.row][self.col] = 0
            self.row, self.col = row, col
            self.env.grid[self.row][self.col] = 4
            self.reward = 0
            if obj == 1:  # Food
                self.energy += 1
                self.reward = 1
            elif obj == 2:  # Money
                self.wealth += 1
                self.reward = 2
        else:  # Barrier
            self.reward = -1

    def _direction_to_points(self, direction):
            row, col = 0, 0
            if direction == 0:  # North
                row, col = self.row-1, self.col
            elif direction == 1:  # North east
                row, col = self.row-1, self.col+1
            elif direction == 2:  # East
                row, col = self.row, self.col+1
            elif direction == 3:  # South east
                row, col = self.row+1, self.col+1
            elif direction == 4:  # South
                row, col = self.row+1, self.col
            elif direction == 5:  # South west
                row, col = self.row+1, self.col-1
            elif direction == 6:  # West
                row, col = self.row, self.col-1
            elif direction == 7:  # North west
                row, col = self.row-1, self.col-1
            return row, col
