import numpy as np
from Mdp import Mdp


class Gtg(Mdp):
    """The simple game of Getting To the Goal of a matrix by moving 1 step at a time

    Example of a game: * is the current position and o is the destination
        * + +
        + o +
        + + +

    Attributes:
        psa(list:float): deterministic transition probability function
        reward(float): reward function
        show(None): print current state
        determinates(None): tell whether the game should be terminated

    """

    def __init__(self, gamma, side, starting_state, action=None, goal=None):
        self.gamma = gamma
        self.side = side
        self.state_space = [(i, j) for i in range(self.side) for j in range(self.side)]
        self.action_space = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
        if action is None:
            self.action = (0, 0)
        else:
            self.action = action
        self.starting_state = starting_state
        self.state = self.starting_state
        if goal is None:
            self.goal = (int(side / 2), int(side / 2))

    def psa(self, state=None, action=None):
        if state is None:
            state = self.state
        if action is None:
            action = self.action
        position = (state[0] + action[0], state[1] + action[1])
        if position in self.state_space and self.action in self.action_space:
            probabilities = [0] * len(self.state_space)
            probabilities[self.state_space.index(position)] = 1.
            return probabilities
        else:
            raise IndexError('Action not valid')

    def reward(self, state=None):
        if state is None:
            state = self.state
        distance_to_goal = np.sqrt((self.goal[0] - state[0]) ** 2 + (self.goal[0] - state[1]) ** 2)
        return 1 / (0.1 + distance_to_goal)

    def show(self):
        matrix = []
        for i in range(self.side):
            row = []
            for j in range(self.side):
                row.append('+')
            matrix.append(row)
        if self.state != self.goal:
            matrix[self.state[0]][self.state[1]] = '*'
            matrix[self.goal[0]][self.goal[1]] = 'o'
        else:
            matrix[self.state[0]][self.state[1]] = 'x'
        print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                         for row in matrix]))

    def terminates(self):
        if self.state == self.goal:
            return True

if __name__ == '__main__':
    game = Gtg(0.5, 9, (0, 0), action=(0, 1))
    game.step(True)
    print(game.reward())
