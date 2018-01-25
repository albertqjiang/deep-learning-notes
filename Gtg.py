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

    """

    def __init__(self, gamma, side, starting_state, action = None, goal=None):
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
            self.goal = (int(side/2), int(side/2))

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
            return None

    def reward(self, state=None):
        if state is None:
            state = self.state
        distance_to_goal = np.sqrt((self.goal[0] - state[0]) ** 2 + (self.goal[0] - state[1]) ** 2)
        return 1 / (0.1 + distance_to_goal)


if __name__ == '__main__':
    game = Gtg(0.5, 9, (0, 0), action = (0, 1))
    game.step()
    print(game.reward())