import numpy as np


class mdp:
    """The class for a Markov Decision Process

    Attributes:
        Step (None): Evolve the state for one step

    """

    def __init__(self, gamma, state_space, action_space, psa, reward, state=None, action=None, policy=None):
        """
        Args:
            :param gamma: reward factor:
            :type gamma: int
            :param state_space: set of states:
            :type state_space: list of object
            :param state: current state
            :type state: object
            :param action_space: set of actions
            :type action_space: list of object
            :param action: current action
            :type action: object
            :param psa: transition probability function
            :type psa: func
            :param policy: policy function
            :type policy: func
            :param reward: reward function
            :type reward: func

        """
        self.gamma = gamma
        self.state_space = state_space
        self.state = state
        self.action_space = action_space
        self.action = action
        self.psa = psa
        self.policy = policy
        self.reward = reward

    def step(self):
        print(state_space)
        new_state = state_space[int(np.random.choice(a=range(len(state_space)), size=1, p=self.psa(self.state, self.action)))]
        print(new_state)


if __name__ == '__main__':
    # A very simple example to try to get to the center of a 3 * 3 matrix starting from the upper left corner

    gamma = 0.5
    state_space = [(i, j) for i in range(3) for j in range(3)]
    action_space = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    starting_state = (0, 0)
    action = (1, 0)

    def psa(state, action):
        position = (state[0] + action[0], state[1] + action[1])
        if position in state_space:
            probabilities = [0] * len(state_space)
            probabilities[state_space.index(position)] = 1.
            return probabilities
        else:
            raise Exception('Action not valid')


    def reward(state):
        distance_to_center = np.sqrt((1 - state[0]) ** 2 + (1 - state[1]) ** 2)
        return 1 / (0.1 + distance_to_center)


    game = mdp(gamma=gamma, state_space=state_space, action_space=action_space, psa=psa, reward=reward,
               state=starting_state, action=action)
    game.step()
