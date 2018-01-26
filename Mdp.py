import numpy as np


class Mdp:
    """The class for a Markov Decision Process

    Attributes:
        step(None): Evolve the state for one step
        show(None): Print current state
    """

    def __init__(self, gamma, state_space, action_space, psa, reward, state=None, action=None, policy=None):
        """
        Args:
            :param gamma: reward factor:
            :type gamma: float
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

    def step(self, verbose=False):
        new_state = self.state_space[
            int(np.random.choice(a=range(len(self.state_space)), size=1, p=self.psa(self.state, self.action)))]
        self.state = new_state
        if verbose:
            print("The new state is", new_state)
            self.show()

    def show(self):
        pass


if __name__ == '__main__':
    # A very simple example to try to get to the center of a 3 * 3 matrix starting from the upper left corner

    gamma = 0.5
    state_space = [(i, j) for i in range(3) for j in range(3)]
    action_space = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    starting_state = (0, 0)
    action = (1, 0)


    def psa(state, action):
        position = (state[0] + action[0], state[1] + action[1])
        if position in state_space and action in action_space:
            probabilities = [0] * len(state_space)
            probabilities[state_space.index(position)] = 1.
            return probabilities
        else:
            raise IndexError('Action not valid')


    def reward(state):
        distance_to_center = np.sqrt((1 - state[0]) ** 2 + (1 - state[1]) ** 2)
        return 1 / (0.1 + distance_to_center)


    game = Mdp(gamma=gamma, state_space=state_space, action_space=action_space, psa=psa, reward=reward,
               state=starting_state, action=action)
    game.step(True)
