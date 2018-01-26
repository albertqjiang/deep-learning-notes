from Gtg import Gtg
import numpy as np


class ValueIteration:
    """
    Attributes:
        iteration(): Iterate one time to update the value functions(asynchronous)

    """

    def __init__(self, game):
        """
        Args:
            :param game: a game that is a markov decision process
            :type game: obj: Mdp

        """

        self.game = game
        self.vs = np.array([0.] * len(self.game.state_space))
        self.numpy_state_space = np.array(self.game.state_space)

    def iteration(self):
        order = np.arange(len(self.game.state_space))
        np.random.shuffle(order)

        for i in order:
            state = self.game.state_space[i]
            current_reward = self.game.reward(state)

            max_future_expect = -9999.
            for a in self.game.action_space:
                try:
                    ps = self.game.psa(state, a)
                except IndexError:
                    continue
                else:
                    ps = np.array(ps)
                    future_expect = np.sum(np.multiply(ps, self.vs))
                    if future_expect > max_future_expect:
                        max_future_expect = future_expect
            self.vs[i] = current_reward + self.game.gamma * max_future_expect


if __name__ == '__main__':
    game = Gtg(0.5, 3, (0, 0), action=(0, 1))
    vpi = ValueIteration(game)
    for _ in range(100):
        vpi.iteration()
        print(vpi.vs)
    game.step(True)


