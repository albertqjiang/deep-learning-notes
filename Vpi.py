from Gtg import Gtg
import numpy as np
import time


class ValueIteration:
    """
    Attributes:
        value(state): value function
        max_expectation(state): the maximum expectation of future value functions and the action to reach that
        iteration(None): iterate one time to update the value functions(asynchronous)
        __policy(state): current policy for state
        make_policy(None): based on current value functions, generate corresponding policy
        play_game(None): play the game based on current policy

    """

    def __init__(self, game):
        """
        Args:
            :param game: a game that is a markov decision process
            :type game: obj: Mdp

        """

        self.game = game
        # Value function array
        self.vs = np.array([0.] * len(self.game.state_space))
        # Policy function list
        self.ps = [None] * len(self.game.state_space)

    def value(self, state=None):
        if state is None:
            state = self.game.state
        return self.vs[self.game.state_space.index(state)]

    def max_expectation_action(self, state=None):
        if state is None:
            state = self.game.state
        action_max = None
        expect_max = -9999.

        for a in self.game.action_space:
            try:
                ps = self.game.psa(state, a)
            except IndexError:
                continue
            else:
                ps = np.array(ps)
                future_expect = np.sum(np.multiply(ps, self.vs))
                if future_expect > expect_max:
                    expect_max = future_expect
                    action_max = a
        return action_max, expect_max

    def iteration(self):
        order = np.arange(len(self.game.state_space))
        np.random.shuffle(order)

        for i in order:
            state = self.game.state_space[i]
            current_reward = self.game.reward(state)

            _, maxExpectation = self.max_expectation_action(state=state)
            self.vs[i] = current_reward + self.game.gamma * maxExpectation

    def __policy(self, state=None):
        if state is None:
            state = self.game.state
        return self.ps[self.game.state_space.index(state)]

    def make_policy(self):
        order = np.arange(len(self.game.state_space))
        np.random.shuffle(order)

        for i in order:
            state = self.game.state_space[i]
            self.ps[i], _ = self.max_expectation_action(state=state)

    def play_game(self, delay=0):
        status = ""
        steps = 0
        while status != "Game terminates":
            steps += 1
            self.game.action = self.__policy()
            status = self.game.step(verbose=True)
            print(status)
            time.sleep(delay)
        return steps

    def run_policy_iter_algo(self, iter=100, delay=0):
        for _ in range(iter):
            self.iteration()
        vpi.make_policy()
        steps = vpi.play_game(delay=delay)
        print("Game takes {} steps to terminate".format(steps))



if __name__ == '__main__':
    game = Gtg(0.5, 9, (0, 0), action=(0, 0))
    vpi = ValueIteration(game)
    vpi.run_policy_iter_algo()

