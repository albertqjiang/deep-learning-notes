from mxnet import nd
from mxnet.gluon import nn
import mxnet as mx
import gym
import numpy as np

from actor_network import ActorNetwork, TargetNetMu
from critic_network import CriticNetwork, TargetNetQ
from replay_buffer import Memory

np.random.seed(1)
mx.random.seed(1)

M = 70  # outer loop iteration
T = 400  # inner loop iteration
tau = 0.01  # update lagging coefficient
gamma = 0.99  # discount factor

ENV_NAME = 'Pendulum-v0'


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    env.seed(1)
    env = env.unwrapped

    # Get state and action dimension
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize actor, critic and target networks
    actor = ActorNetwork(action_dim)
    critic = CriticNetwork()
    target_a = TargetNetMu(actor)
    target_mu = TargetNetQ(critic)












    # Initialize the actor, critic and two target networks
    actor = ActorNetwork(1)
    critic = CriticNetwork(1)
    target_mu = TargetNetMu(actor)
    tar_q = TargetNetQ(critic)

    # Test whether parameters are copied correctly
    test_input = nd.array([1])
    assert nd.equal(actor.net(test_input), target_mu.net(test_input))[0][0] == 1.
    assert nd.equal(critic.net(test_input), tar_q.net(test_input))[0][0] == 1.

    # Initialize memory buffer
    memory = Memory(capacity=7000, dims=8)

    for m in range(M):

        # Start new environment
        # To do: inner implementation of DDPG
        pass
