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
buffer_size = 5000  # buffer capacity


ENV_NAME = 'Pendulum-v0'


if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    env.seed(1)
    env = env.unwrapped

    # Get state and action dimension
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize actor, critic and target networks
    actor = ActorNetwork(action_dim=action_dim)
    critic = CriticNetwork()
    target_mu = TargetNetMu(actor)
    target_q = TargetNetQ(critic)

    # Testing network initializations
    # test_array = nd.array([1.])
    # assert actor.net(test_array)[0][0] == -1.2251712
    # assert target_mu.net(test_array)[0][0] == -1.2251712
    # assert critic.net(test_array)[0][0] == -0.9443339
    # assert target_q.net(test_array)[0][0] == -0.9443339

    # Initialize buffer
    memory = Memory(capacity=buffer_size, dims=2*state_dim + action_dim + 1)

    # Outer iteration
    for i in range(M):

        # Receive initial observation
        s = env.reset()
        explore_variance = 2  # initial exploration variance

        s = nd.array(s).reshape((1, -1))[0]
        # print(s)
        # Inner iteration
        for j in range(T):

            # Generate action from action net and add exploring variation
            action = actor.net(s)
            action = action[0].asscalar()
            action = nd.clip(nd.random.normal(action, explore_variance), -2, 2)
            action = action.asnumpy()

            # Get info of next state
            s_, r, done, info = env.step(action)

            memory.store_transition(s.asnumpy(), action, r, s_)
            if memory.pointer > buffer_size:

                # Decrease exploring area, 1. for 0 decreasing
                explore_variance *= 1.

                # Get target a from target_mu network
                a_target = target_mu.net(nd.array(s_))
                # Transform s(i+1) into handy ndarray
                nded_s_ = nd.array([s_]).reshape((-1, 1))
                # Combine the two ndarrays
                combined_sa = nd.concatenate([nded_s_, a_target]).reshape((1, -1))

                # print(combined_sa)
                q_target = target_q.net(combined_sa)
                print(q_target)



    # # Initialize the actor, critic and two target networks
    # actor = ActorNetwork(1)
    # critic = CriticNetwork(1)
    # target_mu = TargetNetMu(actor)
    # tar_q = TargetNetQ(critic)
    #
    # # Test whether parameters are copied correctly
    # test_input = nd.array([1])
    # assert nd.equal(actor.net(test_input), target_mu.net(test_input))[0][0] == 1.
    # assert nd.equal(critic.net(test_input), tar_q.net(test_input))[0][0] == 1.
    #
    # # Initialize memory buffer
    # memory = Memory(capacity=7000, dims=8)
    #
    # for m in range(M):
    #
    #     # Start new environment
    #     # To do: inner implementation of DDPG
    #     pass
