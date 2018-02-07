from mxnet import nd
from mxnet.gluon import nn
import mxnet as mx
from mxnet import autograd
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
batch_size = 2

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

    # Initialize buffer
    memory = Memory(capacity=buffer_size, dims=2*state_dim + action_dim + 1)

    # Outer iteration
    for i in range(M):

        # Receive initial observation
        s = env.reset()
        explore_variance = 2  # initial exploration variance

        s = nd.array(s).reshape((1, -1))
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

            memory.store_transition(s[0].asnumpy(), action, r, s_)

            if memory.pointer > buffer_size:

                # Decrease exploring area, 1. for 0 decreasing
                explore_variance *= 1.

                # Sample
                batch = memory.sample(batch_size)

                b_s = batch[:, :state_dim]
                b_a = batch[:, state_dim:state_dim + action_dim]
                b_r = batch[:, -state_dim - 1:-state_dim]
                b_s_ = batch[:, -state_dim:]
                b_s, b_a, b_r, b_s_ = nd.array(b_s), nd.array(b_a), nd.array(b_r), nd.array(b_s_)
                # print("The batch is ")
                # print(b_s, "\n", b_a, "\n", b_r, "\n", b_s_, "\n")

                # Get a_target from target_mu network
                a_target = target_mu.net(b_s_)

                # print(b_s_, a_target)

                # Combine the two ndarrays
                combined_sa = nd.concatenate([b_s_, a_target], axis=1)

                q_target = target_q.net(combined_sa)

                y = r + q_target * gamma
                # print(y)




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
