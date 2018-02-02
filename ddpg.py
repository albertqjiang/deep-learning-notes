import mxnet as mx
import numpy as np
import gym
import mxnet.gluon
from mxnet.gluon import nn


class Actor(object):
    def __init__(self, action_dim, state_dim, action_bound, learning_rate, t_replace_iter, batch_size=1, ctx=mx.cpu()):
        self.a_dim = action_dim
        self.n_features = state_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.batch_size = batch_size
        self.ctx = ctx
        self.modA_target = self.create_actor_network()
        self.modA_eval = self.create_actor_network(isTrain=True)
        self.copyTargetANetwork()


np.random.seed(1)
mx.random.seed(1)

MAX_EPISODES = 70
MAX_EP_STEPS = 400
LR_A = 0.01
LR_C = 0.01
GAMMA = 0.9
TAU = 0.01
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = 7000
BATCH_SIZE = 32
CTX = mx.cpu()

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)


if __name__ == "__main__":
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.shape[0]
    ACTION_BOUND = env.action_space.high

    actor = Actor(action_dim=ACTION_DIM, state_dim=STATE_DIM, action_bound=ACTION_BOUND, learning_rate=LR_A,
                  t_replace_iter=REPLACE_ITER_A, batch_size=BATCH_SIZE, ctx=CTX)

    critic = Critic(n_action=ACTION_DIM, n_features=STATE_DIM, learning_rate=LR_C, t_replace_iter=REPLACE_ITER_C,
                    batchsize=BATCH_SIZE, ctx=CTX)

    M = Memory(capacity=MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)