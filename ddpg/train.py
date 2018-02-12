from mxnet import nd
import mxnet as mx
from mxnet import autograd
from mxnet import gluon

import gym

import numpy as np
import time

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
batch_size = 32  # batch size
lr = 0.0001  # learning rate

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

    # Total loss for critic
    total_critic_loss = 0
    total_transition_trained_on = 0

    # Outer iteration
    for m in range(M):

        # Receive initial observation
        s = env.reset()
        explore_variance = 2  # initial exploration variance

        s = nd.array(s).reshape((1, -1))
        # print(s)

        inner_time = time.time()
        # Inner iteration
        for t in range(T):

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
                combined_s_a_target = nd.concatenate([b_s_, a_target], axis=1)

                q_target = target_q.net(combined_s_a_target)

                y = r + q_target * gamma
                # print(y)

                # Update the critic network and calculate policy gradient
                # Adam critic_trainer by default
                critic_trainer = gluon.Trainer(critic.net.collect_params(), 'adam', {'learning_rate': lr})

                # Define critic loss
                square_loss = gluon.loss.L2Loss()

                with autograd.record():
                    combined_sa = nd.concatenate([b_s, b_a], axis=1)
                    critic_output = critic.net(combined_sa)
                    loss = square_loss(critic_output, y)

                loss.backward()
                critic_trainer.step(batch_size)

                # Calculate policy gradient

                params_grads = []
                a_grads = []

                for state_i in b_s:
                    state_i = state_i.reshape((1, -1))
                    with autograd.record():
                        a_predict = actor.net(state_i)
                    # 1. Calculate gradient of the critic function w.r.t. the predicted action
                    combined_sa = nd.concatenate([state_i, a_predict], axis=1)

                    # 2. Calculate gradient of the action function w.r.t. its parameters
                    a_predict.backward()

                    params = actor.net.collect_params()
                    param_values = params.values()
                    param_keys = params.keys()
                    if isinstance(params, (dict, gluon.ParameterDict)):
                        params = list(params.values())

                    params_grads.append([param.data().grad for param in params])

                    # 1. Back to 1
                    combined_sa.attach_grad()

                    with autograd.record():
                        critic_output = critic.net(combined_sa)

                    # Take the gradient of q with respect to a
                    critic_output.backward()

                    a_grad = combined_sa.grad[:, state_dim:]
                    a_grads.append(a_grad)

                policy_gradients = []

                for i in range(batch_size):
                    policy_gradient = [a_grads[i]*grad for grad in params_grads[i]]
                    policy_gradients.append(policy_gradient)
                    # Todo: multiply params and action gradients together to get policy gradients

                policy_grad_ave = []
                for grad in policy_gradients:
                    if len(policy_grad_ave) == 0:
                        policy_grad_ave = grad
                    else:
                        # print("Original gradient is")
                        # print(policy_grad_ave)
                        # print("Added gradient is")
                        # print(grad)
                        for grad_layer, added_grad_layer in zip(policy_grad_ave, grad):
                            grad_layer += added_grad_layer
                    # print(policy_grad_ave)
                policy_grad_ave = [item/batch_size for item in policy_grad_ave]
                # print(policy_grad_ave)

                actor_params_dict = actor.net.collect_params()

                total_critic_loss += nd.sum(loss).asscalar()
                total_transition_trained_on += batch_size

        print(time.time() - inner_time)


        # if total_transition_trained_on != 0:
        #     print(total_critic_loss / total_transition_trained_on)