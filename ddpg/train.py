from actor_network import *
from critic_network import *
from replay_buffer import Memory

M = 1000 # outer loop iteration

if __name__ == "__main__":
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
