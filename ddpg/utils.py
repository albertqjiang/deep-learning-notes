from actor_network import *


def update_net_weights(net_weights, new_weights, add_weights=False):
    """

    Update weights of a network, self explanatory

    :param net_weights: old net parameters, can be obtained by net.collect_params()
    :param new_weights: new net parameters, list of new weights, same shape as the net parameters,
                        but in a list of ndarrays
    :param add_weights: default False, whether to add new weights to old weights
    :return: None
    """
    def update(a, b, add=False):
        if add is True:
            return a + b
        else:
            return b

    net_weight_keys = net_weights.keys()
    for net_weight_key, new_weight in zip(net_weight_keys, new_weights):
        net_weights[net_weight_key].set_data(update(net_weights[net_weight_key], new_weight, add=add_weights))


if __name__ == "__main__":
    np.random.seed(0)
    mx.random.seed(0)

    actor = ActorNetwork(1)
    state = nd.array([[1,2,3]])
    action = actor.net(state)

    set_weights = [nd.array([[1,2,3]]), nd.array([1])]

    update_net_weights(actor.net.collect_params(), set_weights)
    print(list(actor.net.collect_params().values())[0].data())
    print(list(actor.net.collect_params().values())[1].data())