from mxnet import nd
from mxnet.gluon import nn
import mxnet as mx

import numpy as np
import copy


class CriticNetwork:
    def __init__(self, output_dim, ctx=mx.cpu()):
        """

        :type output_dim: int
        :type ctx: mx.Context
        """
        self.ctx = ctx
        self.output_dim = output_dim
        self.net = self.get_net()

    # Test architecture
    def get_net(self):
        net = nn.Sequential()
        with net.name_scope():
            net.add(nn.Dense(self.output_dim))
        net.initialize(init=mx.init.Xavier(), ctx=self.ctx)
        return net


class TargetNetQ:
    def __init__(self, critic, ctx=mx.cpu()):
        # Initialize target network with the same structure and weights as actor network
        self.net = copy.copy(critic.net)
        self.ctx = ctx


if __name__ == "__main__":
    np.random.seed(0)
    mx.random.seed(0)

    cri_net = CriticNetwork(output_dim=10)
    tar_net = TargetNetQ(cri_net)
    print(cri_net.net(nd.array([1])))
    print(tar_net.net(nd.array([1])))
    print(cri_net.net is tar_net.net)
