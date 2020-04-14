from collections import OrderedDict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from supernets.interface.Observable import Observable
from supernets.networks.SuperNetwork import SuperNetwork


class StochasticSuperNetwork(Observable, SuperNetwork):
    def __init__(self, sample_type=None, *args, **kwargs):
        super(StochasticSuperNetwork, self).__init__(*args, **kwargs)

        self.stochastic_node_ids = OrderedDict()
        self.stochastic_node_next_id = 0

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()

        self.sample_type = sample_type
        self.mean_entropy = None

        self.samplings = None

        self.node_pre_hook = self.prepare_input
        # self.node_pre_hook = None

        self.node_post_hook = self.apply_sampling_mask
        # self.node_post_hook = self.apply_sampling_mult
        self.register_forward_pre_hook(self._fire_all_samplings)

    def prepare_input(self, node, input):
        if node not in self.stochastic_node_ids:
            # Current node isn't stochastic
            return input
        sampling = self.samplings[:, self.stochastic_node_ids[node]]
        return input[sampling.bool()]

    def apply_sampling_mask(self, node, input, output):
        if node not in self.stochastic_node_ids:
            # Current node isn't stochastic
            return output

        sampling = self.samplings[:, self.stochastic_node_ids[node]]
        out_size = list(output.size())
        out_size[0] = input.size(0)
        res = torch.zeros(out_size, device=output.device)
        res[sampling.bool()] = output
        return res


    def apply_sampling_mult(self, node, input, output):
        if node not in self.stochastic_node_ids:
            # Current node isn't stochastic
            return output

        sampling = self.get_sampling(node, output)
        return output * sampling

    def get_sampling(self, node_name, out):
        """
        Get a batch of sampling for the given node broadcastable with the
        given output.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out size, with all dimensions
        equal to one except the first one (batch)
        """
        sampling = self.samplings[:, self.stochastic_node_ids[node_name]]

        # assert sampling.dim() == 1 or sampling.size() == out.size()
        if sampling.dim() == 1:
            #  Make The sampling broadcastable with the output
            sampling_dim = [1] * out.dim()
            sampling_dim[0] = out.size(0)
            sampling = sampling.view(sampling_dim)
        elif sampling.size() != out.size():
            assert sampling.size() == out.size()[:2]
            sampling_dim = [1] * out.dim()
            sampling_dim[:2] = out.size()[:2]
            sampling = sampling.view(sampling_dim)
        return sampling

    def _fire_all_samplings(self, _, input):
        """
        Method used to notify the observers of the sampling
        """
        self.fire(type='new_iteration')

        # Pytorch hook gives the input as a tuple
        assert isinstance(input, tuple) and len(input) == 1
        input = input[0]

        # Case of multiple input nodes where input is a list:
        if isinstance(input, list):
            input = input[0]

        for node_name in self.traversal_order:
            # todo: Implemented this way to work with old implementation,
            #  can be done in a better way now.
            if node_name in self.stochastic_node_ids:
                node_id = self.stochastic_node_ids[node_name]
                sampling = self.samplings[:, node_id]
                self.fire(type='sampling', node=node_name, value=sampling)
            else:
                batch_size = input.size(0)
                self.fire(type='sampling', node=node_name,
                          value=torch.ones(batch_size))

    def start_new_sequence(self):
        self.samplings = None
        self.fire(type='new_sequence')

    @property
    def n_nodes(self):
        return self.net.number_of_nodes()

    @property
    def n_stoch_nodes(self):
        return len(self.stochastic_node_ids)

    @property
    def n_sampling_params(self):
        return self.stochastic_node_next_id

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

    @property
    def ordered_node_names(self):
        return list(map(str, self.stochastic_node_ids.keys()))

    @property
    def all_params_name(self):
        res = []
        for nodes, idx in self.stochastic_node_ids.items():
            if isinstance(idx, int):
                res.append(str(nodes))
            else:
                assert len(idx) == len(nodes)
                res.extend(str(node) for node in nodes)

        return res

    def get_names_from_probas(self, probas):
        res = {}
        for node_name, idx in self.stochastic_node_ids.items():
            cur_probs = probas[idx]
            if isinstance(cur_probs, np.ndarray):
                assert isinstance(node_name, tuple)\
                       and len(node_name) == len(cur_probs)
                for n, p in zip(node_name, cur_probs):
                    res[n] = p
            else:
                res[node_name] = cur_probs
        return res

    def register_stochastic_node(self, node, n_ops=1, type=None):
        if self.sample_type is not None and self.sample_type != type:
            return
        if node in self.stochastic_node_ids:
            raise ValueError('Node {} already registered'.format(node))
        if n_ops == 1:
            # id = next(self.stochastic_node_id_generator)
            self.stochastic_node_ids[node] = self.stochastic_node_next_id
            self.stochastic_node_next_id += 1
            return self.stochastic_node_ids[node]
        else:
            assert isinstance(node, tuple) and len(node) == n_ops
            ids = [self.stochastic_node_next_id + i for i in range(n_ops)]
            self.stochastic_node_ids[node] = ids
            self.stochastic_node_next_id += n_ops
            return ids

    def __str__(self):
        model_descr = 'Model:{}\n' \
                      '\t{} nodes\n' \
                      '\t\t{} stochastic\n' \
                      '\t{} blocks\n' \
                      '\t{} parametrized layers\n' \
                      '\t{} computation steps\n' \
                      '\t{} parameters\n' \
                      '\t\t{} trainable\n' \
                      '\t\t{} meta-params\n'\
                      '\t\t{}'
        return model_descr.format(type(self).__name__, self.n_nodes,
                                  self.n_stoch_nodes, len(self.blocks),
                                  self.n_layers, self.n_comp_steps,
                                  sum(i.numel() for i in self.parameters()),
                                  sum(i.numel() for i in self.parameters()
                                      if i.requires_grad),
                                  self.n_stoch_nodes,
                                  super(StochasticSuperNetwork, self))
