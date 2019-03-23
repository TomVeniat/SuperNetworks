from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from supernets.interface.Observable import Observable
from supernets.networks.SuperNetwork import SuperNetwork


class StochasticSuperNetwork(Observable, SuperNetwork):
    def __init__(self, deter_eval, *args, **kwargs):
        super(StochasticSuperNetwork, self).__init__(*args, **kwargs)

        self.stochastic_node_ids = defaultdict()
        self.stochastic_node_ids.default_factory = self.stochastic_node_ids.__len__

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()

        self.deter_eval = deter_eval
        self.mean_entropy = None
        self.all_same = False

        self.distrib_entropies = None
        self.log_probas = None
        self.samplings = None
        self.probas = None

        self.node_hook = self.apply_sampling
        self.register_forward_pre_hook(self._sample_archs)
        self.register_forward_pre_hook(self._fire_all_samplings)

    def apply_sampling(self, node, output):
        if node not in self.stochastic_node_ids:
            # Current node isn't stochastic
            return output

        sampling = self.get_sampling(node, output)
        return output * sampling

    def get_sampling(self, node_name, out):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out size, with all dimensions equals to one except the first one (batch)
        """
        sampling = self.samplings[:, self.stochastic_node_ids[node_name]]

        #make The sampling broadcastable with the output
        sampling_dim = [1] * out.dim()
        sampling_dim[0] = out.size(0)
        sampling = sampling.view(sampling_dim)

        return sampling

    def set_probas(self, probas, all_same=False):
        """
        :param probas: B_size*N_nodes Tensor containing the probability of each arch being sampled in the nex forward.
        :param all_same: if True, the same sampling will be used for the whole batch in the next forward.
        :return:
        """
        if probas.dim() != 2 or all_same and probas.size(0) != 1:
            raise ValueError('probas params has wrong dimension: {} (all_same={})'.format(probas.size(), all_same))

        if probas.size(-1) != self.n_stoch_nodes:
            raise ValueError('Should have exactly as many probas as the number of stochastic nodes({}), got {} instead.'
                             .format(self.n_stoch_nodes, probas.size(-1)))

        self.all_same = all_same
        self.probas = probas

    def _sample_archs(self, _, input):
        """
        Hook called by pytorch before each forward
        :param _: Current module
        :param input: Input given to the module's forward
        :return:
        """
        # Pytorch hook gives the input as a tuple
        if isinstance(input, tuple):
            assert len(input) == 1
            input = input[0]

        # Case of multiple input nodes where input is a list:
        if isinstance(input, list):
            input = input[0]

        batch_size = input.size(0)

        # Check the compatibility with the batch_size
        if self.probas.size(0) != batch_size:
            if self.probas.size(0) != 1:
                raise ValueError('Sampling probabilities dimensions {} doesn\'t match with batch size {}.'
                                 .format(self.probas.size(), batch_size))
            if not self.all_same:
                self.probas = self.probas.expand(batch_size, -1)

        distrib = torch.distributions.Bernoulli(self.probas)
        if not self.training and self.deter_eval:
            self.samplings = (self.probas > 0.5).float()
        else:
            self.samplings = distrib.sample()

        if self.all_same:
            self.samplings = self.samplings.expand(batch_size, -1)

        self.distrib_entropies.append(distrib.entropy())
        self.log_probas.append(distrib.log_prob(self.samplings))

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
            # todo: Implemented this way to work with old implementation, can be done in a better way now.
            if node_name in self.stochastic_node_ids:
                sampling = self.samplings[:, self.stochastic_node_ids[node_name]]
                self.fire(type='sampling', node=node_name, value=sampling)
            else:
                batch_size = input.size(0)
                self.fire(type='sampling', node=node_name, value=torch.ones(batch_size))

    @property
    def n_nodes(self):
        return self.net.number_of_nodes()

    @property
    def n_stoch_nodes(self):
        return len(self.stochastic_node_ids)

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

    @property
    def ordered_node_names(self):
        return [str(elt[0]) for elt in sorted(self.stochastic_node_ids.items(), key=lambda x: x[1])]

    def update_probas_and_entropies(self):
        raise NotImplementedError('Not available in current version')
        if self.nodes_param is None:
            self._init_nodes_param()
        self.probas = {}
        self.entropies = {}
        self.mean_entropy = .0
        for node, props in self.graph.nodes.items():
            param = self.sampling_parameters[props['sampling_param']]
            p = param.sigmoid().item()
            self.probas[node] = p
            if p in [0, 1]:
                e = 0
            else:
                e = -(p * np.log2(p)) - ((1 - p) * np.log2(1 - p))
            self.entropies[node] = e
            self.mean_entropy += e
        self.mean_entropy /= self.graph.number_of_nodes()

    # def _init_nodes_param(self):
    #     self.nodes_param = {}
    #     for node, props in self.graph.node.items():
    #         if 'sampling_param' in props and props['sampling_param'] is not None:
    #             self.nodes_param[node] = props['sampling_param']

    def register_stochastic_node(self, node):
        if node in self.stochastic_node_ids:
            raise ValueError('Node {} already registered'.format(node))
        return self.stochastic_node_ids[node]

    def __str__(self):
        model_descr = 'Model:{}\n' \
                      '\t{} nodes\n' \
                      '\t\t{} stochastic\n' \
                      '\t{} blocks\n' \
                      '\t{} parametrized layers\n' \
                      '\t{} computation steps\n' \
                      '\t{} parameters\n' \
                      '\t\t{} trainable\n' \
                      '\t\t{} meta-params\n'
        return model_descr.format(type(self).__name__, self.n_nodes, self.n_stoch_nodes, len(self.blocks), self.n_layers,
                                  self.n_comp_steps, sum(i.numel() for i in self.parameters()),
                                  sum(i.numel() for i in self.parameters() if i.requires_grad), self.n_stoch_nodes) + '\n' + super(StochasticSuperNetwork, self).__str__()
