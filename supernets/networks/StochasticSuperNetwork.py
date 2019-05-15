from collections import OrderedDict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from supernets.interface.Observable import Observable
from supernets.networks.SuperNetwork import SuperNetwork


class StochasticSuperNetwork(Observable, SuperNetwork):
    def __init__(self, deter_eval, sample_type=None, *args, **kwargs):
        super(StochasticSuperNetwork, self).__init__(*args, **kwargs)

        self.stochastic_node_ids = OrderedDict()
        self.stochastic_node_next_id = 0

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()

        self.deter_eval = deter_eval
        self.sample_type = sample_type
        self.mean_entropy = None
        self.all_same = False

        self.distrib_entropies = None
        self._seq_probas = None
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

    def set_probas(self, probas, all_same=False):
        """
        :param probas: B_size*N_nodes Tensor containing the probability of each arch being sampled in the nex forward.
        :param all_same: if True, the same sampling will be used for the whole batch in the next forward.
        :return:
        """
        if probas.dim() != 2 or all_same and probas.size(0) != 1:
            raise ValueError('probas params has wrong dimension: {} (all_same={})'.format(probas.size(), all_same))

        if probas.size(-1) != self.n_sampling_params:
            raise ValueError('Should have exactly as many probas as the number of stochastic nodes({}), got {} instead.'
                             .format(self.n_sampling_params, probas.size(-1)))

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

        self._seq_probas.append(self.probas)
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

    def start_new_sequence(self):
        self.log_probas = []
        self.distrib_entropies = []
        self._seq_probas = []
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

    @property
    def last_sequence_probas(self):
        """
        :return: The probabilities of each arch for the last sequence in format (seq_len*batch_size*n_sampling_params)
        """
        return torch.stack(self._seq_probas)

    def get_names_from_probas(self, probas):
        res = {}
        for node_name, idx in self.stochastic_node_ids.items():
            cur_probs = probas[idx]
            if isinstance(cur_probs, np.ndarray):
                assert isinstance(node_name, tuple) and len(node_name) == len(cur_probs)
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
                      '\t\t{} meta-params\n'
        return model_descr.format(type(self).__name__, self.n_nodes, self.n_stoch_nodes, len(self.blocks),
                                  self.n_layers,
                                  self.n_comp_steps, sum(i.numel() for i in self.parameters()),
                                  sum(i.numel() for i in self.parameters() if i.requires_grad),
                                  self.n_stoch_nodes) + '\n' + super(StochasticSuperNetwork, self).__str__()
