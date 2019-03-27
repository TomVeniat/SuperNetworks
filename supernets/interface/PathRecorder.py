import abc
import copy

import torch


class PathRecorder(object, metaclass=abc.ABCMeta):
    def __init__(self, model):
        self.graph = model.graph
        # todo: check default behavior
        self.default_outs = model.out_nodes

        self.n_nodes = self.graph.number_of_nodes()

        # create node-to-index and index-to-node mapping
        self.op_index = {}
        self.node_index = {}
        self.rev_node_index = [None] * self.n_nodes
        j = 0
        for i, node in enumerate(model.traversal_order):
            n_ops = model.graph.node[node].get('n_ops', 1)

            self.op_index[node] = list(range(j, j+n_ops)) if n_ops > 1 else j
            j += n_ops

            self.node_index[node] = i
            self.rev_node_index[i] = node

        self.n_total_ops = j

        self.global_sampling = None
        self.n_samplings = 0

        self.active_nodes_seq = None
        self.node_samplings = None
        self.all_samplings = None

        model.subscribe(self.new_event)

    def new_event(self, e):
        if e.type is 'new_sequence':
            self.new_sequence()
        elif e.type is 'new_iteration':
            self.new_iteration()
        elif e.type is 'sampling':
            self.new_sampling(e.node, e.value)

    def new_sequence(self):
        self.active_nodes_seq = []
        self.node_samplings = []
        self.all_samplings = []

    def new_iteration(self):
        # Todo: adapt the global sampling idea in the sequence settings, maybe with a different global sampling for each class.
        # if self.default_out is not None and self.active is not None:
        #     pruned = self.get_pruned_architecture(self.default_out)
        #     self.update_global_sampling(pruned)

        self.active_nodes_seq.append(torch.Tensor())
        self.node_samplings.append(torch.Tensor())
        self.all_samplings.append(torch.Tensor())

    def new_sampling(self, node_name, sampling):
        """

        :param node_name: Node considered
        :param sampling: Vector of size (batch_size), corresponding to the sampling of the given node
        :return:
        """
        assert torch.is_tensor(sampling) and sampling.dim() in ([1, 2])
        sampling = sampling.cpu()
        batch_size = sampling.size(0)

        # Get current elements in the sequence
        self.active = self.active_nodes_seq[-1]
        self.node_sampling = self.node_samplings[-1]
        self.all_sampling = self.all_samplings[-1]

        if self.active.numel() == 0 and self.node_sampling.numel() == 0:
            # This is the first step of the sequence
            self.active.resize_(self.n_nodes, self.n_nodes, batch_size).zero_()
            self.node_sampling.resize_(self.n_nodes, batch_size).zero_()
            self.all_sampling.resize_(self.n_total_ops, batch_size).zero_()

        self.all_sampling[self.op_index[node_name]] = sampling.t() if sampling.dim() == 2 else sampling

        if sampling.dim() > 1:
            cur_node_sampling = sampling.view(sampling.size(0), -1).byte().any(dim=1).float()
        else:
            cur_node_sampling = sampling

        assert cur_node_sampling.dim() == 1

        node_ind = self.node_index[node_name]
        self.node_sampling[node_ind] = cur_node_sampling

        # incoming is a (n_nodes*batch_size) matrix.
        # We will set incoming_{i,j} = 1 if the node i contributes to current node computation in batch element j
        incoming = self.active[node_ind]
        assert incoming.sum() == 0

        predecessors = list(self.graph.predecessors(node_name))

        if len(predecessors) == 0:
            # Considered node is an input node
            incoming[node_ind] += cur_node_sampling

        for prev in predecessors:
            # If the predecessor itself is active (has connection with the input),
            # it could contribute to the computation of the considered node.
            incoming += self.active[self.node_index[prev]]

        assert incoming.size() == torch.Size((self.n_nodes, batch_size))

        # has_inputs[i] > 0 if there is at least one predecessor node which is active in batch element i
        has_inputs = incoming.max(0)[0]

        # the current node has outputs if it has at least on predecessor node active AND it is sampled
        has_outputs = ((has_inputs * cur_node_sampling) != 0).float()

        backup = copy.deepcopy(incoming)

        ###
        # other_method = copy.deepcopy(incoming)
        # other_method[node_ind] += has_outputs
        # other_method = (other_method != 0).float()
        ###
        incoming[node_ind] += cur_node_sampling

        sampling_mask = has_outputs.expand(self.n_nodes, batch_size)
        incoming *= sampling_mask

        res = (incoming != 0).float()
        self.active[node_ind] = res

        # eq = res.equal(other_method)

    def update_global_sampling(self, used_nodes):
        self.n_samplings += 1
        mean_sampling = used_nodes.mean(1).squeeze()

        if self.global_sampling is None:
            self.global_sampling = mean_sampling
        else:
            self.global_sampling += (1 / self.n_samplings) * (mean_sampling - self.global_sampling)

    def get_used_nodes(self, architectures):
        """
        Translates each architecture from a vector representation to a list of the nodes it contains
        :param architectures: a batch of architectures in format seq_len * batch_size * n_nodes
        :return: a list of batch_size elements, each elements being a list of nodes.
        """

        res = []
        for step_archs in architectures:
            step = []
            for arch in step_archs:
                nodes = [self.rev_node_index[idx] for idx, used in enumerate(arch) if used == 1]
                step.append(nodes)
            res.append(step)
        return res

    def get_graph_paths(self, out_node):
        sampled, pruned = self.get_architectures(out_node)

        real_paths = []
        for i in range(pruned.size(1)):  # for each batch element
            path = [self.rev_node_index[ind] for ind, used in enumerate(pruned[:, i]) if used == 1]
            real_paths.append(path)

        res = self.get_used_nodes(pruned.t())

        assert real_paths == res

        sampling_paths = []
        for i in range(sampled.size(1)):  # for each batch element
            path = dict((self.rev_node_index[ind], elt) for ind, elt in enumerate(sampled[:, i]))
            sampling_paths.append(path)

        self.update_global_sampling(pruned)

        return real_paths, sampling_paths

    def get_posterior_weights(self):
        return dict((self.rev_node_index[ind], elt) for ind, elt in enumerate(self.global_sampling))

    def get_consistence(self, node):
        """
        Get an indicator of consistence for each sampled architecture up to the given node in last batch.

        :param node: The target node.
        :return: a ByteTensor containing one(zero) only if the architecture is consistent and the param is True(False).
        """
        return self.active[self.node_index[node]].sum(0) != 0

    def is_consistent(self, model):
        model.eval()
        with torch.no_grad():
            input = torch.ones(1, *model.input_size)
            model(input)
        consistence = self.get_consistence(model.out_node)
        return consistence.sum() != 0

    def get_architectures(self, out_nodes=None):
        if out_nodes is None:
            out_nodes = [self.default_out]
        return self.get_sampled_architectures(), self.get_pruned_architecture(out_nodes)

    def get_sampled_architectures(self):
        """

        :return: the real samplings in size (seq_len*n_nodes*batch_size)
        """
        return torch.stack(self.node_samplings)

    def get_pruned_architecture(self, out_nodes):
        """
        :return: the pruned samplings in size (seq_len*n_nodes*batch_size)
        """
        seq_len = len(self.active_nodes_seq)
        n_nodes = self.n_nodes
        batch_size = self.active_nodes_seq[0].size(-1)
        res = torch.zeros((seq_len, n_nodes, batch_size))
        for out_node in out_nodes:
            out_index = self.node_index[out_node]
            res += torch.stack([active[out_index] for active in self.active_nodes_seq])

        return (res != 0).float()

    def get_pruned_operations(self, out_nodes):
        pruned_arch = self.get_pruned_architecture(out_nodes)
        pruned_ops = torch.zeros((pruned_arch.size(0), self.n_total_ops, pruned_arch.size(-1)))
        all_samplings = torch.stack(self.all_samplings)
        for i, is_sampled in enumerate(pruned_arch.split(1, 1)):
            ops_idx = self.op_index[self.rev_node_index[i]]
            sampled_node_ops = all_samplings[:, ops_idx, :]
            pruned_ops[:, ops_idx, :] = sampled_node_ops * is_sampled.squeeze(1) if isinstance(ops_idx, int) else is_sampled

        return pruned_ops



    def get_state(self):
        return {'node_index': self.node_index,
                'rev_node_index': self.rev_node_index,
                'global_sampling': self.global_sampling,
                'n_samplings': self.n_samplings}

    def load_state(self, state):
        for key, val in state.items():
            if not hasattr(self, key):
                raise AttributeError('Given state has unknown attribute {}.')
            setattr(self, key, val)
