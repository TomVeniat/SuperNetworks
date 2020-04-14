import collections

import networkx as nx
import torch
from networkx import NetworkXError, NetworkXUnfeasible
from torch import nn


class SuperNetwork(nn.Module):
    def __init__(self):
        super(SuperNetwork, self).__init__()

        self.net = None
        self.traversal_order = None
        self.in_nodes = None
        self.out_nodes = None
        self.observer = None

        self.output_index = None

    def set_graph(self, network, in_nodes, out_nodes):
        self.net = network
        try:
            self.traversal_order = list(nx.topological_sort(self.net))
        except NetworkXError:
            raise ValueError('The Super Network must be directed graph.')
        except NetworkXUnfeasible:
            raise ValueError('The Super Network must be a DAG.')

        if not (self.traversal_order[0] in in_nodes and self.traversal_order[-1] in out_nodes):
            raise ValueError('Seems like the given graph is broken '
                             '(First node {} not in in_nodes or last node {} not in out_nodes).'.format(self.traversal_order[0],
                                                                                                        self.traversal_order[-1]))

        if not (isinstance(in_nodes, tuple) or isinstance(in_nodes, list)):
            raise ValueError('Input nodes should be list or tuple.')

        if not (isinstance(out_nodes, tuple) or isinstance(out_nodes, list)):
            raise ValueError('Output nodes should be list or tuple.')

        self.in_nodes = in_nodes
        self.set_out_nodes(out_nodes)

    def set_out_nodes(self, out_nodes):
        self.output_index = dict((node, i) for i, node in enumerate(out_nodes))
        self.out_nodes = out_nodes

    def forward(self, inputs):
        # First, set the input for each input node
        if torch.is_tensor(inputs):
            inputs = [inputs]

        assert len(inputs) == len(self.in_nodes), 'Inputs should have as many elements as the number of input nodes. ' \
                            'Got {} elements for {} input nodes'.format(len(inputs), len(self.in_nodes))
        for node, input in zip(self.in_nodes, inputs):
            if not isinstance(input, collections.Mapping):
                input = dict(fed_input=input)
            self.net.node[node]['input'] = input

        # Traverse the graph, saving the output of each out node
        outputs = [None] * len(self.out_nodes)
        for node in self.traversal_order:
            cur_node = self.net.node[node]
            input = self.format_input(cur_node.pop('input'), cur_node.get('squeeze_dict', True))

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            # print(f"{node}({type(cur_node['module'])}) with in size: {in_size}")
            if self.node_pre_hook:
                feed_in = self.node_pre_hook(node, input)
            else:
                feed_in = input

            out = cur_node['module'](feed_in)
            if self.node_post_hook:
                out = self.node_post_hook(node, input, out)

            if node in self.out_nodes:
                outputs[self.output_index[node]] = out

            for succ in self.net.successors(node):
                if 'input' not in self.net.node[succ]:
                    self.net.node[succ]['input'] = {}
                self.net.node[succ]['input'][node] = out

        return outputs

    @property
    def input_size(self):
        if not hasattr(self, '_input_size'):
            raise RuntimeError('SuperNetworks should have an `_input_size` attribute.')
        return self._input_size

    def get_ops_per_node(self):
        return dict((node_name, node_data['module'].get_flop_cost())
                    for node_name, node_data in dict(self.graph.nodes(True)).items())

    def get_params_per_node(self):
        return dict((node_name, sum(param.numel() for param in node_data['module'].parameters()))
                    for node_name, node_data in dict(self.graph.nodes(True)).items())

    @staticmethod
    def format_input(input, squeeze_dict=True):
        assert isinstance(input, dict), 'All inputs should now be of dict type, got {}.'.format(type(input))
        if len(input) == 1 and squeeze_dict:
            input = list(input.values())[0]
        return input
