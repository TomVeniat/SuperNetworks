import logging

import torch

from supernets.implementation.EdgeCostEvaluator import EdgeCostEvaluator
from supernets.interface.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class ComputationCostEvaluator(EdgeCostEvaluator):
    def init_costs(self, model):
        device = next(model.parameters()).device
        with torch.no_grad():

            graph = model.net
            self.costs = torch.Tensor(graph.number_of_nodes())
            inputs = [torch.ones(1, *var_dim).to(device) for var_dim in model.input_size]

            assert len(inputs) == len(model.in_nodes), 'Inputs should have as many elements as the number of input nodes. ' \
                                'Got {} elements for {} input nodes'.format(len(inputs), len(model.in_nodes))
            for node, input in zip(model.in_nodes, inputs):
                graph.node[node]['input'] = [input]

            # Traverse the graph
            for node in model.traversal_order:
                cur_node = graph.node[node]
                input_var = model.format_input(cur_node.pop('input'))

                if len(input) == 0:
                    raise RuntimeError('Node {} has no inputs'.format(node))

                out = cur_node['module'](input_var)

                if isinstance(cur_node['module'], NetworkBlock):
                    cost = cur_node['module'].get_flop_cost(input_var)
                else:
                    logger.warning('Node {} isn\'t a Network block'.format(node))

                self.costs[self.node_index[node]] = cost

                for succ in graph.successors(node):
                    if 'input' not in graph.node[succ]:
                        graph.node[succ]['input'] = []
                    graph.node[succ]['input'].append(out)
