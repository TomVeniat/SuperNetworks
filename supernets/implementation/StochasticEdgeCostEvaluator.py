import logging

import torch

from ..implementation.EdgeCostEvaluator import EdgeCostEvaluator

logger = logging.getLogger(__name__)


class StochasticEdgeCostEvaluator(EdgeCostEvaluator):

    def init_costs(self, model):
        self.costs = torch.Tensor()

        for node in model.traversal_order:
            if node not in model.stochastic_node_ids:
                self.costs = torch.cat([self.costs, torch.zeros(1)])
            else:
                self.costs = torch.cat([self.costs, torch.ones(model.graph.node[node].get('n_ops', 1))])

        print(self.costs)

