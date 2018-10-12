import matplotlib

# matplotlib.use('SVG')
import matplotlib.pyplot as plt

from .Drawer import Drawer


class ThreeDimNeuralFabricDrawer(Drawer):
    NODE_SIZE = 50
    EDGE_WIDTH = 5.0

    @classmethod
    def get_draw_pos(cls, cur_node):
        if cur_node is 'In':
            return -1, -1
        elif cur_node is 'Out':
            return 5, 5

        if isinstance(cur_node[0], tuple):
            cur_node = cur_node[1]
        return cur_node[0], cur_node[1]

    @classmethod
    def get_draw_pos_3d(cls, cur_node):
        if cur_node is 'In':
            return -1, -1, -1
        elif cur_node is 'Out':
            return 7, 7, 5

        if isinstance(cur_node[0], tuple):
            cur_node = cur_node[1]
        return cur_node

    def draw(self, graph, param_list=None, weight_attr='sampling_param', default_w=None, vis_opts=None, vis_win=None,
             vis_env=None, colormap=plt.cm.YlGnBu):
        if param_list is not None:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                return param_list[width_node['sampling_param']].sigmoid().item()
        elif type(weight_attr) is str:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                return width_node[weight_attr] if weight_attr in width_node else default_w
        elif type(weight_attr) is float:
            def weighter(_):
                return weight_attr
        else:
            def weighter(e):
                return weight_attr[graph.get_edge_data(*e)['width_node']]

        def positioner(n):
            if 'pos' not in graph.node[n]:
                return None
            return graph.node[n]['pos']

        img = self._draw_net(graph, positioner=positioner, weighter=weighter, colormap=colormap)

        env = vis_env if vis_env is not None else self.env
        win = vis_win if vis_win is not None else self.win
        if 'width' not in vis_opts:
            vis_opts['width'] = 600
        if 'height' not in vis_opts:
            vis_opts['height'] = 450
        self.win = self.vis.svg(svgstr=img, win=win, opts=vis_opts, env=env)
