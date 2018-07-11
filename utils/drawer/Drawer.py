import io
import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import visdom
from plotly.graph_objs import *

logger = logging.getLogger(__name__)


class Drawer(object):
    __default_env = None

    def __init__(self, env=None):
        with open('resources/visdom.json') as file:
            draw_params = json.load(file)
        server = draw_params['url']
        port = draw_params['port']
        self.vis = visdom.Visdom(server=server, port=port)
        self.win = None
        self.env = Drawer.__default_env if env is None else env

        logger.info('Init drawer to {}:{}/env/{}'.format(server, port, self.env))

    @staticmethod
    def set_default_env(env):
        Drawer.__default_env = env

    def set_env(self, env):
        self.env = env
        return self

    def _draw_net(self, graph, filename=None, show_fig=False, normalize=False,
                  nodefilter=None, edgefilter=None, positioner=None, weighter=None, colormap=None):
        plt.close()

        nodes = graph.nodes()
        if nodefilter:
            nodes = [node for node in nodes if nodefilter(node)]

        edges = graph.edges()
        if edgefilter:
            edges = [edge for edge in edges if edgefilter(edge)]

        if positioner is None:
            pos = nx.spring_layout(graph)
        else:
            pos = dict((n, positioner(n)) for n in graph.nodes())

        weights = 1.0
        if weighter is not None:
            weights = [weighter(e) for e in edges]

        weights = np.array(weights)
        w_min = weights.min()
        w_max = weights.max()
        if normalize and w_min != w_max:
            weights = np.log(weights + 1e-5)
            weights = (weights - w_min) * 1.0 / (w_max - w_min) + 2

        if w_min != w_max:
            diff = w_max - w_min
            w_min -= 0.01 * diff
            w_max += 0.01 * diff
        else:
            w_min -= .01
            w_max += .01

        nx.draw_networkx_nodes(graph, nodelist=nodes, pos=pos, node_size=self.NODE_SIZE, node_color='red')
        res = nx.draw_networkx_edges(graph, edgelist=edges, pos=pos, width=self.EDGE_WIDTH, arrows=False,
                                     edge_color=weights,
                                     edge_cmap=colormap, edge_vmin=w_min, edge_vmax=w_max)
        plt.colorbar(res)

        if show_fig:
            plt.show()
        if filename is not None:
            plt.savefig(filename, format='svg')

        img_data = io.StringIO()
        plt.savefig(img_data, format='svg')

        return img_data.getvalue()

    def scatter(self, x, y, opts, vis_win):
        points = []
        labels = []
        legend = []

        for i, (name, abs) in enumerate(x.items()):
            ord = 0 if y is None else y[name]

            points.append((abs, ord))
            labels.append(i + 1)
            legend.append(name)

        points = np.asarray(points)

        vis_opts = dict(
            legend=legend,
            markersize=5,
        )
        vis_opts.update(opts)
        self.vis.scatter(points, labels, win=vis_win, opts=vis_opts, env=self.env)

    def plotly_graph(self, g, weights=None, vis_opts=None, vis_win=None):
        edges = _get_edges_plotly(g, weights)
        nodes = _get_nodes_plotly(g, weights)
        traces = edges + [nodes]

        if vis_opts is None:
            vis_opts = {}

        layout = dict(
            title='Network graph',
            titlefont=dict(size=16),
            showlegend=False,
            # hovermode='none',
            margin=dict(b=20, l=5, r=5, t=40),
            hovermode='closest',
            hoverdistance=-1,
            annotations=[dict(
                text="Super-Network",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002)],
            scene=dict(
                camera=dict(
                    # up=dict(x=0, y=0, z=1),
                    # center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=-0.1, z=-2)
                )
            ),
        )

        fig = dict(opts=vis_opts,
                   win=vis_win,
                   eid=self.env,
                   data=traces,
                   layout=layout)

        self.vis._send(fig)


def _get_edges_plotly(graph, weights=None):
    edges = []
    nodes_pos = graph.nodes('pos3d', default=(-1, -1, -1))

    for u, v in graph.edges:
        if nodes_pos[u] == nodes_pos[v]:
            # Don't add trace for overlapping nodes
            continue

        x, y, z = zip(nodes_pos[u], nodes_pos[v])  # (x_u, x_v), (y_u, y_v), ...
        mid_x, mid_y, mid_z = np.mean(x), np.mean(y), np.mean(z)

        val = weights[(u, v)] if weights else 1

        edge_trace = Scatter3d(
            x=x,
            y=y,
            z=z,
            line=Line(
                width=10,
                showscale=True,
                color=[val, val],
                cmin=0,
                cmax=1,
                colorscale='Viridis',
            ),
            hoverinfo="none",
            mode='lines',
        )
        edges.append(edge_trace)

        middle_node = Scatter3d(
            x=[mid_x],
            y=[mid_y],
            z=[mid_z],
            name='Edge{}->{}: {:.2f}'.format(u, v, val),
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.01
            ),
            hoverinfo="name",
        )

        edges.append(middle_node)

    return edges


def _get_nodes_plotly(graph, weights=None):
    nodes_trace = Scatter3d(
        x=[],
        y=[],
        z=[],
        text=[],
        name=[],
        mode='markers',
        hoverlabel=dict(bgcolor='rgb(10,10,10)'),
        hoverinfo='all',

        hovertext='all',
        marker=Marker(
            symbol='circle',
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=dict(
                thickness=10,
                title='Nodes colorbar',
                xanchor='left',
                titleside='right'
            ),
            colorscale='Viridis',
            color=[],
            size=5,
            line=dict(width=2)
        )
    )

    nodes_pos = graph.nodes('pos3d', default=(-1, -1, -1))

    for node in graph.nodes():
        if isinstance(node[0], tuple):
            continue
        x, y, z = nodes_pos[node]
        nodes_trace['x'].append(x)
        nodes_trace['y'].append(y)
        nodes_trace['z'].append(z)

        val = weights[node] if weights else 1

        # nodes_trace['marker']['color'].append(val)
        nodes_trace['marker']['color'].append('rgb(255,0,0)')
        node_info = 'Value : {} '.format(val)
        nodes_trace['text'].append(node_info)
        nodes_trace['name'].append(node_info)

    return nodes_trace


if __name__ == '__main__':
    Drawer.set_default_env('main')
    d = Drawer()
    d.plotly_graph()
