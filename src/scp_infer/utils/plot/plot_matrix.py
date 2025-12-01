"""Plotting functions for Inference Outputs"""

import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns


"""
def plot_adjacency_matrix(
        estimate: np.ndarray,
        title: str = "GIES",
        output_folder: str = "../data/data_out",
        show: bool = False
) -> None:
    """
    #Plot the adjacency matrix of the estimated graph.

    #Args:
    #    estimate: The estimated adjacency matrix
    #    title: The title of the graph
    #    output_folder: The output folder
    #    show: Whether to show the plot
"""
    _, ax = plt.subplots()
    fig1 = ax.matshow(estimate)
    plt.colorbar(fig1)
    plt.title(title + ": Adjacency matrix")
    plt.savefig(os.path.join(output_folder, title + "_adjacency_matrix.png"))
    if show:
        plt.show()
    plt.close()

    return None
"""

def plot_adjacency_matrix(adjacency_matrix, var_names, ax, title, xlabel = 'genes', ylabel = 'perturbations', filename = None):
    #fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(adjacency_matrix, ax=ax, cbar=False,linewidths = 1, linecolor = 'gray', cmap='gray',vmin=-0.4,vmax=1,square=True, xticklabels=var_names, yticklabels=var_names)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    #ax.set_xticks(adata.var_names, fontsize=10)
    #ax.set_yticks(adata.var_names, fontsize=10)
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_dag(graph: nx.Graph, figsize=(5, 5),title = None, axis = None) -> None:
    """Plot a directed acyclic graph with topological ordering using networkx.

    Args:
        graph: networkx graph
        figsize: Size of the figure


    Returns:
        None
    """
    G = graph
    for layer, nodes in enumerate(nx.topological_generations(G)):
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    pert_nodes = [node for node in G if G.pred[node] != {}]

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')
    for k in pos:
        pos[k][-1] *= -1

    # If axis is provided, use it, otherwise create a new figure and axis
    if axis is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = axis.figure
        ax = axis
    # 1. draw base Graph
    # color source nodes green, others blue
    color_map = []
    for node in G:
        if G.pred[node] == {}:
            color_map.append('green')
        else:
            color_map.append('blue')

    nx.draw_networkx(G, pos=pos, ax=ax, node_color=color_map, font_color='white')
    ax.set_title(title if title else "Directed Acyclic Graph")

    if axis is None:
        fig.tight_layout()
        plt.show()