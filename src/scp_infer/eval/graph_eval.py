"""
Evaluation of Predicted Causal Graphs.
"""

import numpy as np
import scipy
from anndata import AnnData
import scanpy as sc
import networkx as nx


def jaccard_index(
        adjacency1: np.ndarray,
        adjacency2: np.ndarray,
        cutoff1: float = 0.5,
        cutoff2: float = 0.5
):
    """Calculate the Jaccard index between two adjacency matrices.

    Args:
        adjacency1: The first adjacency matrix.
        adjacency2: The second adjacency matrix.
        cutoff1: The cutoff value for the first adjacency matrix. Defaults to 0.5.
        cutoff2: The cutoff value for the second adjacency matrix. Defaults to 0.5.

    Returns:
        float: The Jaccard index between the two adjacency matrices.

    """
    adjacency1 = (adjacency1 > cutoff1).astype(int)
    adjacency2 = (adjacency2 > cutoff2).astype(int)
    intersection = np.sum(np.logical_and(adjacency1, adjacency2))
    union = np.sum(np.logical_or(adjacency1, adjacency2))
    return intersection / union


def jaccard_pairwise(adjacency_matrices: list) -> list:
    """
    Calculate the pairwise Jaccard index between a list of adjacency matrices.

    Args:
        adjacency_matrices (List of numpy.ndarray): List of adjacency matrices.

    Returns:
        list: Pairwise Jaccard index between the adjacency matrices.
    """
    n = len(adjacency_matrices)
    jaccard_values = []
    for i in range(n):
        for j in range(i+1, n):
            jaccard_values.append(jaccard_index(
                adjacency_matrices[i], adjacency_matrices[j]))
    return jaccard_values


def graph_edit_distance(adjacency_matrices) -> list:
    """
    Calculate the pairwise graph edit distance between a list of adjacency matrices.
    use networkx.graph_edit_distance

    Args:
        adjacency_matrices (List of numpy.ndarray): List of adjacency matrices.

    Returns:
        list: Pairwise graph edit distance between the adjacency matrices.
    """
    n = len(adjacency_matrices)
    graph_edit_values = []
    for i in range(n):
        for j in range(i+1, n):
            graph_edit_values.append(nx.graph_edit_distance(nx.from_numpy_array(
                adjacency_matrices[i]), nx.from_numpy_array(adjacency_matrices[j])))
    return graph_edit_values


def edge_density(adjacency: np.ndarray) -> float:
    """Calculate the edge density of an adjacency matrix.

    Args:
        adjacency: The adjacency matrix.

    Returns:
        float: The edge density of the adjacency matrix.
    """
    n = adjacency.shape[0]
    return np.sum(adjacency) / (n * (n - 1))
