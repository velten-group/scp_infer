"""Evaluation of Causal Graph Prediction for Simulated Data, i.e. when Ground-Truth isis available."""

import numpy as np
from anndata import AnnData
import networkx as nx
#import cdt


def symetric_extension(adjacency: np.ndarray) -> np.ndarray:
    """Extend the adjacency matrix to be symmetric.

    Args:
        adjacency: The adjacency matrix.

    Returns:
        np.ndarray: The symmetric adjacency matrix.
    """
    return np.maximum(adjacency, adjacency.T)

def indirect_extension(adjacency: np.ndarray, steps: int = 1) -> np.ndarray:
    """Extend the adjacency matrix to include indirect edges.

    Args:
        adjacency: The adjacency matrix.
        steps: The number of steps to consider for indirect edges.

    Returns:
        np.ndarray: The adjacency matrix with indirect edges.
    """
    #return np.clip(adjacency + np.linalg.matrix_power(adjacency, steps), 0, 1)
    indirect_matrix = np.zeros(adjacency.shape)
    for i in range (steps):
        indirect_matrix += np.linalg.matrix_power(adjacency, i+1)
    return np.clip(adjacency + indirect_matrix, 0, 1)

def edge_count_difference(gt_adjacency: np.ndarray, pred_adjacency: np.ndarray, indirect_steps: int = 2, verbose: bool = False) -> dict:
    """Separate edges in the predicted adjacency matrix into categories:
    - True Positive (TP): edge is present in both the GT and P adjacency matrix
    - Indirect True Positives (ITP): edge mediates an indirect connection between two nodes in the GT adjacency matrix (specify number of steps??)
    - Symmetric True Positives (STP): False Positives that are present with the opposite direction in the ground-truth adjacency matrix
    - False Positives (FP): edges that are present in the predicted but not in the ground-truth adjacency matrix

    Args:
        gt_adjacency: The ground-truth adjacency matrix.
        pred_adjacency: The predicted adjacency matrix.
        indirect_steps: The number of steps to consider for indirect edges - default is 2 as 1 step is direct edge.

    Returns:
        dict: A dictionary containing the counts of each edge category.
    """
    assert gt_adjacency.shape == pred_adjacency.shape, "Ground-truth and predicted adjacency matrices must have the same shape."
    assert indirect_steps >= 1, "Number of steps for indirect edges must be greater than or equal to 1."
    assert np.all(np.logical_or(pred_adjacency == 0, pred_adjacency == 1)), "Predicted adjacency matrix must be binary."
    assert np.all(np.logical_or(gt_adjacency == 0, gt_adjacency == 1)), "Ground-truth adjacency matrix must be binary."

    # Calculate the True Positives (TP)
    tp_matrix = np.logical_and(gt_adjacency, pred_adjacency)
    tp_count = np.sum(tp_matrix)

    # Calculate the Indirect True Positives (ITP)
    itp_matrix = np.logical_and(indirect_extension(gt_adjacency, indirect_steps), pred_adjacency)
    itp_count = np.sum(itp_matrix)
    itp_exclusive = np.logical_and(itp_matrix, np.logical_not(tp_matrix))
    itp_count_exclusive = np.sum(itp_exclusive)

    # Calculate the Symmetric True Positives (STP)
    sym_gt_adjacency = symetric_extension(gt_adjacency)
    stp_matrix = np.logical_and(sym_gt_adjacency, pred_adjacency)
    stp_count = np.sum(stp_matrix)
    stp_exclusive = np.logical_and(stp_matrix, np.logical_not(tp_matrix))
    stp_count_exclusive = np.sum(stp_exclusive)

    # Calculate the False Positives (FP)
    fp_matrix = np.logical_and(np.logical_not(gt_adjacency), pred_adjacency)
    fp_count = np.sum(fp_matrix)
    exclude = np.logical_or(tp_matrix, itp_matrix)
    exclude = np.logical_or(exclude, stp_matrix)
    fp_exclusive = np.logical_and(fp_matrix, np.logical_not(exclude))
    fp_count_exclusive = np.sum(fp_exclusive)

    # verify that the counts are correct
    if verbose:
        print("TP: ", tp_count)
        print("ITP: ", itp_count)
        print("ITP exclusive: ", itp_count_exclusive)
        print("STP: ", stp_count)
        print("STP exclusive: ", stp_count_exclusive)
        print("FP: ", fp_count)
        print("FP exclusive: ", fp_count_exclusive)
        print("Total positives: ", np.sum(pred_adjacency))
        print("Total edges in GT: ", np.sum(gt_adjacency))
        print("TP + FP: ", tp_count + fp_count, " == ", np.sum(pred_adjacency))
        print("TP + ITP_e + STP_e + FP_e: ", tp_count + itp_count_exclusive + stp_count_exclusive + fp_count_exclusive, " == ", np.sum(pred_adjacency))
    

    # pytest: assert tp_count + fp_count == np.sum(pred_adjacency)
    # pytest: assert tp_count + itp_count_exclusive + stp_count_exclusive + fp_count_exclusive == np.sum(pred_adjacency)
    # pytest: test results with simple examples
    return {
        "TP": tp_count,
        "ITP": itp_count,
        "ITP_exclusive": itp_count_exclusive,
        "STP": stp_count,
        "STP_exclusive": stp_count_exclusive,
        "FP": fp_count,
        "FP_exclusive": fp_count_exclusive
    }




def precision(gt_adjacency: np.ndarray, pred_adjacency: np.ndarray) -> float:
    """Calculate the precision of the predicted adjacency matrix.

    Args:
        gt_adjacency: The ground-truth adjacency matrix.
        pred_adjacency: The predicted adjacency matrix.
    
    Returns:
        float: The precision of the predicted adjacency matrix.
    """
    if gt_adjacency.shape != pred_adjacency.shape:
        raise ValueError(
            "Ground-truth and predicted adjacency matrices must have the same shape.")
    tp = np.sum(np.logical_and(gt_adjacency, pred_adjacency))
    fp = np.sum(np.logical_and(np.logical_not(gt_adjacency), pred_adjacency))
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def recall(gt_adjacency: np.ndarray, pred_adjacency: np.ndarray) -> float:
    """Calculate the recall of the predicted adjacency matrix.

    Args:
        gt_adjacency: The ground-truth adjacency matrix.
        pred_adjacency: The predicted adjacency matrix.

    Returns:
        float: The recall of the predicted adjacency matrix.
    """
    if gt_adjacency.shape != pred_adjacency.shape:
        raise ValueError(
            "Ground-truth and predicted adjacency matrices must have the same shape.")

    tp = np.sum(np.logical_and(gt_adjacency, pred_adjacency))
    fn = np.sum(np.logical_and(gt_adjacency, np.logical_not(pred_adjacency)))
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f_score(gt_adjacency: np.ndarray, pred_adjacency: np.ndarray, beta: float = 1.0) -> float:
    """Calculate the F-score of the predicted adjacency matrix.

    Args:
        gt_adjacency: The ground-truth adjacency matrix.
        pred_adjacency: The predicted adjacency matrix.
        beta: The beta parameter for the F-score.

    Returns:
        float: The F-score of the predicted adjacency matrix.
    """
    if gt_adjacency.shape != pred_adjacency.shape:
        raise ValueError(
            "Ground-truth and predicted adjacency matrices must have the same shape.")

    p = precision(gt_adjacency, pred_adjacency)
    r = recall(gt_adjacency, pred_adjacency)
    if p + r == 0:
        return 0
    return (1 + beta**2) * (p * r) / (beta**2 * p + r)


def precision_recall_curve(gt_adjacency: np.ndarray, pred_adjacency: np.ndarray) -> tuple:
    """Calculate the precision-recall curve of a weighted predicted adjacency matrix by varying the cutoff.

    Args:
        gt_adjacency: The ground-truth adjacency matrix.
        pred_adjacency: The predicted adjacency matrix.

    Returns:
        tuple: tuple containing:
            - precision (list[float]): list of precision vals of the predicted adjacency matrix
            - recall (list[float]): list of recall vals of the predicted adjacency matrix
    """
    if gt_adjacency.shape != pred_adjacency.shape:
        raise ValueError(
            "Ground-truth and predicted adjacency matrices must have the same shape.")

    min = np.min(pred_adjacency)
    max = np.max(pred_adjacency)
    precision_vals = []
    recall_vals = []
    for cutoff in np.linspace(min, max, 100):
        pred_adjacency_cutoff = (pred_adjacency > cutoff).astype(int)
        precision_vals.append(precision(gt_adjacency, pred_adjacency_cutoff))
        recall_vals.append(recall(gt_adjacency, pred_adjacency_cutoff))
    return precision_vals, recall_vals


def structural_hamming_distance(gt_adjacency: np.ndarray, pred_adjacency: np.ndarray) -> float:
    """Calculate the structural Hamming distance between the ground-truth and predicted adjacency matrices.

    Args:
        gt_adjacency: The ground-truth adjacency matrix.
        pred_adjacency: The predicted adjacency matrix.

    Returns:
        float: The structural Hamming distance between the ground-truth and predicted adjacency matrices.
    """
    if gt_adjacency.shape != pred_adjacency.shape:
        raise ValueError(
            "Ground-truth and predicted adjacency matrices must have the same shape.")

    abs_diff =  np.abs(gt_adjacency - pred_adjacency)
    mistakes = abs_diff + np.swapaxes(abs_diff, -2, -1)  # mat + mat.T (transpose of last two dims)

    # ignore double edges
    mistakes_adj = np.where(mistakes > 1, 1, mistakes)

    return np.triu(mistakes_adj).sum((-1, -2))
    #return cdt.metrics.SHD(target=gt_adjacency, pred=pred_adjacency, double_for_anticausal=False)


def compute_path_matrix(adj_matrix):
    """
    Computes the reachability matrix (path matrix) for a given adjacency matrix - now using networkx.
    """
    G = nx.DiGraph(adj_matrix)
    shortest_path_lengths = nx.floyd_warshall_numpy(G)
    path_matrix = np.zeros_like(adj_matrix)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if shortest_path_lengths[i, j] != np.inf:
                path_matrix[i, j] = 1
    return path_matrix

def is_valid_adjustment_set(G, X, Y, Z):
    """
    Check if a given set Z is a valid adjustment set for estimating the effect of X on Y in a DAG G.

    Parameters:
        G (nx.DiGraph): The causal DAG.
        X (str): The treatment (exposure) variable.
        Y (str): The outcome variable.
        Z (set): The proposed adjustment set.

    Returns:
        bool: True if Z is a valid adjustment set, False otherwise.
    """
    # Get all nodes in the graph
    all_nodes = set(G.nodes())

    # Step 1: Identify backdoor paths from X to Y
    backdoor_paths = [
        path for path in nx.all_simple_paths(nx.DiGraph(G.to_undirected()), source=X, target=Y)
        # Ensures the path starts with an incoming edge to X
        if G.has_edge(path[1], X)
    ]

    # Step 2: Check if Z blocks all backdoor paths
    for path in backdoor_paths:
        path_blocked = False
        for node in path[1:-1]:  # Exclude X and Y themselves
            # Z blocks a non-collider
            if node in Z and not is_collider(G, node, path):
                path_blocked = True
                break
            if is_collider(G, node, path) and not has_descendant_in_Z(G, node, Z):
                path_blocked = True
                break
        if not path_blocked:
            return False  # If any path is not blocked, Z is invalid

    # Step 3: Ensure Z does not include descendants of X
    descendants_of_X = nx.descendants(G, X)
    if any(node in Z for node in descendants_of_X):
        return False

    # If all checks pass, Z is valid
    return True


def is_collider(G, node, path):
    """
    Check if a node is a collider on a given path in DAG G.
    """
    idx = path.index(node)
    # Colliders cannot be the first or last node
    if idx == 0 or idx == len(path) - 1:
        return False
    return G.has_edge(path[idx - 1], node) and G.has_edge(path[idx + 1], node)


def has_descendant_in_Z(G, node, Z):
    """
    Check if any descendant of a node is in the adjustment set Z.
    """
    descendants = nx.descendants(G, node)
    return any(descendant in Z for descendant in descendants)

def structural_intervention_distance(G, H, verbose=False):
    """
    Computes the Structural Intervention Distance (SID) between two causal graphs G and H. Does not work for the case of CPDAGs or cyclic graphs, that might b handled by the SID implementation in R. (https://cran.r-project.org/web/packages/SID/index.html)

    Args:
        G (np.ndarray): The ground-truth adjacency matrix.
        H (np.ndarray): The predicted adjacency matrix.
        verbose (bool): Whether to print intermediate results.
    
    Returns:
        float: The Structural Intervention Distance (SID) between the two causal graphs.
    """

    #check that both are DAGs
    if not nx.is_directed_acyclic_graph(nx.DiGraph(G)):
        raise ValueError("The ground-truth adjacency matrix must be a DAG.")
    if not nx.is_directed_acyclic_graph(nx.DiGraph(H)):
        raise ValueError("The predicted adjacency matrix must be a DAG.")
    p = np.shape(G)[0]

    # path matrix
    path_matrix_G = compute_path_matrix(G)
    if verbose:
        print('path_matrix_G:\n', path_matrix_G)
    # path_matrix_desc_g = path_matrix_G - np.eye(p)
    incorrect_causal_effects = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            # if j in PA(i;H):
            parents_i_H = np.where(H[:, i] == 1)[0]
            if verbose:
                print(f'parents_{i}_H:', parents_i_H)
            if j in parents_i_H:
                if verbose:
                    print(f'{j} in PA({i};H)')
                # check j in DESC(i;G)
                if path_matrix_G[i, j] == 1:
                    if verbose:
                        print(f'{j} in DESC({i};G)')
                    incorrect_causal_effects[i, j] += 1
                else:
                    if verbose:
                        print(f'{j} not in DESC({i};G)')
            else:
                # check PA(i;H) is not a valid adjustment set for (G; i; j)
                if verbose:
                    print(f'{j} not in PA({i};H)')
                G_graph = nx.DiGraph(G)
                if not is_valid_adjustment_set(G_graph, i, j, parents_i_H):
                    if verbose:
                        print(
                            f'PA({i};H) is not a valid adjustment set for ({i};{j})')
                    incorrect_causal_effects[i, j] += 1
                else:
                    if verbose:
                        print(
                            f'PA({i};H) is a valid adjustment set for ({i};{j})')
    SID = np.sum(incorrect_causal_effects)
    return SID

#def structural_intervention_distance(gt_interv: np.ndarray, pred_interv: np.ndarray) -> float:
#    """Calculate the structural intervention distance between the ground-truth and predicted intervention matrices.
#
#    Args:
#        gt_interv: The ground-truth intervention matrix.
#        pred_interv: The predicted intervention matrix.
#
#    Returns:
#        float: The structural intervention distance between the ground-truth and predicted intervention matrices.
#    """
#    if gt_interv.shape != pred_interv.shape:
#        raise ValueError(
#            "Ground-truth and predicted intervention matrices must have the same shape.")
#
#    return cdt.metrics.SID(target=gt_interv, pred=pred_interv)
#
