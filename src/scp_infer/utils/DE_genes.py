import numpy as np
import networkx as nx
import pandas as pd
import scanpy as sc
import scipy.stats as stats
import warnings


from ..eval import structural_hamming_distance

def get_DE_expression_matrix(adata, score = 'z-score', test='t-test'):
    """ 
    Returns a matrix with the DE scores for each gene under each perturbation.
    """
    genes = adata.var_names
    perturbed_genes = adata.var_names[adata.var['gene_perturbed']]
    DE_measures = pd.DataFrame(columns=['perturbation','gene','wasserstein','z-score','pval','logfoldchanges'], dtype=float)

    adata.obs['perturbation_group'] = adata.obs['perturbation']
    adata.obs['perturbation_group'] = adata.obs['perturbation_group'].astype('category')
    key = 'rank_genes_perturbations'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(adata, groupby='perturbation_group', method=test, key_added=key, reference='non-targeting')
    i = 0

    for perturbation in perturbed_genes:
        results_df = sc.get.rank_genes_groups_df(adata, group=perturbation,key=key)
        for target_gene in genes:
            target_idx = adata.var.index.get_loc(target_gene)
            target_data = adata[adata.obs['perturbation'] == perturbation].X[:, target_idx]
            unpert_data = adata[adata.obs['perturbation'] == 'non-targeting'].X[:, target_idx]
            wstein = stats.wasserstein_distance(target_data, unpert_data)
            zscore = results_df['scores'][results_df['names'] == target_gene].values[0]
            pval = results_df['pvals_adj'][results_df['names'] == target_gene].values[0]
            logfoldchanges = results_df['logfoldchanges'][results_df['names'] == target_gene].values[0]
            DE_measures.loc[i] = [perturbation, target_gene, wstein, zscore, pval,logfoldchanges]
            i += 1

    score_mat = DE_measures.pivot(index='perturbation', columns='gene', values=score)
    return score_mat


def transitive_reduction(G: nx.DiGraph):
    """
    Compute the transitive reduction of a directed graph G.
    The transitive reduction of a directed graph is the smallest graph that has the same reachability relation as G.
    """
    # Create a copy of the original graph to avoid modifying it
    G_reduced = G.copy()

    # Iterate over all edges in the graph
    for u, v in list(G.edges()):
        # Check if there is a path from u to v through any other node
        for w in list(G.nodes()):
            if w != u and w != v and G.has_edge(u, w) and G.has_edge(w, v):
                # If such a path exists, remove the edge (u, v)
                G_reduced.remove_edge(u, v)
                break

    return G_reduced

def complete_with_indirect_edges(adjacency_matrix,n_iterations=2):
    """
    Complete the adjacency matrix with indirect edges by adding multiplication of the adjacency matrix with itself.
    """
    adjacency_matrix_original = np.array(adjacency_matrix)
    adjacency_matrix_step = np.array(adjacency_matrix)
    for i in range(n_iterations):
        adjacency_matrix_step = adjacency_matrix_original + np.dot(adjacency_matrix_original, adjacency_matrix_step)
        adjacency_matrix_step[adjacency_matrix_step > 0.5] = 1
    
    return adjacency_matrix_step

def compute_graph(adjacency_matrix):
    # create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return G

def compute_true_positives(adj_mat, significant_pairs):
    # compute the true positive edges, i.e. edges in the adjacency matrix that are also significant
    true_positives = np.logical_and(adj_mat, significant_pairs).astype(int)
    np.fill_diagonal(true_positives, 0)
    return true_positives

def compute_false_positives(adj_mat, significant_pairs):
    # compute the false positive edges, i.e. edges in the adjacency matrix that are not significant
    false_positives = np.logical_and(adj_mat, np.logical_not(significant_pairs)).astype(int)
    np.fill_diagonal(false_positives, 0)
    return false_positives

def compute_false_negatives(adj_mat, significant_pairs):
    # compute the false negative edges, i.e. edges that are significant but have no path in the adjacency matrix
    false_negatives = np.logical_and(np.logical_not(adj_mat), significant_pairs).astype(int)
    np.fill_diagonal(false_negatives, 0)
    return false_negatives

def compute_true_negatives(adj_mat, significant_pairs):
    # compute the true negative edges, i.e. edges that are not significant and have no path in the adjacency matrix
    true_negatives = np.logical_and(np.logical_not(adj_mat), np.logical_not(significant_pairs)).astype(int)
    np.fill_diagonal(true_negatives, 0)
    return true_negatives



def f_beta_score(precision, recall, beta):
    if precision + recall == 0:
        return 0.0
    if beta < 0:
        raise ValueError("Beta must be non-negative.")
    else:
        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def get_scores(adj_mat, significant_pairs):
    adj_mat_original = adj_mat
    n_indirect_list = list(range(1,11))
    evaluation_metrics = pd.DataFrame()

    adj_mat_list = [adj_mat_original]
    for n_indirect in n_indirect_list:
        adj_mat_completed_n = complete_with_indirect_edges(adj_mat_original, n_iterations=n_indirect)
        adj_mat_list.append(adj_mat_completed_n)
    adj_mat_completed = nx.to_numpy_array(nx.transitive_closure(nx.from_numpy_array(adj_mat_original, create_using=nx.DiGraph)))
    adj_mat_list.append(adj_mat_completed)
    n_indirect_list = [0] + n_indirect_list + ['inf']
    for adj_mat,n_indirect in zip(adj_mat_list, n_indirect_list):
        # compute the TP, FP, FN, TN
        true_positives = compute_true_positives(adj_mat, significant_pairs)
        false_positives = compute_false_positives(adj_mat, significant_pairs)
        false_negatives = compute_false_negatives(adj_mat, significant_pairs)
        true_negatives = compute_true_negatives(adj_mat, significant_pairs)
        
        assert np.sum(true_positives) + np.sum(false_positives) + np.sum(false_negatives) + np.sum(true_negatives) == adj_mat.shape[0] * (adj_mat.shape[1]-1), 'sum of all entries does not equal the possible number of edges'
        negatives = np.logical_not(significant_pairs)
        n_edges = np.sum(adj_mat)
        
        # compute the derived metrics
        TP_rate = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
        FP_rate = np.sum(false_positives) / (np.sum(negatives))
        precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
        recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
        f1_score = f_beta_score(precision, recall, 1)
        f05_score = f_beta_score(precision, recall, 0.5)
        accuracy = (np.sum(true_positives) + np.sum(true_negatives)) / (np.sum(true_positives) + np.sum(true_negatives) + np.sum(false_positives) + np.sum(false_negatives))
        shd = structural_hamming_distance(significant_pairs, adj_mat)
        
        # add new row to the dataframe
        evaluation_metrics = pd.concat([evaluation_metrics, pd.DataFrame({'n_indirect': n_indirect, 'n_edges': n_edges, 'TP': np.sum(true_positives), 'FP': np.sum(false_positives),
                                    'FN': np.sum(false_negatives), 'TN': np.sum(true_negatives),
                                    'TP_rate': TP_rate, 'FP_rate': FP_rate,
                                    'precision': precision, 'recall': recall, 'f1_score': f1_score,
                                    'f05_score': f05_score, 'accuracy': accuracy, 'shd': shd}, index=[0])], ignore_index=True)
    evaluation_metrics = evaluation_metrics.set_index('n_indirect')
    return evaluation_metrics