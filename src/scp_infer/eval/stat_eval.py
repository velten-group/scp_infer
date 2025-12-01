"""
Statistical / Data-Driven Evaluation of Causal graph prediction.

Functions for statistical evaluation of Causal Graph predictions.
Some of these are closely related to the CausalBench approaches, whose code is available at:
https://github.com/causalbench/causalbench
"""
import numpy as np
import scipy
from anndata import AnnData
import networkx as nx
import scanpy as sc


def _get_observational(adata_obj: AnnData, child: str) -> np.array:
    """Return all the samples for gene "child" in cells where there was no perturbations

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        child: Gene name of child to get samples for

    Returns:
        np.array: 1D-matrix of corresponding samples
    """
    observations = adata_obj[adata_obj.obs["perturbation"] == "non-targeting"]
    gene_index = adata_obj.var_names.get_loc(child)
    observations = observations[:, gene_index].X
    return np.reshape(observations, np.size(observations))


def _get_interventional(adata_obj: AnnData, child: str, parent: str) -> np.array:
    """Return all the samples for gene "child" in cells where "parent" was perturbed

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        child: Gene name of child to get samples for
        parent: Gene name of gene that must have been perturbed

    Returns:
        np.array: 1D-matrix of corresponding samples
    """
    observations = adata_obj[adata_obj.obs["perturbation"] == parent]
    gene_index = adata_obj.var_names.get_loc(child)
    observations = observations[:, gene_index].X
    return np.reshape(observations, np.size(observations))


def evaluate_wasserstein(
        adata_obj: AnnData,
        adjacency_matrix: np.array,
        p_value_threshold: float = 0.05,
        verbose: bool = False
):
    """Evaluate the network's positive predictions (pair of genes with a directed edge) using the observational and interventional data.

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        adjacency_matrix: The (binary) adjacency matrix of the network
        p_value_threshold: threshold for statistical significance, default 0.05

    Returns:
        tuple: tuple containing:

            - true_positive (int): number of true positives
            - false_positive (int): number of false positives
            - wasserstein_distances (list): list of wasserstein distances between observational and interventional samples
    """
    gene_names = adata_obj.var_names
    true_positive = 0
    false_positive = 0
    wasserstein_distances = []
    network_graph = nx.from_numpy_array(
        adjacency_matrix.T, create_using=nx.DiGraph)
    for parent in network_graph.nodes():
        children = network_graph.successors(parent)
        for child in children:
            # getting obs. samples
            observational_samples = _get_observational(
                adata_obj, gene_names[child])
            # getting int. samples
            interventional_samples = \
                _get_interventional(
                    adata_obj, gene_names[child], gene_names[parent])
            # ranking and whitney U test
            if len(observational_samples) == 0 or len(interventional_samples) == 0:
                # skip if no samples are available, e.g. due to no perturbations
                continue
            ranksum_result = scipy.stats.mannwhitneyu(
                observational_samples, interventional_samples
            )
            # getting wassertstein distance
            wasserstein_distance = scipy.stats.wasserstein_distance(
                observational_samples, interventional_samples,
            )
            wasserstein_distances.append(wasserstein_distance)
            p_value = ranksum_result[1]
            if verbose:
                print("obs. samples: ", np.shape(observational_samples))
                print("int. samples: ", np.shape(interventional_samples))
                print("wasserstein_distance: ", wasserstein_distance)

            if p_value < p_value_threshold:
                # Mannwhitney test rejects the hypothesis that the two distributions are similar
                # -> parent has an effect on the child
                true_positive += 1
            else:
                false_positive += 1
    return true_positive, false_positive, wasserstein_distances


def evaluate_f_o_r(adata_obj: AnnData, adjacency_matrix: np.array, p_value_threshold: float = 0.05):
    """Evaluate the network's negative predictions (pair of genes where the latter is not downstream of the former) using the observational and interventional data.

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        adjacency_matrix: The (binary) adjacency matrix of the network
        p_value_threshold: threshold for statistical significance, default 0.05

    Returns:
        tuple: tuple containing:

        - false_omission_rate: estimated false omission rate
        - negative_mean_wasserstein: mean wasserstein distance of negative predictions
    """

    network_graph = nx.from_numpy_array(
        adjacency_matrix.T, create_using=nx.DiGraph)
    tranclo_graph = nx.transitive_closure(network_graph)
    independent_pair_graph = nx.complement(tranclo_graph)
    # remove self loops
    independent_pair_graph.remove_edges_from(
        nx.selfloop_edges(independent_pair_graph))
    unrelated_adj_matrix = nx.to_numpy_array(independent_pair_graph)

    # print("unrelated_adj_matrix: ", unrelated_adj_matrix)
    # print("Evaluating Wasserstein")

    f_p, _, wasserstein = evaluate_wasserstein(
        adata_obj, unrelated_adj_matrix, p_value_threshold)
    if independent_pair_graph.number_of_edges() == 0:
        print("No edges in independent pair graph")
        print("f_p: ", f_p)
        print("wasserstein: ", np.mean(wasserstein))
        print("adjacency_matrix: ", adjacency_matrix)
        return 0, 0
    else:
        false_omission_rate = f_p / independent_pair_graph.number_of_edges()
        negative_mean_wasserstein = np.mean(wasserstein)

        return false_omission_rate, negative_mean_wasserstein


def de_graph_hierarchy(
        adata_obj: AnnData,
        adjacency_matrix: np.array,
        verbose: bool = False
):
    """
    identify differentially expressed genes per perturbation and score whether they are
    placed upstream, downstream or unrelated to the perturbation in the network

    Args:
        adata_obj: annotated Anndata object containing the expression matrix and interventions
        adjacency_matrix: The (binary) adjacency matrix of the network

    Returns:
        tuple: tuple containing:

            - tag_arr (list): list of perturbed genes
            - upstream_arr (list): number of DE genes upstream of the perturbed gene
            - downstream_arr (list): number of DE genes downstream of the perturbed gene
            - unrelated_arr (list): number of DE genes unrelated to the perturbed gene
            - cyclic_arr (list): number of DE genes that are cyclic to the perturbed gene
    """
    perturbed_genes = adata_obj.var_names[adata_obj.var['gene_perturbed']]
    network_graph = nx.from_numpy_array(
        adjacency_matrix.T, create_using=nx.DiGraph)
    tranclo_graph = nx.transitive_closure(network_graph)

    # 1. compute DE genes for each perturbation with respect to rest (-> non-targeting?)
    # filter unusable cells
    # adata_obj = adata_obj.copy()
    adata_obj = adata_obj[adata_obj.obs['gene_perturbation_mask']
                          | adata_obj.obs['non-targeting']]
    # a. Add key to adata_obj.obs for perturbation groupings to be used in DE analysis
    adata_obj.obs['perturbation_group'] = adata_obj.obs['perturbation']
    adata_obj.obs['perturbation_group'] = adata_obj.obs['perturbation_group'].astype(
        'category')
    # perturbation group should only contain the perturbed genes and non-targeting
    print(adata_obj.obs['perturbation_group'].value_counts())

    # b. perform DE analysis
    key = 'rank_genes_perturbations'
    sc.tl.rank_genes_groups(
        adata_obj, groupby='perturbation_group', method='t-test', key_added=key, reference='non-targeting')
    reference = str(adata_obj.uns[key]["params"]["reference"])
    group_names = adata_obj.uns[key]["names"].dtype.names
    if verbose:
        print("perturbed_genes: ", perturbed_genes)
        print('reference:', reference)
        print('group_names:', group_names)

    # 2. compute the number of true positives
    tag_arr = []
    upstream_arr = []
    downstream_arr = []
    unrelated_arr = []
    cyclic_arr = []
    for perturbed_gene in perturbed_genes:
        dataframe = sc.get.rank_genes_groups_df(
            adata_obj, group=perturbed_gene, key=key)
        if perturbed_gene == 'non-targeting':
            continue
        # get the DE genes for the perturbation
        if verbose:
            print("perturbed_gene: ", perturbed_gene, type(perturbed_gene))
            print("dataframe: ", dataframe)
        perturbed_gene_index = adata_obj.var_names.get_loc(perturbed_gene)
        gene_names = adata_obj.uns[key]["names"][perturbed_gene]
        # remove the perturbed gene itself from DE genes
        gene_names = [gene for gene in gene_names if gene != perturbed_gene]
        # count where DE genes are located in the network
        tag = perturbed_gene
        upstream = 0
        downstream = 0
        unrelated = 0
        cyclic = 0
        for gene in gene_names:
            # if gene not in adata_obj.var_names:  # temporary fix - should not happen
            #     continue
            # 1st check the pvalue of the gene
            pval = dataframe.loc[dataframe['names']
                                 == gene, 'pvals_adj'].values[0]
            if pval > 0.05:
                continue
            gene_index = adata_obj.var_names.get_loc(gene)
            if gene_index in tranclo_graph.successors(perturbed_gene_index):
                if gene_index in tranclo_graph.predecessors(perturbed_gene_index):
                    cyclic += 1
                else:
                    downstream += 1
            elif gene_index in tranclo_graph.predecessors(perturbed_gene_index):
                upstream += 1
            else:
                unrelated += 1
        for entry, arr in zip(
            [tag, upstream, downstream, unrelated, cyclic],
            [tag_arr, upstream_arr, downstream_arr, unrelated_arr, cyclic_arr]
        ):
            arr.append(entry)

    return tag_arr, upstream_arr, downstream_arr, unrelated_arr, cyclic_arr

def find_cutoff(adata_obj: AnnData, adj_matrix: np.array, exp: float = 1.5, verbose = False):
    """Find the optimal cutoff for the adjacency matrix using Wasserstein distance"""

    cutoffs = np.linspace(0, np.max(adj_matrix), 20)[:-1]
    wstein_array = []
    n_edges_array = []

    for cutoff in cutoffs:
        bool_adj_matrix = (adj_matrix > cutoff).astype(int)
        _, _, wstein_distances = evaluate_wasserstein(adata_obj, bool_adj_matrix)
        wstein_array.append(np.mean(wstein_distances))
        n_edges_array.append(np.sum(bool_adj_matrix))

    if verbose:
        print("wstein_array: ", wstein_array)
        print("n_edges_array: ", n_edges_array)

    # filter out nan values
    nan_wstein = np.isnan(wstein_array)
    wstein_array = np.array(wstein_array)[~nan_wstein]
    n_edges_array = np.array(n_edges_array)[~nan_wstein]
    cutoffs = cutoffs[~nan_wstein]
    # find the optimal cutoff
    # normalize the values
    wstein_array = wstein_array / np.mean(wstein_array)
    vals = wstein_array**exp * n_edges_array
    #vals = np.exp(np.array(wstein_array)*exp) * (np.array(n_edges_array))
    if verbose:
        print("vals: ", vals)
    optimal_cutoff = cutoffs[np.argmax(vals)]
    return optimal_cutoff
        