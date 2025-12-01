"""
Biological Evaluation of Causal graph prediction.

Using the CollecTRI TF-target interaction Collection.
"""

import numpy as np
#import decoupler as dc
from anndata import AnnData
import pandas as pd


def colltectri_subgraph(genes,collectri_mapping):
    """Get the subset of the CollecTRI dataset that is relevant for the given genes.
    """
    collectri = collectri_mapping

    tf_list = collectri['source'].astype('category').cat.categories
    target_list = collectri['target'].astype('category').cat.categories

    # 1. get relevant TFs, Target genes & interactions for given genes in adata_obj
    adata_tf_list = [tf for tf in tf_list if tf in genes]
    adata_target_list = [target for target in target_list if target in genes]
    collectri = collectri[collectri['source'].isin(adata_tf_list)]
    collectri = collectri[collectri['target'].isin(adata_target_list)]

    collectri = collectri[collectri['source'].isin(tf_list)]
    collectri = collectri[collectri['target'].isin(target_list)]

    return collectri

def collectri_overlap(
        genes: list,
        adjacency_matrix: np.array,
        colltectri_mapping=None,
        verbose: bool = False, 
        ):
    """Evaluate the network's positive predictions using the CollecTRI dataset.

    Only takes into account interactions outgoing from TFs (for human data only).

    Args:
        genes: list of genes
        adjacency_matrix: The (binary) adjacency matrix of the network
        colltectri_mapping: dict, mapping of TFs to their respective targets
        verbose: print recall and precision

    Returns:
        tuple: tuple containing:

            - recall (float): recall of the network's prediction
            - precision (float): precision of the network's prediction
        
    """
    collectri = colltectri_mapping

    tf_list = collectri['source'].astype('category').cat.categories
    target_list = collectri['target'].astype('category').cat.categories
    genes = [gene for gene in genes if not ":" in gene]

    # 1. get relevant TFs, Target genes & interactions for given genes in adata_obj
    adata_tf_list = [tf for tf in tf_list if tf in genes]
    adata_target_list = [target for target in target_list if target in genes]
    collectri = collectri[collectri['source'].isin(adata_tf_list)]
    collectri = collectri[collectri['target'].isin(adata_target_list)]
    if verbose:
        print("contained TFs: ",adata_tf_list)
        print("contained targets: ",adata_target_list)
        print("nr. of interactions: ", len(collectri))
    if len(collectri) == 0:
        print("No interactions found in collectri for given genes.")
        return 0, 0

    # 2. get list of interactions from adata_obj
    columns = ['source','target']
    predicted = pd.DataFrame(columns=columns)
    for source_idx, source in enumerate(genes):
        for target_idx, target in enumerate(genes):
            if adjacency_matrix[source_idx,target_idx] == 1:
                if source in collectri.source.values:
                    df = pd.DataFrame([[source,target]], columns=columns)
                    predicted = pd.concat([predicted, df], ignore_index=True)

    # 3. Compare to get Recall
    n_collectri = collectri.shape[0]
    n_predicted = predicted.shape[0]
    n_overlap   = collectri.merge(predicted, on=['source','target'], how='inner').shape[0]
    if n_predicted == 0:
        print("No interactions predicted.")
        return 0, 0
    recall = n_overlap/n_collectri
    precision = n_overlap/n_predicted
    if verbose:
        print("recall: ", recall)
        print("precision: ", precision)
    
    return recall, precision