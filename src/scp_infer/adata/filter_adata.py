""" Operations on AnnData objects to annotate perturbations, filter genes and adjust counts """

import numpy as np
import scipy
from anndata import AnnData
import scanpy as sc
from tqdm import tqdm
import warnings
import pandas as pd
from statsmodels.stats.multitest import multipletests



def value_counts(lst: list) -> list:
    """Get the value counts of a list and return them as a list of tuples.

    Args:
        list: list, list of values
    Returns:
        sorted_counts: list of tuples, sorted value counts
    """
    counts = {}
    for item in lst:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    # Convert the dictionary to a list of tuples and sort it
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts

def create_perturb_bools(
        adata_obj: AnnData,
        perturbation_entry='perturbation'
) -> None:
    """Create boolean masks for the perturbation labels from the AnnData object.

    Args:
        adata_obj: AnnData object
        perturbation_entry: str, name of the perturbation entry in the observation annotation
        non_targeting_label: str, label for the non-targeting perturbations
    Returns:
        None
    """
    # Get the perturbation labels and filter them (depends on the dataset being used)
    perturb_labels = adata_obj.obs[perturbation_entry].astype(str).copy()

    adata_obj.obs['non-targeting'] = adata_obj.obs['perturbation'] == 'non-targeting'
    adata_obj.obs['multiplet'] = adata_obj.obs['perturbation'] == "multiplet"
    adata_obj.obs['control'] = adata_obj.obs['perturbation'] == "control"
    adata_obj.obs['nan'] = adata_obj.obs['perturbation'] == "nan"
    adata_obj.obs['gene_perturbation_mask'] = ~adata_obj.obs['non-targeting'] & ~adata_obj.obs['multiplet'] \
        & ~adata_obj.obs['control'] & ~adata_obj.obs['nan']

    print("Non-targeting:", adata_obj.obs['non-targeting'].sum())
    print("Multiplet:", adata_obj.obs['multiplet'].sum())
    print("Control:", adata_obj.obs['control'].sum())
    print("Nan:", adata_obj.obs['nan'].sum())
    print("Normal pert.:", adata_obj.obs['gene_perturbation_mask'].sum())

    return None


"""
def create_perturb_matmask(
        adata_obj: AnnData,
) -> None:
    #Create a matrix in adata layers that masks the perturbed genes, using perturbed_gene_indices.
    mask = np.zeros(adata_obj.shape, dtype=bool)
    for i in range(adata_obj.n_obs):
        if adata_obj.obs['gene_perturbation_mask'].iloc[i]:
            mask[i, adata_obj.obs['perturbed_gene_indices'].iloc[i]] = True
    adata_obj.layers['perturbed_elem_mask'] = mask
    return None



def create_pert_indices_from_labels(
        adata_obj: AnnData
) -> None:
    # Create perturbed gene indices from the perturbation labels in a AnnData object.
    # store indices of perturbed genes - for backward compatibility 
    pert_indices_list = []
    for i in range(adata_obj.n_obs):
        if adata_obj.obs['gene_perturbation_mask'].iloc[i]:
            j = adata_obj.var_names.get_loc(
                adata_obj.obs['perturbation'].iloc[i])
            pert_indices_list.append(j)
    adata_obj.obs.insert(len(adata_obj.obs.columns), 'perturbed_gene_indices', pert_indices_list)
    return None
"""


def convert_onehot_to_indices(
        adata_obj: AnnData,
) -> None:
    """Convert the perturbation mask / onehot encoding to list of gene indices per sample for ease of use. 

    Args:
        adata_obj: AnnData object
    Returns:
        pert_indices_list: list of gene indices per sample
    """
    pert_indices_list = []
    for i, row in enumerate(adata_obj.layers['perturbed_elem_mask']):
        indices = tuple(np.sort(np.where(row)[0].tolist()))
        pert_indices_list.append(indices)
    return pert_indices_list


def convert_indices_to_onehot(
        adata_obj: AnnData,
        pert_indices_list: list
) -> None:
    """Convert the gene indices to a onehot encoding for the perturbation mask.

    Args:
        adata_obj: AnnData object
        pert_indices_list: list of gene indices per sample
    Returns:
        None
    """
    perturbed_elem_mask = np.zeros(adata_obj.shape, dtype=bool)
    for i, indices in enumerate(pert_indices_list):
        perturbed_elem_mask[i, indices] = True
    adata_obj.layers['perturbed_elem_mask'] = perturbed_elem_mask
    return None


def convert_perturb_labels_to_onehot(
        adata_obj: AnnData,
) -> None:
    """Convert the perturbation labels to indices an create a layer in the AnnData object (for backward compatibility).

    Args:
        adata_obj: AnnData object
    Returns:
        None
    """
    # store indices of perturbed genes
    pert_indices_list = []
    perturbed_elem_mask = np.zeros(adata_obj.shape, dtype=bool)
    for i in range(adata_obj.n_obs):
        if adata_obj.obs['gene_perturbation_mask'].iloc[i]:
            j = adata_obj.var_names.get_loc(
                adata_obj.obs['perturbation'].iloc[i])
            pert_indices_list.append([j])
            perturbed_elem_mask[i, j] = True
        else:
            pert_indices_list.append([])
    if 'perturbed_gene_indices' not in adata_obj.obs.columns:
        adata_obj.obs.insert(len(adata_obj.obs.columns),
                            'perturbed_gene_indices', pert_indices_list)
    adata_obj.layers['perturbed_elem_mask'] = perturbed_elem_mask

    return None


def gene_labels_2_index(
        adata_obj: AnnData,
        gene_labels: list
) -> list:
    """Get the indices of the genes in the AnnData object. > outdated, use onehot encoding instead

    Args:
        adata_obj: AnnData object
        gene_labels: list, gene labels
    Returns:
        gene_indices: list, indices of the genes in the AnnData object
    """
    # print warning that this is deprecated:
    print("Warning: gene_labels_2_index is deprecated, use onehot encoding instead")

    gene_indices = [adata_obj.var_names.get_loc(
        label) for label in gene_labels]
    return gene_indices


def get_perturb_labels(
        adata_obj: AnnData,
        filter_genes: bool = True,
        perturbation_entry='perturbation',
        non_targeting_label='non-targeting',
) -> None:
    """Get the perturbation labels from the AnnData object.

    Filter by multiplet, non-targeting and normal perturbations
    Store results inplace in the AnnData object observation annotation.

    !Update this along with data NoteBook to make this more general!

    Args:
        adata_obj: AnnData object
        filter_genes: bool, whether to filter the genes that are not in the gene
        perturbation_entry: str, name of the perturbation entry in the observation annotation
        non_targeting_label: str, label for the non-targeting perturbations
    Returns:
        None
    """
    # Get the perturbation labels and categorize them (depends on the dataset being used)
    perturb_labels = adata_obj.obs[perturbation_entry].astype(str).copy()

    perturb_labels_f = [label.split("_")[0] for label in perturb_labels]
    perturb_labels_simple = []
    for entry in perturb_labels_f:
        if ':' in entry:
            perturb_labels_simple.append('nan')
        elif entry == non_targeting_label:
            perturb_labels_simple.append('non-targeting')
        else:
            perturb_labels_simple.append(entry)

    adata_obj.obs['perturbation'] = perturb_labels_simple
    adata_obj.obs['perturbation'] = adata_obj.obs['perturbation'].astype(
        'category')

    # Create boolean masks for the perturbation labels
    create_perturb_bools(adata_obj, perturbation_entry)

    if filter_genes:
        gene_in_var = [
            gene in adata_obj.var_names for gene in adata_obj.obs['perturbation']]
        gene_pert_f = gene_in_var & adata_obj.obs['gene_perturbation_mask']
        print("Filtered", np.sum(adata_obj.obs['gene_perturbation_mask'])-np.sum(
            gene_pert_f), "un-identifiable perturbations: ",
            np.sum(gene_pert_f), "filtered perturbations")
        adata_obj.obs['gene_perturbation_mask'] = gene_pert_f

    adata_obj.var['gene_perturbed'] = adata_obj.var_names.isin(
        adata_obj.obs['perturbation'][adata_obj.obs['gene_perturbation_mask']])

    # Create a mask for the perturbed count values
    # create_perturb_matmask(adata_obj)

    return None


def scale_counts(
        adata_obj: AnnData,
        copy: bool = False,
        max_value: float = 10.0,
        verbose: bool = False
):
    """
    Scale the counts of the AnnData object to zero mean and unit variance per each gene.
    Also account for perturbed expression of genes:
        - perturbed counts will be left out when derivating the scaling parameters
        - will be included in the scaling

    -> Still needs to be verified how stable this is, for now rely on scanpy.pp.scale.

    Args:
        adata_obj: AnnData object
        copy: whether to return a copy of the AnnData object or to modify it in place
        max_value: maximum value for the scaled data
        verbose: whether to print the shapes of the arrays
    Returns:
        adata_obj: AnnData object, with the scaled data if copy is False
    """
    if copy:
        adata_obj = adata_obj.copy()

    # 1. Create a mask for the perturbed cases > outdated move somewhere else
    mask = np.zeros(adata_obj.shape, dtype=bool)
    for i in range(adata_obj.n_obs):
        if adata_obj.obs['gene_perturbation_mask'].iloc[i]:
            j = adata_obj.var_names.get_loc(
                adata_obj.obs['perturbation'].iloc[i])
            mask[i, j] = True
    mask = ~mask
    adata_obj.layers['perturbed_elem_mask'] = mask

    # 2. Calculate the scaling parameters
    if scipy.sparse.issparse(adata_obj.X):
        print(np.shape(adata_obj.X.toarray()))
        mean = np.mean(adata_obj.X.toarray(), axis=1, where=mask)
        std = np.std(adata_obj.X.toarray(), axis=1, ddof=1, where=mask)
    else:
        mean = np.mean(np.array(adata_obj.X), axis=1, where=mask)
        std = np.std(adata_obj.X, axis=1, ddof=1, where=mask)
    # is this a unbiased estimator? check later

    print("mean shape: ", np.shape(mean))
    print("std shape: ", np.shape(std))

    # reshape arrays to be able to broadcast
    if verbose:
        print("Mean:", mean.shape)
        print("Std:", std.shape)
        print("X:", adata_obj.X.shape)
    mean = np.repeat(np.array([mean]).T, adata_obj.shape[1], axis=1)
    std = np.repeat(np.array([std]).T, adata_obj.shape[1], axis=1)

    # 3. Apply the scaling
    adata_obj.X = (adata_obj.X - mean) / std

    # 4. Clip the values
    if max_value is not None:
        adata_obj.X = np.clip(adata_obj.X, -max_value, max_value)

    adata_obj.X = np.asarray(adata_obj.X)

    if copy:
        return adata_obj
    else:
        return None


def filter_gRNAs(
        adata_obj: AnnData,
        filter_label: str,
):
    """Filter genes that correspond to CRISPR intervention (not cell nucleus) if present in dataset.

    Args:
        adata_obj: AnnData object
        filter_label: label for non-genes to be filtered
    """
    gRNA_mask = [(filter_label not in name) for name in adata_obj.var_names]
    print("Filtered", len(adata_obj.var_names) -
          np.sum(gRNA_mask), "non-gene elements")
    adata_obj = adata_obj[:, gRNA_mask]


def filter_genes_HVG(
        adata_obj: AnnData,
        n_genes: int = None,
        prioritise_perturbed: bool = True,
        keep_all_perturbed: bool = False,
        prioritise_TFs: bool = False,
        flavour: str = 'seurat',
) -> None:
    """Get gene dispersions and filter them down to n_genes with regard to perturbation and TFs

    1. calculate gene dispersions
    2. sort genes by normalised dispersion
    3. filter genes:

    A: biased towards perturbed genes
        - first all perturbed genes
        - plus other hv genes to get n genes
    B: biased towards TFs
        - all (highly variable) transcription factors
        - plus other hv genes to get n genes

    Args:
        adata_obj: AnnData object
        n_genes: list, target number of genes
        prioritise_perturbed: bool, keep perturbed genes regardless of of variability
        keep_all_perturbed: bool, keep all perturbed genes (regardless of n_genes)
        prioritise_TFs: bool, whether to prioritise TFs (not implemented yet)
        flavour: str, flavour of the filtering method ('seurat' or 'scanpy')
    """

    # get dispersion for all genes:
    print("All genes before filtering: ", len(adata_obj.var_names))
    print("n_genes: ", n_genes)
    sc.pp.highly_variable_genes(adata_obj, flavor=flavour)
    dispersion = adata_obj.var['dispersions_norm']
    # rank genes by dispersion
    ordering = np.argsort(dispersion)[::-1]
    gene_names_ordered = adata_obj.var_names[ordering]

    if prioritise_perturbed:
        # 1. get the ordering of the perturbed genes         #recalculate the ordering
        perturbed_genes = adata_obj.var_names[adata_obj.var['gene_perturbed']]
        pert_dispersion = dispersion[adata_obj.var['gene_perturbed']]
        perturbed_ordering = np.argsort(pert_dispersion)[::-1]
        # recalculate the ordering

        # 2. order the perturbed genes by dispersion
        pert_genes_ordered = perturbed_genes[perturbed_ordering]

        # 2. remove perturbed genes from the ordering
        gene_names_ordered = gene_names_ordered.drop(perturbed_genes)

        # 3. add perturbed genes in order at the beginning (working? -ceck if right order)
        gene_names_ordered = pert_genes_ordered.append(gene_names_ordered)

    adata_obj.var['highly_variable'] = False
    if keep_all_perturbed:
        # set perturbed genes to highly variable
        adata_obj.var['highly_variable'][adata_obj.var['gene_perturbed']] = True
    elif prioritise_perturbed:
        # set the top n_genes to highly variable
        top_n_genes = gene_names_ordered[:n_genes]
        print("Top n genes: ", top_n_genes)
        print("Top n genes: ", len(top_n_genes))
        adata_obj.var.loc[top_n_genes, 'highly_variable'] = True
    else:
        # set the top n_genes to highly variable
        adata_obj.var.loc[gene_names_ordered[:n_genes],
                          'highly_variable'] = True

    if prioritise_TFs:
        raise NotImplementedError("Not implemented yet")

    print("All genes after filtering: ", np.sum(
        adata_obj.var['highly_variable']))



def filter_genes(adata, option=4, N_genes = 40):
    """
    Filter genes of an AnnData object based on different criteria.
    1. Top N highly variable (HV) genes
    2. All perturbed genes
    3. Top N target genes based on DE-score (using scanpy t-test)
    4. All target genes that show DE (p-value < 0.05)
    5. Target genes plus all genes that show DE (p-value < 0.05)

    """
    # Preparation: run scanpy DE tests
    perturbed_genes = adata.var_names[adata.var['gene_perturbed']]
    print(f'Found {len(perturbed_genes)} target genes in the data.')

    # Compute DE-score for each TF and each perturbation
    verbose = True

    # use perturbation as group label for DE-test
    adata.obs['perturbation_group'] = adata.obs['perturbation']
    adata.obs['perturbation_group'] = adata.obs['perturbation_group'].astype(
        'category')
    key = 'rank_genes_perturbations'

    # Do DE-estimation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(
            adata, groupby='perturbation_group', method='t-test', key_added=key, reference='non-targeting')
        reference = str(adata.uns[key]["params"]["reference"])
    group_names = adata.uns[key]["names"].dtype.names
    if verbose:
        print("perturbed_genes: ", perturbed_genes)
        print('reference:', reference)
        print('group_names:', group_names)

    
    # extract all DE scores (here: p_values)
    verbose=True
    pvalues = np.zeros((adata.shape[1], len(perturbed_genes)))
    perturbation_scores = pd.DataFrame()
    for i,perturbed_gene in tqdm(enumerate(perturbed_genes)):
        # get the DE genes for the perturbation
        dataframe = sc.get.rank_genes_groups_df(
            adata, group=perturbed_gene, key=key)
        # get adjusted p-values of all genes, and filter for significant genes
        pval_column = 'pvals_adj'
        pvals = dataframe[pval_column]
        # add to the p-values np.array
        pvalues[:, i] = pvals.values
        # extract the row of the target gene
        target_gene_row = dataframe[dataframe['names'] == perturbed_gene]
        perturbation_scores = pd.concat(
            [perturbation_scores, target_gene_row], axis=0)
        
    # select genes to keep depending on the selected criteria
    if option == 1:
        # 2. Top N highly variable (HV) genes
        sc.pp.highly_variable_genes(adata, n_top_genes=N_genes)
        genes_to_use = adata.var_names[adata.var['highly_variable']]
        print(f'Found {len(genes_to_use)} most highly variable genes.')
    elif option == 2:
        # 3. all perturbed genes
        genes_to_use = perturbed_genes
    elif option == 3:
        # 4. Top N target genes based on DE-score
        column = 'pvals_adj'
        perturbation_scores = perturbation_scores.sort_values(by=column, ascending=True)
        genes_sorted = perturbation_scores['names'].to_list()
        genes_to_use = genes_sorted[:N_genes]
        print(f'Found {len(genes_to_use)} target genes based on DE-score.')
    elif option == 4:
        # 5. target genes that show DE
        perturbation_scores_filtered = perturbation_scores[perturbation_scores['pvals_adj'] < 0.05]
        genes_to_use = perturbation_scores_filtered['names'].to_list()
        print(f'Found {len(genes_to_use)} target genes that show DE in the data.')
    else:
        # 6. target genes plus all genes that show DE
        # 1. Get all DE genes: get the p-values for each perturbation, store in a matrix
        verbose=True
        print(f'Found {pvalues.shape[0]} genes with p-values for {pvalues.shape[1]} perturbations.')

        # 2. perform multiple testing adjustment
        # Flatten the p-value matrix
        pvals_flat = pvalues.flatten()
        print("rejected(naive): ",np.sum(pvals_flat<0.05))
        # Apply FDR-BH correction
        rejected, pvals_corrected, _, _ = multipletests(pvals_flat, alpha=0.1, method='fdr_bh')
        # Reshape results to original shape
        pvals_corrected_2d = pvals_corrected.reshape(pvalues.shape)

        # get significant genes by pooling across perturbations
        significant_DE_genes_adj = np.any(pvals_corrected_2d < 0.05, axis=1)
        significant_DE_genes = np.any(pvalues < 0.05, axis=1)

        print(f'Found {np.sum(significant_DE_genes)} genes with p-values < 0.05.')
        print(f'Found {np.sum(significant_DE_genes_adj)} genes with adjusted p-values < 0.05.')

        # combine significant genes with 
        genes_to_use = list(set(perturbed_genes + adata.var_names[significant_DE_genes_adj]))
        print(f'Found {len(genes_to_use)} genes that are target genes or show DE.')
    return genes_to_use