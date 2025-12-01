import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData


def plot_perturb_vs_non(
        adata_obj: AnnData,
        all_non_pert: bool = True,
        non_pert_pert: bool = True,
        pert_pert: bool = True,
        log: bool = True,
        xlim: tuple | None = None,
        ylim: tuple | None = None,
        filename: str = None,
        title: str = "Perturbed vs Non-Perturbed Gene Counts"
):
    """Plot the historgram of pertubed gene counts vs non-perturbed gene counts.

    Args:
        adata_obj: AnnData object
        all_non_pert: bool, whether to plot all non-perturbed genes
        non_pert_pert: bool, whether to plot non-perturbed counts for perturbed genes
        pert_pert: bool, whether to plot perturbed genes
        xlim: tuple, x-axis limits
        ylim: tuple, y-axis limits
        filename: str, filename for saving the plot
        title: str, title of the plot
    """

    numpy_counts = adata_obj.X
    # check data type and convert to numpy array if necessary
    if not isinstance(numpy_counts, np.ndarray):
        numpy_counts = numpy_counts.toarray()

    total_entries = numpy_counts.size
    numpy_counts_1d = numpy_counts.flatten()
    # take 1st element if the array is 2d
    if len(numpy_counts_1d) != total_entries:
        numpy_counts_1d = numpy_counts_1d[0]
    mask_1d = adata_obj.layers['perturbed_elem_mask'].flatten()
    perturbed_counts = numpy_counts_1d[~mask_1d]
    all_non_perturbed_counts = numpy_counts_1d[mask_1d]

    # non perturbed counts for only pertubed genes:
    numpy_counts_pert_genes = numpy_counts[:, adata_obj.var['gene_perturbed']]
    mask_1d_pert_genes = adata_obj.layers['perturbed_elem_mask'][:,
                                                                 adata_obj.var['gene_perturbed']].flatten()
    non_perturbed_counts_pert_genes = numpy_counts_pert_genes.flatten()[
        mask_1d_pert_genes]

    # 2. plot histogram
    if all_non_pert:
        plt.hist(all_non_perturbed_counts, bins=100, alpha=0.5,
                 label='non-perturbed', density=True)
    if non_pert_pert:
        plt.hist(non_perturbed_counts_pert_genes, bins=100, alpha=0.5,
                 label='unperturbed target genes', density=True)
    if pert_pert:
        plt.hist(perturbed_counts, bins=100, alpha=0.5,
                 label='perturbed target genes', density=True)
    plt.legend(loc='upper right')
    plt.title(title)

    if log:
        plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_target_downreg(adata_obj: AnnData, filename: str = None, non_targeting_only=False, n_genes=10, cut_dropouts=False):
    """Plot the downregulation of target genes by comparing the expression in perturbed and non-perturbed cells.

    Plot a sns violin plot of the expression of target genes in perturbed and non-perturbed cells.

    Args:
        adata_obj: AnnData object
        filename: str, filename for saving the plot
        non_targeting_only: bool, whether to only use non-targeting cells for comparison
    """
    sns.set(style="whitegrid")
    sns.set(rc={'figure.figsize': (20, 10)})
    perturbed_genes = adata_obj.var_names[adata_obj.var['gene_perturbed']]
    df = pd.DataFrame(columns=['Expression', 'Perturbed', 'Gene'])

    # transform sparse array to dense array
    if not isinstance(adata_obj.X, np.ndarray):
        counts = adata_obj.X.toarray()
    else:
        counts = adata_obj.X

    for gene in perturbed_genes[:n_genes]:
        perturbed_gene_mask = adata_obj.obs['perturbation'] == gene
        if non_targeting_only:
            non_perturbed_gene_mask = adata_obj.obs['non-targeting']
        else:
            non_perturbed_gene_mask = ~perturbed_gene_mask
        perturbed_gene_expression = counts[perturbed_gene_mask, adata_obj.var_names.get_loc(
            gene)]
        non_perturbed_gene_expression = counts[non_perturbed_gene_mask, adata_obj.var_names.get_loc(
            gene)]

        for expr_val in perturbed_gene_expression:
            df.loc[len(df)] = [expr_val, True, gene]
        for expr_val in non_perturbed_gene_expression:
            df.loc[len(df)] = [expr_val, False, gene]

    if cut_dropouts:
        df = df[df['Expression'] > 0]
    # sns.violinplot(data=df, x='Perturbed', y='Expression')
    sns.violinplot(data=df, x="Gene", y="Expression",
                   hue="Perturbed", split=True, gap=.1, cut=0, bw_adjust=.5, inner="quart", density_norm='width')
    plt.xlabel(None)
    plt.ylabel("Expression", fontsize=15)
    plt.title(f"Downregulation of target genes", fontsize=20)
    plt.xticks(rotation=0, fontsize=15)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_perturbation_expression(adata_obj: AnnData, filename: str = None, non_targeting_only=False, n_genes=10, pert_gene=None, cut_dropouts=False):
    """Plot the expression levels genes under a specific perturbation by comparing the expression in perturbed and non-perturbed cells.

    Plot a sns violin plot of the expression of target genes in perturbed and non-perturbed cells.

    Args:
        adata_obj: AnnData object
        filename: str, filename for saving the plot
        non_targeting_only: bool, whether to only use non-targeting cells for comparison
    """
    #sns.set(style="whitegrid")
    #sns.set(rc={'figure.figsize': (20, 10)})
    df = pd.DataFrame(columns=['Expression', 'Perturbed', 'Gene'])

    # transform sparse array to dense array
    if not isinstance(adata_obj.X, np.ndarray):
        counts = adata_obj.X.toarray()
    else:
        counts = adata_obj.X

    for gene in adata_obj.var_names[:n_genes]:
        perturbed_gene_mask = adata_obj.obs['perturbation'] == pert_gene
        if non_targeting_only:
            non_perturbed_gene_mask = adata_obj.obs['non-targeting']
        else:
            non_perturbed_gene_mask = ~perturbed_gene_mask
        perturbed_gene_expression = counts[perturbed_gene_mask, adata_obj.var_names.get_loc(
            gene)]
        non_perturbed_gene_expression = counts[non_perturbed_gene_mask, adata_obj.var_names.get_loc(
            gene)]

        for expr_val in perturbed_gene_expression:
            df.loc[len(df)] = [expr_val, True, gene]
        for expr_val in non_perturbed_gene_expression:
            df.loc[len(df)] = [expr_val, False, gene]

    if cut_dropouts:
        df = df[df['Expression'] > 0]
    # sns.violinplot(data=df, x='Perturbed', y='Expression')
    sns.violinplot(data=df, x="Gene", y="Expression",
                   hue="Perturbed", split=True, gap=.1, cut=0, bw_adjust=.5, inner="quart", density_norm='width')
    plt.xlabel(None)
    plt.ylabel("Expression", fontsize=15)
    plt.title(f"Expression unter perturbation of {pert_gene}", fontsize=20)
    plt.xticks(rotation=0, fontsize=15)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def print_expression_mean_std(adata_obj: AnnData):
    """Print the mean and standard deviation of the expression of perturbed and non-perturbed genes.

    Args:
        adata_obj: AnnData object
    """
    # Check the expression of downregulated genes
    perturbed_gene_expression = np.array([])
    for observation in range(adata_obj.shape[0]):
        if adata_obj.obs['gene_perturbation_mask'].iloc[observation]:
            perturbed_gene = adata_obj.obs['perturbation'].iloc[observation]
            if perturbed_gene in adata_obj.var_names:
                perturbed_gene_expression = np.append(
                    perturbed_gene_expression, adata_obj.X[observation, adata_obj.var_names.get_loc(perturbed_gene)])

    print("")
    print("Perturbed Gene Expression:")
    print("Mean: ", np.mean(perturbed_gene_expression))
    print("Std: ",  np.std(perturbed_gene_expression))
    print("Min: ",  np.min(perturbed_gene_expression))
    print("Max: ",  np.max(perturbed_gene_expression))
    print("95% percentile: ", np.percentile(perturbed_gene_expression, 5),
          " - ", np.percentile(perturbed_gene_expression, 95))

    # Check the expression of non-targeting genes
    non_target_gene_expression = np.array([])
    for observation in range(adata_obj.shape[0]):
        if adata_obj.obs['non-targeting'].iloc[observation]:
            non_target_gene_expression = np.append(
                non_target_gene_expression, adata_obj.X[observation, np.random.randint(adata_obj.shape[1])])

    print("")
    print("Non-Target Gene Expression:")
    print("Mean: ", np.mean(non_target_gene_expression))
    print("Std: ",  np.std(non_target_gene_expression))
    print("Min: ",  np.min(non_target_gene_expression))
    print("Max: ",  np.max(non_target_gene_expression))
    print("95% percentile: ", np.percentile(non_target_gene_expression, 5),
          " - ", np.percentile(non_target_gene_expression, 95))
