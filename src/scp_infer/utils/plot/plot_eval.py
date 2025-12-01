import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scp_infer.eval.eval_manager import EvalManager
from scp_infer.eval.graph_eval import edge_density


# Define Plotting Functions

def plot_results_errorbar(
        xval_arrs: list,
        yval_arrs: list,
        labels: list,
        title: str,
        x_label: str,
        y_label: str,
        filename: str | None = None
        ):
    """Plot the results with error bars"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for xvals, yvals, label in zip(xval_arrs, yval_arrs, labels):
        ax.errorbar(np.mean(xvals), np.mean(yvals), xerr=np.std(xvals), yerr=np.std(
            yvals), marker='o', ms=6, ecolor='gray', capsize=3,  label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(linestyle='dotted')
    ax.legend()
    plt.title(title)

    if filename is not None:
        plt.savefig('./plots/eval/ebar_'+filename)
    plt.show()


def plot_results_scatter(
        xval_arrs: list,
        yval_arrs: list,
        labels: list,
        title: str, 
        x_label: str,
        y_label: str,
        filename: str | None = None
        ):
    """Plot the results as scatter plot"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for xvals, yvals, label in zip(xval_arrs, yval_arrs, labels):
        ax.scatter(xvals, yvals, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(linestyle='dotted')
    ax.legend()
    plt.title(title)

    if filename is not None:
        plt.savefig('./plots/eval/'+filename)
    plt.show()
    return fig, ax


def plot_de_hierarchy(
        dataset_name: str,
        model_names: list = ['GIES', 'GRNBoost2'],
        cutoffs: list = [None,15],
        split_version: str = 'shuffled',
        folder: str = 'data_out',
        verbose: bool = False
        ):
    """Plot the DE hierarchy for a given dataset

    Args:
        dataset_name: str, name of the dataset
        model_names: list, names of the models to compare
        cutoffs: list, cutoffs for weighted adjacency matrix
        split_version: str, version of the split
    """
    data_file = "../data/edited/"+dataset_name+".h5ad"
    adata = sc.read_h5ad(data_file)
    print('dataset: ', dataset_name)
    data_mng = ScpiDataManager(adata, dataset_name, output_folder='../data/'+folder)
    eval_mng = EvalManager(data_mng, replace=False)
    # split_labels, _ = data_mng.get_train_test_splits()
    

    # A. Load Data
    de_upstream = {}
    de_downstream = {}
    ctrl_upstr_arr = []
    ctrl_downstr_arr = []
    density_dic = {}
    for model_name, cutoff in zip(model_names, cutoffs):
        print(model_name)
        split_labels, adj_matrices = \
            eval_mng.load_inference_results(
                split_version, model_name)
        n_upstr_arr = []
        n_downstr_arr = []
        density_arr = []
        for label, adj_mat in zip(split_labels, adj_matrices):
            print("label: ", label)
            # 1. Load values from evaluation results
            n_upstr = eval_mng.load_evaluation_results(
                split_version=split_version,
                model_name=model_name,
                metric='DE_n_upstream',
                split_label=label
            )['value'].to_numpy()
            n_downstr = eval_mng.load_evaluation_results(
                split_version=split_version,
                model_name=model_name,
                metric='DE_n_downstream',
                split_label=label
            )['value'].to_numpy()
            # Load Control Values
            ctrl_upstream = eval_mng.load_evaluation_results(
                split_version=split_version,
                model_name=model_name,
                metric='DE_n_upstream',
                split_label_ew=label,
                split_label_sw='negative_control'
            )['value'].to_numpy()
            ctrl_downstream = eval_mng.load_evaluation_results(
                split_version=split_version,
                model_name=model_name,
                metric='DE_n_downstream',
                split_label_ew=label,
                split_label_sw='negative_control'
            )['value'].to_numpy()
            # 2. store the sum over the n_de_genes for each perturbed gene (=tag)
            n_upstr_arr.append(np.mean(n_upstr))
            n_downstr_arr.append(np.mean(n_downstr))
            ctrl_upstr_arr.append(np.mean(ctrl_upstream))
            ctrl_downstr_arr.append(np.mean(ctrl_downstream))
            if cutoff is not None:
                adj_mat = (adj_mat > cutoff).astype(int)
            density = edge_density(adj_mat)
            density_arr.append(density)
            if verbose:
                print("n_upstr: ", n_upstr)
                print("n_downstr: ", n_downstr)
                print("ctrl_upstream: ", ctrl_upstream)
                print("ctrl_downstream: ", ctrl_downstream)
        if verbose:
            print("n_upstr_arr: ", n_upstr_arr)
            print("n_downstr_arr: ", n_downstr_arr)
            print("ctrl_upstr_arr: ", ctrl_upstr_arr)
            print("ctrl_downstr_arr: ", ctrl_downstr_arr)
        de_upstream[model_name] = np.array(n_upstr_arr)
        de_downstream[model_name] = np.array(n_downstr_arr)
        density_dic[model_name] = np.array(density_arr)
    de_upstream['Control'] = np.array(ctrl_upstr_arr)
    de_downstream['Control'] = np.array(ctrl_downstr_arr)
    density_dic['Control'] = np.array([])
    for model in model_names:
        density_dic['Control'] = np.append(density_dic['Control'], density_dic[model])

    if verbose:
        print("de_upstream: ", de_upstream)
        print("de_downstream: ", de_downstream)
    # B. Make Plot
    de_upstream_arr = [de_upstream[model_name]
                       for model_name in ['Control']+model_names]
    de_downstream_arr = [de_downstream[model_name]
                         for model_name in ['Control']+model_names]
    density_arr = [density_dic[model_name]
                     for model_name in ['Control']+model_names]

    xvals = []
    yvals = []
    for upstr, dstr, den in zip(de_upstream_arr, de_downstream_arr, density_arr):
        xvals.append(den)
        #xvals.append(upstr + dstr)
        yvals.append(dstr/upstr)

    plot_results_scatter(
        xvals, yvals,
        ['Control']+model_names,
        dataset_name[17:] +
        " - Upstream vs Downstream DE Genes",
        #"Mean Related DE Genes (per Perturbation): $N_{downstr.}+N_{upstr.}$",
        "Graph Edge Density",
        r'Ratio: $N_{downstr.}$/$N_{upstr.}$',
        filename=dataset_name+'de_hierarchy.png'
    )