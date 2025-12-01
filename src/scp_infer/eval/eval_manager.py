"""
Class to use for evaluation of model prediction and management of results
"""

import os
import pandas as pd
import numpy as np

from scp_infer.eval.stat_eval import evaluate_wasserstein, evaluate_f_o_r, de_graph_hierarchy


class EvalManager():
    """
    Evaluation utilities for a given dataset

    A:
        - load inference results & Validation data
        - Evaluate model predictions: each adj-matrix(test dat.) x each metric (+ negative control)
        - Save evaluation results: pd.DataFrame

    B:
        - Load evaluation results
        - Compare & Plot results

    To-Do:
        - Add column for real vs. control

    Args:
        datamanager: Data manager object (ScpiDataManager)
        replace: Whether to replace / overwrite the existing evaluation results

    Attributes:
        datamanager: Data manager object (ScpiDataManager)
        adata_obj: Annotated expression data object from scanpy (outdated)
        output_folder: Folder to save the files in
        dataset_name: Name of the dataset - where files will be stored
        dataframe: pd.DataFrame to store the evaluation results
        dataframe_cols: Columns of the evaluation results dataframe
        csv_file: Path to the csv file to save the evaluation results
    """

    def save_evaluation_results(self):
        """Save evaluation results in the appropriate folder"""
        self.dataframe.to_csv(self.csv_file)

    def __init__(self, datamanager: object, replace: bool = False):
        """Initialize the manager

        Currently wraps around the datamanager object
        -> maybe make this more elegant?
        """
        self.datamanager = datamanager
        #self.adata_obj = datamanager.adata_obj
        self.output_folder = datamanager.output_folder
        self.dataset_name = datamanager.dataset_name
        self.dataframe_cols = [
            "split-version", "split-label", "model-name", "metric", "tag", "value"]
        self.csv_file = os.path.join(
            self.output_folder, self.dataset_name, "evaluation_results.csv")
        # Load the Dataframe:
        if not replace and os.path.exists(self.csv_file):
            print("Loading evaluation results from file")
            self.dataframe = pd.read_csv(self.csv_file, index_col=0)
        else:
            print("Creating new evaluation results dataframe")
            self.dataframe = pd.DataFrame(columns=self.dataframe_cols)

    def append_eval_result(self, results: list) -> None:
        """Append new evaluation results to the dataframe.

        If an entry with the same labels but different values exists, it will be overwritten.

        Args:
            results: List of results to append, must have the same length as the dataframe columns
        """
        if np.shape(results)[1] != len(self.dataframe_cols):
            raise ValueError("Results do not match the dataframe columns")
        df_new = pd.DataFrame(results, columns=self.dataframe_cols)
        df_old = self.dataframe

        
        # fix value error for None entries of 'tag' column
        df_new['tag'] = df_new['tag'].astype(str)
        df_old['tag'] = df_old['tag'].astype(str)

        # merge dataframes, old values are placed in columns with suffix _old
        df_merge = df_old.merge(df_new, on=["split-version", "split-label", "model-name",
                                "metric", "tag"], validate='1:1', how='outer', suffixes=('_old', None))
        new_cols = df_merge.columns[df_merge.columns.str.endswith('_old')]
        orig_cols = new_cols.str[:-4]
        # dictionary for renaming columns
        d = dict(zip(new_cols, orig_cols))
        # filter columns and replace NaNs by values from extra column
        df_merge[orig_cols] = df_merge[orig_cols].combine_first(
            df_merge[new_cols].rename(columns=d))
        # remove extra columns
        df_merge = df_merge.drop(new_cols, axis=1)
        self.dataframe = df_merge
        """
        self.dataframe = pd.concat([df_old, df_new], ignore_index=True)
        """

    def load_inference_results(
            self,
            split_version: str = 'shuffled',
            model_name: str = None,
            split_labels: list = None,
    ) -> tuple:
        """Load inference results for a given split_version

        Args:
            split_version: Version of the split
            model_name: Name of the model
            split_labels: List of split labels

        Returns:
            tuple: tuple containing:

                - split_labels: List of split labels
                - adj_matrices: List of adjacency matrices
        """
        return self.datamanager.load_inference_results(split_version, model_name, split_labels)

    def create_control(self, adj_matrices):
        """Create a negative control for the adjacency matrices: shuffle the features"""
        ctrl_adj_matrices = []
        for adj_matrix in adj_matrices:
            rand_perm = np.random.permutation(adj_matrix.shape[0])
            ctrl_adj_matrices.append(adj_matrix[rand_perm, :][:, rand_perm])
        return ctrl_adj_matrices

    def evaluate_model(
            self,
            split_version: str,
            model_name: str,
            metric: str,
            adj_cutoff: float | None = None,
            random_control: bool = False,
            test_data_only: bool = True

    ):
        """Evaluate model predictions: each adj-matrix x each metric (+ negative control)

        Saves the results in the evaluation dataframe

        Args:
            split_version: Version of the split
            model_name: Name of the model to evaluate
            metric: Name of the metric to evaluate
            adj_cutoff: Cutoff value for the adjacency matrix
            random_control: Whether to create a negative control
            test_data_only: Whether to evaluate only the test data
        """
        # 1. Load the test data and inference results:
        split_labels, split_datasets = \
            self.datamanager.get_train_test_splits(
                split_version, split_labels=None, backed=False)
        split_labels, adj_matrices = \
            self.load_inference_results(
                split_version, model_name, split_labels=None)

        if adj_cutoff is not None:
            # Binarize the adjacency matrices
            adj_matrices = [(adj_matrix > adj_cutoff).astype(int)
                            for adj_matrix in adj_matrices]

        if random_control:
            # For each adj x dataset pair, create a shuffled control
            ctrl_adj_matrices = self.create_control(adj_matrices)
            adj_matrices = adj_matrices + ctrl_adj_matrices
            split_labels = split_labels + \
                ["negative_control_"+split_label for split_label in split_labels]
            split_datasets = split_datasets + split_datasets

        # 2. Filter the AnnData object for the test data:
        if test_data_only:
            for split_data in split_datasets:
                split_data = split_data[split_data.obs["set"] == "test"]

        # 3. Evaluate the model:
        for split_label, split_data, adj_matrix in zip(split_labels, split_datasets, adj_matrices):
            # Temporarily store results in a list
            labels = []
            tags = []
            values = []
            if metric == "wasserstein":
                # Evaluate the wasserstein distance
                tp, fp, wasserstein_distances = evaluate_wasserstein(
                    split_data, adj_matrix, p_value_threshold=0.05)
                mean_wasserstein = np.mean(wasserstein_distances)
                labels += ["wasserstein", "wasserstein_TP", "wasserstein_FP"]
                tags += [None, None, None]
                values += [mean_wasserstein, tp, fp]
            elif metric == "false_omission_ratio":
                # Evaluate the false omission ratio
                f_o_r, neg_mean_wasserstein = evaluate_f_o_r(
                    split_data, adj_matrix, p_value_threshold=0.05)
                labels += ["false_omission_ratio", "neg_mean_wasserstein"]
                tags += [None, None]
                values += [f_o_r, neg_mean_wasserstein]
            elif metric == "de_graph_hierarchy":
                # Evaluate the de-graph hierarchy
                tag_arr, upstr_arr, downstr_arr, unrel_arr, cyclic_arr = de_graph_hierarchy(
                    split_data, adj_matrix, verbose=False)
                for tag, n_upstr, n_downstr, n_unrel, n_cyclic in zip(tag_arr, upstr_arr, downstr_arr, unrel_arr, cyclic_arr):
                    labels += ["DE_n_upstream",
                               "DE_n_downstream", "DE_n_unrelated", "DE_n_cyclic"]
                    tags += [tag, tag, tag, tag]
                    values += [n_upstr, n_downstr, n_unrel, n_cyclic]
            else:
                raise ValueError("Metric not implemented")
            # Save the results in the dataframe
            for label, tag, value in zip(labels, tags, values):
                self.append_eval_result(
                    [[split_version, split_label, model_name, label, tag, value]])
        # Save the results
        self.save_evaluation_results()

    def evaluate_gene_holdout(
            self,
            model_name: str,
            metric: str,
            adj_cutoff: float | None = None,
            random_control: bool = False,
            test_data_only: bool = True
    ):
        """Evaluate model predictions for a gene holdout experiment

        (compare when the gene is held out)

        Args:
            model_name: Name of the model to evaluate
            metric: Name of the metric to evaluate
            adj_cutoff: Cutoff value for the adjacency matrix
            random_control: Whether to create a negative control
            test_data_only: Whether to evaluate only the test data
        """
        # 1.a) Load the gene-holdout/train data and inference results:
        split_labels, split_datasets = \
            self.datamanager.get_train_test_splits(
                "gene-holdout", split_labels=None)
        split_labels, adj_matrices = \
            self.load_inference_results(
                "gene-holdout", model_name, split_labels=None)

        # 1.b) Load the all-train data and inference results:
        split_data_all = \
            self.datamanager.get_train_test_splits(
                "all_train", split_labels=None)[1][0]
        adj_matrix_all = \
            self.load_inference_results(
                "all_train", model_name, ["all_train"])[1][0]

        if adj_cutoff is not None:
            # Binarize the adjacency matrices
            adj_matrices = [(adj_matrix > adj_cutoff).astype(int)
                            for adj_matrix in adj_matrices]
            adj_matrix_all = (adj_matrix_all > adj_cutoff).astype(int)

        if random_control:
            # For each adj x dataset pair, create a shuffled control
            ctrl_adj_matrices = self.create_control(adj_matrices)
            adj_matrices = adj_matrices + ctrl_adj_matrices
            split_labels = split_labels + \
                ["negative_control_"+split_label for split_label in split_labels]
            split_datasets = split_datasets + split_datasets

        # 2. get the perturbed / holdout_genes from the split labels (gene_...)
        held_out_genes = [label[5:] for label in split_labels]

        # 3. Evaluate the model:
        for split_label, ho_gene, adj_matrix in zip(split_labels, held_out_genes, adj_matrices):
            # Temporarily store results in a list
            labels = []
            tags = []
            values = []
            if metric == "de_graph_hierarchy":
                # Evaluate the de-graph hierarchy
                # Store only the results for the hold_out_gene
                tag_arr, upstr_arr, downstr_arr, unrel_arr = de_graph_hierarchy(
                    split_data_all, adj_matrix)
                hold_out_idx = tag_arr.index(ho_gene)
                labels += ["holdout_DE_n_upstream",
                           "holdout_DE_n_downstream", "holdout_DE_n_unrelated"]
                tags += [ho_gene, ho_gene, ho_gene]
                values += [upstr_arr[hold_out_idx],
                           downstr_arr[hold_out_idx], unrel_arr[hold_out_idx]]
            else:
                raise ValueError("Metric not implemented")
            # Save the results in the dataframe
            for label, tag, value in zip(labels, tags, values):
                self.append_eval_result(
                    [["gene-holdout", split_label, model_name, label, tag, value]])

        # Evaluate the model for the all-train data
        labels = []
        tags = []
        values = []
        if metric == "de_graph_hierarchy":
            # Evaluate the de-graph hierarchy
            tag_arr, upstr_arr, downstr_arr, unrel_arr = de_graph_hierarchy(
                split_data_all, adj_matrix_all)
            for tag, n_upstr, n_downstr, n_unrel in zip(tag_arr, upstr_arr, downstr_arr, unrel_arr):
                labels += ["DE_n_upstream",
                           "DE_n_downstream", "DE_n_unrelated"]
                tags += [tag, tag, tag]
                values += [n_upstr, n_downstr, n_unrel]

        for label, tag, value in zip(labels, tags, values):
            self.append_eval_result(
                [["all_train", "all_train", model_name, label, tag, value]])
        # Save the results
        self.save_evaluation_results()

    def load_evaluation_results(
            self,
            split_version: str | None = None,
            split_label: str | None = None,
            model_name: str | None = None,
            metric: str | None = None,
            tag: str | None = None,
            split_label_sw: str | None = None,
            split_label_ew: str | None = None,
            control: bool = False
    ) -> pd.DataFrame:
        """Load evaluation results with filtering

        If no filter is given, load all results

        Args:
            split_version: Version of the split
            split_label: Name of the split
            model_name: Name of the model
            metric: Name of the metric
            tag: Tag (e.g. gene)
            split_label_sw: Startswith filter for split_label
            split_label_ew: Endswith filter for split_label
            control: Whether to load the control results

        Returns:
            pd.DataFrame: Filtered evaluation results
        """
        df = self.dataframe
        if split_version is not None:
            df = df.loc[df["split-version"] == split_version]
        if split_label is not None:
            df = df.loc[df["split-label"] == split_label]
        if split_label_sw is not None:
            df = df.loc[[label.startswith(split_label_sw)
                         for label in df['split-label']]]
        if split_label_ew is not None:
            df = df.loc[[label.endswith(split_label_ew)
                         for label in df['split-label']]]
        if model_name is not None:
            df = df.loc[df["model-name"] == model_name]
        if metric is not None:
            df = df.loc[df["metric"] == metric]
        if tag is not None:
            df = df.loc[df["tag"] == tag]
        if control:
            df = df.loc[df["split-label"].str.contains("negative_control")]
        return df
