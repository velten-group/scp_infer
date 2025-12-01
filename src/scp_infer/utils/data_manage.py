"""
Data management utilities to accomodate for:
different datasets, holdout strategies, splits and inference methods.
"""

import os
import scanpy as sc
import numpy as np
from anndata import AnnData

import scp_infer as scpi
from scp_infer.utils import shuffled_split, shuffled_split_proportion, gene_holdout, intervention_proportion_split, gene_holdout_stratified


class ScpiDataManager():
    """Data management utilities for (a) given dataset(s).

    This class can:
        - create multiple train test splits
        - load and save adata with split information:
            split_version, split_label
            -> for inference and evaluation
        - load data for inference
        - store inference results
        - load data for evaluation

    Args:
        adata_obj: Annotated expression data object from scanpy
            should be fully preprocessed and ready for inference
        dataset_name: Name of the dataset - where files will be stored
        output_folder: Folder to save the files in

    Attributes:
        adata_obj: Annotated expression data object from scanpy
            should be fully preprocessed and ready for inference
        dataset_name: Name of the dataset - where files will be stored
        output_folder: Folder to save the files in

    """

    def __init__(self, adata_obj: AnnData, dataset_name: str, output_folder: str = "../data/data_out"):
        """Initialize the data manager.

        Args:
            adata_obj (AnnData): Annotated expression data object from scanpy. Should be fully preprocessed and ready for inference.
            dataset_name (str): Name of the dataset - where files will be stored.
            output_folder (str, optional): Base Location to save the files in. Defaults to "../data/data_out".
        """
        self.adata_obj = adata_obj
        self.output_folder = output_folder
        self.dataset_name = dataset_name

    def split_ver_folder(self, split_version: str):
        """Get the folder for a given split version for this dataset"""
        split_ver_folder = os.path.join(
            self.output_folder, self.dataset_name, split_version)
        split_labels = os.listdir(split_ver_folder)
        split_labels.sort()
        return split_ver_folder, split_labels

    def save_split(self,
                   adata: AnnData,
                   split_version: str,
                   split_label: str
                   ) -> None:
        """Save the data split in the appropriate folder

        saves the annotated adata object in folder hierarchy:
        - dataset
            - split-version 1
                - split_label 1
                - split_label 2
                - ...

        Args:
            adata: Annotated expression data object from scanpy
                should be fully preprocessed and ready for inference
            split_version: Version of the split
            split_label: Label for the split
            output_folder:Folder to save the files in
        """
        save_folder = os.path.join(self.output_folder,
                                   self.dataset_name, split_version, split_label)
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"{split_label}.h5ad")
        sc.write(save_file, adata)
        return None

    def store_train_test_split(
            self,
            split_version: str = "shuffled",
            test_size: float = 0.2,
            n_splits: int = 1,
            gene_holdout: str = None
    ) -> None:
        """Create train test splits for a given split_version

        Args:
            split_version: Version of the split
                shuffled: random split
                gene-holdout: holdout perturbation on specific genes
                total-intervention: holdout intervention by proportion on entire dataset
            test_size: Fraction of the test set
            n_splits: Number of splits to create
        """
        adata = self.adata_obj
        # Create a folder for the dataset
        dataset_name = self.dataset_name
        os.makedirs(os.path.join(self.output_folder,
                    dataset_name), exist_ok=True)

        # Create a folder for the split version
        os.makedirs(os.path.join(self.output_folder,
                    dataset_name, split_version), exist_ok=True)

        # Create train test splits
        if split_version == "shuffled-regimes":
            raise NotImplementedError("Not implemented yet")
            for i in range(n_splits):
                # label with test ratio and index
                split_label = f"%.f_test_split_{i}" % (test_size*100)
                shuffled_split(adata, test_frac=test_size, seed=i)
                self.save_split(adata, split_version, split_label)
        elif split_version == "shuffled":
            for i in range(n_splits):
                # label with test ratio and index
                split_label = f"%.f_test_split_{i}" % (test_size*100)
                shuffled_split_proportion(adata, test_frac=test_size, seed=i)
                self.save_split(adata, split_version, split_label)
        elif split_version == "intervention-proportion":
            for i in range(n_splits):
                split_label = f"%.f_interv_split_{i}" % (test_size*100)
                intervention_proportion_split(
                    adata, interv_frac=test_size, sample_frac=0.2, seed=i)
                self.save_split(adata, split_version, split_label)
        elif split_version == "gene-holdout":
            for i, gene in enumerate(adata.var_names[adata.var['gene_perturbed']]):
                split_label = f"gene_{gene}"
                gene_holdout(adata, hold_out_gene=gene)
                self.save_split(adata, split_version, split_label)
            split_label = f"gene_{gene}"
        elif split_version == "all_train":
            split_label = "all_train"
            adata.obs['set'] = 'train'
            self.save_split(adata, split_version, split_label)
        elif split_version == "gene-holdout-stratified":
            for i in range(n_splits):
                split_label = f"gene_holdout_stratified_{gene_holdout}_{i}_%0.1f" % (test_size)
                gene_holdout_stratified(adata, hold_out_gene=gene_holdout, test_frac=test_size, seed=i)
                self.save_split(adata, split_version, split_label)
        else:
            raise ValueError("Invalid split version")
        return None

    def get_train_test_splits(self,
                              split_version: str = "shuffled",
                              split_labels: list = None,
                              backed: bool = False
                              ):
        """Load a train test split dataset for a given split_version

        Args:
            split_version: Version of the split
                shuffled: random split
                gene-holdout: holdout perturbation on specific genes
                total-intervention: holdout intervention by proportion on entire dataset
            split_labels: List of str
                List of split labels
                if None - use all found in the folder
            backed: Whether to load the data backed

        Returns:
            split_labels: List of split labels
            split_datasets: List of train test split datasets
        """
        # store split_label entries in array:
        split_version_folder, split_labels_ = self.split_ver_folder(
            split_version)

        if split_labels is None:
            split_labels = split_labels_

        split_datasets = []
        for label in split_labels:
            file = os.path.join(split_version_folder, label, label + ".h5ad")
            if backed:
                split_datasets.append(sc.read_h5ad(file, backed='r+'))
            else:
                split_datasets.append(sc.read_h5ad(file))

        return split_labels, split_datasets

    def store_inference_results(
            self,
            split_labels: list[str],
            adj_matrices: list[np.array],
            split_version: str = "shuffled",
            model_name: str = None,
            plot: bool = False
    ) -> None:
        """Store inference results for a given split_version.

        Save results in Folder Hierarchy:
        - dataset_name
            - split-version
                - split_label
                    - model_name

        Args:
            split_labels: List of split labels.
            adj_matrix: Adjacency matrix from the model (for each split label).
            split_version: Version of the split.
            split_label: Label of the split.
            model_name: Name of the model to run inference.
            plot: Whether to create plot the adjacency matrix.
        """

        if model_name is None:
            raise ValueError("Model name not provided")

        split_version_folder, _ = self.split_ver_folder(split_version)

        for label, adj_matrix in zip(split_labels, adj_matrices):
            model_output_folder = os.path.join(
                split_version_folder, label, model_name)
            os.makedirs(model_output_folder, exist_ok=True)
            np.save(os.path.join(model_output_folder,
                    model_name + "_adj_matrix.npy"), adj_matrix)
            if plot:
                scpi.utils.plot_adjacency_matrix(
                    adj_matrix, title=model_name, output_folder=model_output_folder)
                # scpi.eval.plot_adjacency_matrix(adj_matrix, title=model_name, output_folder=model_output_folder)
        return None

    def load_inference_results(self,
                               split_version: str = "shuffled",
                               model_name: str = None,
                               split_labels: list[str] = None
                               ):
        """Load inference results for a given split_version.

        Args:
            split_version: Version of the split.
            model_name: Name of the model to run inference.
            split_labels (optional): List of split labels. If None, load all found in the folder.

        Returns:
            split_labels: List of split labels.
            adj_matrices: List of adjacency matrices.
        """
        if model_name is None:
            raise ValueError("Model name not provided")

        split_version_folder, split_labels_ = self.split_ver_folder(
            split_version)
        if split_labels is None:
            split_labels = split_labels_

        adj_matrices = []
        for label in split_labels:
            model_output_folder = os.path.join(
                split_version_folder, label, model_name)
            adj_matrix = np.load(os.path.join(
                model_output_folder, model_name + "_adj_matrix.npy"))
            adj_matrices.append(adj_matrix)

        return split_labels, adj_matrices


class SimDataManager():
    """Data management utilities for (a) given dataset(s).

    This class can:
        - create multiple train test splits
        - load and save adata with split information:
            split_version, split_label
            -> for inference and evaluation
        - load data for inference
        - store inference results
        - load data for evaluation

    Args:
        output_folder: Folder to save the files in
        dataset_name: Name of the dataset - where files will be stored
        adata_objects: Annotated expression data objects from scanpy
            should be fully preprocessed and ready for inference
        adj_matrices: Ground-Truth Adjacency Matrices for each Simulated Dataset

    Attributes:
        output_folder: Folder to save the files in
        dataset_name: Name of the dataset - where files will be stored
        adata_objects: Annotated expression data object from scanpy
            should be fully preprocessed and ready for inference
        adj_matrices: Ground-Truth Adjacency Matrices for each Simulated Dataset
    """

    def __init__(self, dataset_name: str, adata_objects: list, adj_matrices: list, output_folder: str = "../data/data_out"):
        """Initialize the data manager.

        Args:
            dataset_name: Name of the dataset - where files will be stored.
            adata_objects: List of AnnData - annotated expression data objects. Should be fully preprocessed and ready for inference.
            adj_matrices: List of np.arrays - Adjacency matrices for each simulated dataset.
            output_folder (str, optional): Base Location to save the files in. Defaults to "../data/data_out".
        """
        self.adata_objects = adata_objects
        self.adj_matrices = adj_matrices
        self.output_folder = output_folder
        self.dataset_name = dataset_name

    def split_ver_folder(self, split_version: str):
        """Get the folder for a given split version for this dataset"""
        split_ver_folder = os.path.join(
            self.output_folder, self.dataset_name, split_version)
        split_labels = os.listdir(split_ver_folder)
        split_labels.sort()
        return split_ver_folder, split_labels

    def save_split(self,
                   adata_train: AnnData,
                   adata_val: AnnData,
                   adj_matrix: np.array,
                   split_version: str,
                   split_label: str
                   ) -> None:
        """Save the data split in the appropriate folder

        saves the annotated adata object in folder hierarchy:
        output_folder/
        ├─ dataset_name/
        │  ├─ split_version/
        │  │  ├─ split_label/
        │  │  │  ├─ split_label_train.h5ad (can include test subset)
        │  │  │  ├─ split_label_val.h5ad
        │  |  |  ├─ gt_adj_matrix.csv
        │  │  │  ├─ model_name/
        │  │  │  │  └─ ...
        │  │  │  └─ ...
        │  │  └─ ...
        │  └─ ...

        Args:
            adata: Annotated expression data object from scanpy
                should be fully preprocessed and ready for inference
            split_version: Version of the split
            split_label: Label for the split
            output_folder:Folder to save the files in
        """
        save_folder = os.path.join(self.output_folder,
                                   self.dataset_name, split_version, split_label)
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, f"adata_train.h5ad")
        sc.write(save_file, adata_train)
        save_file = os.path.join(save_folder, f"adata_val.h5ad")
        sc.write(save_file, adata_val)
        save_file = os.path.join(save_folder, f"gt_adj_matrix.csv")
        np.savetxt(save_file, adj_matrix.astype(int), delimiter=",")
        return None

    def store_train_test_split(
            self,
            split_version: str = "shuffled",
            test_size: float = 0.2,
            n_splits: int = 1,
            verbose: bool = False
    ) -> None:
        """Create train test splits for a given split_version

        Args:
            split_version: Version of the split
                shuffled: random split
                gene-holdout: holdout perturbation on specific genes
                total-intervention: holdout intervention by proportion on entire dataset
            test_size: Fraction of the test set
            n_splits: Number of splits to create
        """
        #adata_objs = self.adata_objects
        # Create a folder for the dataset
        dataset_name = self.dataset_name
        os.makedirs(os.path.join(self.output_folder,
                    dataset_name), exist_ok=True)

        # Create a folder for the split version
        os.makedirs(os.path.join(self.output_folder,
                    dataset_name, split_version), exist_ok=True)

        # Create train test splits
        if split_version == "intervention-proportion":
            for i, adata, adj_mat in zip(range(n_splits), self.adata_objects, self.adj_matrices):
                split_label = f"%.f_interv_split_{i}" % (test_size*100)
                adata_train, adata_val = intervention_proportion_split(
                    adata, interv_frac=test_size, sample_frac=0.2, seed=i)
                if verbose:
                    print(adata_train.obs['perturbation'].value_counts())
                    print(adata_val.obs['perturbation'].value_counts())
                self.save_split(adata_train, adata_val, adj_mat,
                                split_version, split_label)
        else:
            raise ValueError("Invalid split version")
        return None

    def get_train_test_splits(self,
                              split_version: str = "shuffled",
                              split_labels: list = None,
                              backed: bool = False,
                              validation: bool = False,
                              gt_adj: bool = False
                              ):
        """Load a train test split dataset for a given split_version

        Args:
            split_version: Version of the split
            split_labels: List of str
                List of split labels
                if None - use all found in the folder
            backed: Whether to load the data backed

        Returns:
            split_labels: List of split labels
            split_datasets_train: List of train split datasets 
            split_datasets_val: List of validation split datasets
            gt_adj_matrices: List of ground-truth adjacency matrices
        """
        # store split_label entries in array:
        split_version_folder, split_labels_ = self.split_ver_folder(
            split_version)

        if split_labels is None:
            split_labels = split_labels_

        split_datasets = []
        gt_adj_matrices = []
        for label in split_labels:
            if validation:
                file = os.path.join(split_version_folder,
                                    label, "adata_val.h5ad")
            else:
                file = os.path.join(split_version_folder,
                                    label, "adata_train.h5ad")
            if backed:
                split_datasets.append(sc.read_h5ad(file, backed='r+'))
            else:
                split_datasets.append(sc.read_h5ad(file))
            gt_adj_matrices.append(np.loadtxt(os.path.join(
                split_version_folder, label, "gt_adj_matrix.csv"), delimiter=","))
        
        if gt_adj:
            return split_labels, split_datasets, gt_adj_matrices
        else:
            return split_labels, split_datasets

    def store_inference_results(
            self,
            split_labels: list[str],
            adj_matrices: list[np.array],
            split_version: str = "shuffled",
            model_name: str = None,
            plot: bool = False
    ) -> None:
        """Store inference results for a given split_version.

        Save results in Folder Hierarchy:
        - dataset_name
            - split-version
                - split_label
                    - model_name

        Args:
            split_labels: List of split labels.
            adj_matrix: Adjacency matrix from the model (for each split label).
            split_version: Version of the split.
            split_label: Label of the split.
            model_name: Name of the model to run inference.
            plot: Whether to create plot the adjacency matrix.
        """

        if model_name is None:
            raise ValueError("Model name not provided")

        split_version_folder, _ = self.split_ver_folder(split_version)

        for label, adj_matrix in zip(split_labels, adj_matrices):
            model_output_folder = os.path.join(
                split_version_folder, label, model_name)
            os.makedirs(model_output_folder, exist_ok=True)
            np.save(os.path.join(model_output_folder,
                    model_name + "_adj_matrix.npy"), adj_matrix)
            if plot:
                scpi.utils.plot_adjacency_matrix(
                    adj_matrix, title=model_name, output_folder=model_output_folder)
                # scpi.eval.plot_adjacency_matrix(adj_matrix, title=model_name, output_folder=model_output_folder)
        return None

    def load_inference_results(self,
                               split_version: str = "shuffled",
                               model_name: str = None,
                               split_labels: list[str] = None
                               ):
        """Load inference results for a given split_version.

        Args:
            split_version: Version of the split.
            model_name: Name of the model to run inference.
            split_labels (optional): List of split labels. If None, load all found in the folder.

        Returns:
            split_labels: List of split labels.
            adj_matrices: List of adjacency matrices.
        """
        if model_name is None:
            raise ValueError("Model name not provided")

        split_version_folder, split_labels_ = self.split_ver_folder(
            split_version)
        if split_labels is None:
            split_labels = split_labels_

        adj_matrices = []
        for label in split_labels:
            print(split_version_folder, label, model_name)
            print(label)
            model_output_folder = os.path.join(
                split_version_folder, label, model_name)
            adj_matrix = np.load(os.path.join(
                model_output_folder, model_name + "_adj_matrix.npy"))
            adj_matrices.append(adj_matrix)

        return split_labels, adj_matrices
