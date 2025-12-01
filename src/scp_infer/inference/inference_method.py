"""class InferenceMethod: template for inference methods"""
import os
from abc import ABC, abstractmethod
import numpy as np
from anndata import AnnData


class InferenceMethod(ABC):
    """
    Template for Inference Methods

    Contains placeholders for all the utilities neccesary for implementing inference Methods.

    Wanted functionalities:
        - training for each Algorithm
        - test/val split
        - give to algorithm?
    class Methods:
    infer (train)
        - save trained parameters
        - return: Graph / Adjacency Matrix
    eval (test)
        - return: Graph / Adjacency Matrix
    add save/load data options?

    Args:
        adata_obj: Annotated expression data object from scanpy
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation

    Attributes:
        adata_obj: Annotated expression data object from scanpy, should be fully preprocessed, filtered for training data and ready for inference
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation
    """
    max_features: int | None = None  # no limit by default

    def __init__(
            self,
            adata_obj: AnnData,
            output_dir: str = None,
            verbose: bool = False,
            limit_feat_nr: bool = True
    ) -> None:
        """initialize Inference Method"""            
        self.adata_obj = adata_obj
        self.output_dir = output_dir
        self.verbose = verbose

        # Feature number limit check
        n_features = adata_obj.n_vars
        max_allowed = getattr(self, 'max_features', None)

        if max_allowed is not None and n_features > max_allowed:
            msg = (
                f"Warning: This method supports up to {max_allowed} features, "
                f"but received {n_features}."
            )
            if limit_feat_nr:
                raise ValueError(msg)
            elif verbose:
                print(msg)

    @abstractmethod
    def convert_data(self):
        """convert adata entries into respective format for algorithm"""
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        save_output: bool = True,
        **kwargs
    ) -> np.array:
        """perform inference

        Args:
            save_output: bool, save the output to a file
            **kwargs: additional arguments for inference

        Returns:
            np.array: adjacency matrix
            time: time taken for inference
        """
        raise NotImplementedError
    
    def save_output(self, model_name) -> None:
        """save output - save results of inference to files in output directory:
        - adjacency matrix
        - time taken for inference
        - memory usage
        """
        if self.verbose:
            print("Saving output")
        if self.output_dir is None:
            raise ValueError("Output directory not provided")
        else:

            os.makedirs(self.output_dir, exist_ok=True)
            np.savetxt(f"{self.output_dir}/{model_name}_adjacency_matrix.txt", self.adjacency_matrix, fmt='%f')
            np.savetxt(f"{self.output_dir}/{model_name}_time.txt", [self.runtime])
            # save memory usage
            np.savetxt(f"{self.output_dir}/{model_name}_memory_usage.txt", [self.memory_usage])
            if self.verbose:
                print("Output saved")

    def compute_loss(self) -> float:
        """compute loss"""
        raise NotImplementedError

    def save_model(self) -> None:
        """save model"""
        raise NotImplementedError
    
    def load_model(self) -> None:
        """load model"""
        raise NotImplementedError