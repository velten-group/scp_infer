"""Application of Stable Differential Causal Discovery (SDCD) algorithm to infer GRN (using sdcd-PyPI package)"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..inference_method import InferenceMethod


try:
    from sdcd.models import SDCD
    from sdcd.utils import create_intervention_dataset
except ImportError:
    print("Error importing StableDCD. StableDCD dependencies might not be installed.")



class SDCDImp(InferenceMethod):
    """StableDCD implementation

    The input data should be formatted as a Pandas Dataframe where each row corresponds 
    to one observation and each column corresponds to one variable. There should be an additional column 
    reports which variable(s) was intervened on for the given observation. Here it is labeled as perturbation_label. 
    For rows that do not have any interventions, the value should be set to "obs".

    Then:
    X_dataset = create_intervention_dataset(X_df, perturbation_colname="perturbation_label")

    Args:
        adata_obj: Annotated expression data object from scanpy
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation

    Attributes:
        X_dataset: expression data used by SDCD


    """

    def convert_data(self, verbose: bool = False):
        """convert adata entries into GRNBoost2 format"""
        # create dataframe for count data
        X_df = self.adata_obj.to_df()
        # create perturbation label column:
        # non-targeting -> obs
        perturbation_col = []
        for sample, perturbation in zip(self.adata_obj.obs.index, self.adata_obj.obs["perturbation"]):
            if perturbation == "non-targeting":
                perturbation_col.append("obs")
            else:
                # append index of the perturbed feature
                perturbation_col.append(
                    self.adata_obj.var_names.get_loc(perturbation))
        # add perturbation label column as Series
        
        X_df.insert(len(X_df.columns), "perturbation_label", np.array(perturbation_col))

        # transform to sdcd data format
        X_dataset = create_intervention_dataset(
            X_df, perturbation_colname="perturbation_label")
        self.X_dataset = X_dataset

    def infer(
        self,
        plot: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> np.array:
        if self.verbose:
            print("Running SDCD")

        start_time = time.process_time()
        # run algorithm
        model = SDCD()
        model.train(self.X_dataset, finetune=True)
        adj_matrix = model.get_adjacency_matrix(threshold=True)

        end_time = time.process_time()
        total_time = end_time - start_time

        if self.verbose:
            print("SDCD finished")

        if plot or self.verbose:
            _, ax = plt.subplots()
            fig1 = ax.matshow(adj_matrix)
            plt.colorbar(fig1)
            plt.title("GRNBoost2: Adjacency matrix")
            plt.savefig("GRNBoost2_adjacency_matrix.png")
            plt.plot()

        return adj_matrix, total_time
