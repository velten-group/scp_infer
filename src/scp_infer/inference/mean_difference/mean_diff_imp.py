"""Application of Mean Difference Algorithm"""
import time

from networkx import adjacency_matrix
import numpy as np
import pandas as pd
try:
    from scp_infer.thirdparty.arboreto_local.algo import grnboost2
except ImportError:
    print("Error importing arboreto. Please install arboreto using 'pip install arboreto'")
import matplotlib.pyplot as plt
import tracemalloc

from ..inference_method import InferenceMethod

class MeanDifferenceImp(InferenceMethod):
    """Mean Difference implementation

    Args:
        adata_obj: Annotated expression data object from scanpy
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation

    Attributes:
        tf_names: TF/gene names used by GRNBosst2
        expression_data: expression data used by GRNBoost2

    """

    def convert_data(self, verbose: bool = False):
        """convert adata entries into Mean Difference format"""
        # Load the TF names
        return None


    def infer(
        self,
        plot: bool = False,
        verbose: bool = False,
        save_time: bool = True,
        file_label: str = "MeanDifference",
        **kwargs
    ) -> np.array:
        if self.verbose:
            print("Running MeanDifference")

        tracemalloc.start()
        start_time = time.time()
        # run algorithm
        
        
        # A. compute mean expression of each gene in observational and each perturbation condition
        observational_data = self.adata_obj.X[self.adata_obj.obs['perturbation'] == 'non-targeting']
        observational_mean = np.mean(observational_data, axis=0)

        perturbed_genes = self.adata_obj.var_names[self.adata_obj.var['gene_perturbed']]
        pert_mean_expressions = np.zeros((len(perturbed_genes), self.adata_obj.n_vars))
        for i, perturbation in enumerate(perturbed_genes):
            perturbed_data = self.adata_obj.X[self.adata_obj.obs['perturbation'] == perturbation]
            pert_mean_expressions[i, :] = np.mean(perturbed_data, axis=0)

        # B. compute mean difference between perturbed and non-perturbed conditions
        mean_diff = pert_mean_expressions - observational_mean
        # set diagonal to zero to avoid self-loops
        np.fill_diagonal(mean_diff, 0)

        # C. optional: scaling with Bayes correction factor
        # not implemented in this version

        adjacency_matrix = np.abs(mean_diff).T  # Use absolute values for adjacency matrix

        end_time = time.time()
        total_time = end_time - start_time

        self.runtime = total_time
        self.memory_usage = tracemalloc.get_traced_memory()[1]



        

        if plot:
            _, ax = plt.subplots()
            fig1 = ax.matshow(adjacency_matrix, cmap='viridis')
            plt.colorbar(fig1)
            plt.title("MeanDifference: Adjacency matrix")
            plt.savefig("MeanDifference_adjacency_matrix.png")
            plt.plot()
        
        #if save_time:
        #    np.savetxt(self.output_dir + "/grnboost2_time.txt", [total_time])
        
        self.adjacency_matrix = adjacency_matrix
        super(MeanDifferenceImp, self).save_output(file_label)

        return adjacency_matrix, total_time
