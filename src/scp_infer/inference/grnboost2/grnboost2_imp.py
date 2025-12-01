"""Application of GRNBoost2 Algorithm (using arboreto)"""
import time
import numpy as np
import pandas as pd
try:
    from scp_infer.thirdparty.arboreto_local.algo import grnboost2
except ImportError:
    print("Error importing arboreto. Please install arboreto using 'pip install arboreto'")
import matplotlib.pyplot as plt
import tracemalloc

from ..inference_method import InferenceMethod

class GRNBoost2Imp(InferenceMethod):
    """GRNBoost2 implementation

    Args:
        adata_obj: Annotated expression data object from scanpy
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation

    Attributes:
        tf_names: TF/gene names used by GRNBosst2
        expression_data: expression data used by GRNBoost2

    """

    def convert_data(self, verbose: bool = False):
        """convert adata entries into GRNBoost2 format"""
        # Load the TF names
        self.tf_names = self.adata_obj.var_names
        self.expression_data = self.adata_obj.to_df()
        if verbose:
            print("TF names: ", len(self.tf_names))
            print("Expression data shape: ", self.expression_data.shape)


    def infer(
        self,
        plot: bool = False,
        verbose: bool = False,
        save_time: bool = True,
        file_label: str = "GRNBoost2",
        **kwargs
    ) -> np.array:
        if self.verbose:
            print("Running GRNBoost2")

        tracemalloc.start()
        start_time = time.time()
        # run algorithm
        network = grnboost2(expression_data=self.expression_data, verbose=self.verbose)
        end_time = time.time()
        total_time = end_time - start_time

        self.runtime = total_time
        self.memory_usage = tracemalloc.get_traced_memory()[1]

        if self.verbose:
            print("GRNBoost2 fnished")
            print("Time taken: ", total_time)
            print("network shape: ", network.shape)
            network.head()

        # Create adjacency matrix: Format x-axis(ind1): target, y-axis(ind2): TF
        num_genes = len(self.adata_obj.var_names)
        grnboost_matrix = np.zeros((num_genes, num_genes))
        for i in range(network.shape[0]):
            for ind1, gene_1 in enumerate(self.adata_obj.var_names):
                for ind2, gene_2 in enumerate(self.adata_obj.var_names):
                    if network['target'].iloc[i] == gene_1:
                        if network['TF'].iloc[i] == gene_2:
                            grnboost_matrix[ind1, ind2] = network['importance'].iloc[i]

        if plot:
            _, ax = plt.subplots()
            fig1 = ax.matshow(grnboost_matrix)
            plt.colorbar(fig1)
            plt.title("GRNBoost2: Adjacency matrix")
            plt.savefig("GRNBoost2_adjacency_matrix.png")
            plt.plot()
        
        #if save_time:
        #    np.savetxt(self.output_dir + "/grnboost2_time.txt", [total_time])
        
        self.adjacency_matrix = grnboost_matrix
        super(GRNBoost2Imp, self).save_output(file_label)

        return grnboost_matrix, total_time
