"""
Application of GIES Algorithm

GIES implementation from: https://github.com/juangamella/gies
Slightly modified files from the original used
they should be stored in scp-infer/algorithm_implementations/gies_local

"""

#import scp_infer.thirdparty.gies_local as gies
import gies
import os
import sys
import time
import tracemalloc

import numpy as np
from ..inference_method import InferenceMethod
from scp_infer.adata import convert_onehot_to_indices


# 1. the local gies algorithm has to be loaded

current_dir = os.path.abspath(".")

# print("Current dir: ", current_dir)
sys.path.append(os.path.join(current_dir, 'algorithm_implementations'))
# print(sys.path)
# pyright: reportMissingImports=false


class GIESImp(InferenceMethod):
    """GIES implementation

    Args:
        adata_obj: Annotated expression data object from scanpy
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation

    Attributes:
        data_matrix: np.array, data matrix for the GIES algorithm
        intervention_list: list, list of list of indices of the perturbed genes per intervention
    """

    def __create_data_matrix_gies(self, verbose=False):
        """
        Create the data matrix for the GIES algorithm.
            - shape: (n_interventions, n_observations/intervention, n_features)
            - here: take minimum number of observations/intervention and discard all else

        Returns:
            intervention_list: list, list of list of indices of the perturbed genes per intervention
            data_matrix: list, data matrix for the GIES algorithm
                each entry is a np.array containing the data for one intervention
        """
        adata_obj = self.adata_obj
        #perturbed_indices = adata_obj.obs["perturbed_gene_indices"]
        perturbed_indices = convert_onehot_to_indices(self.adata_obj)

        # Step 0: Create dictionary to map gene indices to unique intervention scenarios
        # key: perturbation_indices - tuple, value: list of sample indices
        geneidx_to_interventions = {}
        for idx, pert in enumerate(perturbed_indices):
            pert_tuple = tuple(pert)
            if pert not in geneidx_to_interventions:
                geneidx_to_interventions[pert_tuple] = []
            geneidx_to_interventions[pert_tuple].append(idx)

        # Step 1: Create Intervention List
        intervention_list = []
        for pert_tuple, samples in geneidx_to_interventions.items():
            intervention_list.append(list(pert_tuple))
            intervention_gene_names = [
                adata_obj.var_names[i] for i in pert_tuple]

        if verbose:
            print("Intervention List created: ", len(
                intervention_list), "unique perturbations")

        # Step 2: Create Data Matrix
        # 1st create skeleton
        data_matrix = []
        for i in range(len(intervention_list)):
            data_matrix.append([])

        # 2nd fill in the data in index according to intervention list

        for i, (pert_tuple, samples) in enumerate(geneidx_to_interventions.items()):
            for idx in samples:
                data_matrix[i].append(adata_obj.X[idx, :])

        # 2.3 turn each list entry into numpy matrix
        for i, dm_elem in enumerate(data_matrix):
            data_matrix[i] = np.array(dm_elem, dtype=float)

        return intervention_list, data_matrix

    def convert_data(self, singularized: bool = False, verbose: bool = False):
        """
        convert adata entries into GIES format and store in  data_matrix, intervention_list

        Args:
            singularized: if True, each observation gets stored under separate intervention entry,
                if False, store all observations for each intervention in one entry
            verbose: verbosity of the conversion process
        """
        # Load the data matrix
        if verbose:
            print("Converting data to GIES format")
        self.intervention_list, self.data_matrix = self.__create_data_matrix_gies(
            verbose=verbose)

        if verbose:
            # Look at results
            print(self.adata_obj.obs['gene_perturbation_mask'].sum(
            ), " gene perturbations")
            print(len(self.intervention_list), " interventions")
            print("Intervention list: ", self.intervention_list[:15])

            print("")
            print("Data matrix:")
            print("Length of data matrix: ", len(self.data_matrix))

            length = np.array([])
            for sub_array in self.data_matrix:
                length = np.append(length, len(sub_array))

            print("Minimum length: ", np.min(length))
            print("Maximum length: ", np.max(length))
            print("Average length: ", np.mean(length))
            print("Total Samples: ", np.sum(length))
            print("Total interventional Samples: ", np.sum(length[1:]))

            print("Entries per Intervention: ", length)

            # print("GIES final data shape: ", np.shape(self.data_matrix))

    def infer(
        self,
        plot: bool = False,
        save_time: bool = True,
        file_label: str = "GIES",
        **kwargs
    ) -> np.array:
        """perform inference

        Args:
            plot: bool, plot the adjacency matrix
            **kwargs: additional arguments for inference
        Returns:
            estimate: np.array, adjacency matrix estimate
            time: float, time taken for inference
        """
        if self.verbose:
            print("Running GIES")

        tracemalloc.start()
        start_time = time.time()
        # run algorithm
        estimate, _ = gies.fit_bic(
            self.data_matrix, self.intervention_list, A0=None)
        end_time = time.time()
        estimate = estimate.T

        total_time = end_time - start_time
        self.runtime = total_time
        self.adjacency_matrix = estimate
        self.memory_usage = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        if self.verbose:
            print("GIES fnished")
            print("Time taken: ", total_time)
            print("estimate shape: ", estimate.shape)
            #print("GIES matrix: ", estimate)

        if plot:
            import matplotlib.pyplot as plt
            plt.imshow(estimate)
            plt.colorbar()
            plt.show()

        #if save_time:
        #    np.savetxt(self.output_dir + "/gies_time.txt", [total_time])
        super(GIESImp, self).save_output(file_label)
        return estimate, total_time
