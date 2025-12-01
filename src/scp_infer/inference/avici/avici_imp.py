"""Application of GRNBoost2 Algorithm (using arboreto)"""
import time
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

from ..inference_method import InferenceMethod

try:
    import avici
except ImportError:
    print("Error importing AVICI. AVICI or dependencies might not be installed.")


class AVICI_Imp(InferenceMethod):
    """AVICI implementation

    Args:
        adata_obj: Annotated expression data object from scanpy
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation

    Attributes:
        x: np.array, data matrix for the AVICI algorithm
        interv: np.array, intervention matrix for the AVICI algorithm
        model_weights: str, model weights to be used for the AVICI algorithm
        model: model object for the AVICI algorithm
    """

    def __init__(
            self,
            adata_obj,
            output_dir: str = None,
            model_weights: str = "scm-v0",
            verbose: bool = False
    ):
        #tracemalloc.start()
        super(AVICI_Imp, self).__init__(adata_obj, output_dir, verbose)
        self.model_weights = model_weights
        self.model = avici.load_pretrained(model_weights)

    def convert_data(self):
        """convert adata entries into AVICI format"""
        self.x = self.adata_obj.X
        self.interv = self.adata_obj.layers['perturbed_elem_mask'].astype(int)
        if self.verbose:
            print("AVICI data conversion done")

    def infer(
        self,
        plot: bool = False,
        save_time: bool = True,
        return_weighted: bool = False,
        file_label: str = "AVICI",
        **kwargs
    ) -> np.array:
        if self.verbose:
            print("Running AVICI")

        tracemalloc.start()
        start_time = time.time()
        # run algorithm
        g_prob = self.model(x=self.x, interv=self.interv, return_probs = return_weighted).T

        end_time = time.time()
        
        self.runtime = end_time - start_time
        self.adjacency_matrix = g_prob
        self.memory_usage = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        if self.verbose:
            print("AVICI fnished")
            print("Time taken: ", self.runtime)
            print("Memory usage: ", self.memory_usage)

        if plot:
            _, ax = plt.subplots()
            fig1 = ax.matshow(g_prob)
            plt.colorbar(fig1)
            plt.title("AVICI: Adjacency matrix")
            plt.savefig("AVICI_adjacency_matrix.png")
            plt.plot()

        #if save_time:
        #    np.savetxt(self.output_dir + "/AVICI_time.txt", [self.runtime])

        super(AVICI_Imp, self).save_output(file_label)

        return g_prob, self.runtime
