"""Application of DCDI algorithm to infer GRN from single-cell data. (local implementation)


"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

from ..inference_method import InferenceMethod
from scp_infer.adata import convert_onehot_to_indices
try:
    import torch

    import scp_infer.thirdparty.dcdi_local as dcdi
    from scp_infer.thirdparty.dcdi_local.models.learnables import LearnableModel_NonLinGaussANM, LearnableModel_NonLinGauss_DropOut, LearnableModel_NonLinGauss_DropOut_global,LearnableModel_NonLin_NegBin
    from scp_infer.thirdparty.dcdi_local.models.learnables_global_param import LearnableModel_NonLinGauss_Sigmoid_Dropout, LearnableModel_NonLin_NegBin_sc
    from scp_infer.thirdparty.dcdi_local.models.learnables_catLL import LearnableModel_NonLinGauss_DropOut_cat, LearnableModel_NonLinGauss_DropOut_global_cat, LearnableModel_NonLinGauss_Sigmoid_Dropout_cat
    from scp_infer.thirdparty.dcdi_local.models.flows import DeepSigmoidalFlowModel
    from scp_infer.thirdparty.dcdi_local.train import train, retrain, compute_loss
    #from scp_infer.thirdparty.dcdi_local.data import DataManagerFile
    from scp_infer.thirdparty.dcdi_local.utils.save import dump, load

    #import dcdi_local as dcdi
    #from dcdi_local.models.learnables import LearnableModel_NonLinGaussANM, LearnableModel_NonLinGauss_DropOut, LearnableModel_NonLinGauss_DropOut_global,LearnableModel_NonLin_NegBin
    #from dcdi_local.models.learnables_global_param import LearnableModel_NonLinGauss_Sigmoid_Dropout
    #from dcdi_local.models.flows import DeepSigmoidalFlowModel
    #from dcdi_local.train import train, retrain, compute_loss
    #from dcdi_local.data import DataManagerFile
    #from dcdi_local.utils.save import dump, load
    from .dcdi_load import DataManagerDirect
except ImportError:
    print("Error importing DCDI. DCDI dependencies might not be installed.")

# import the local dcdi algorithm
# pyright: reportMissingImports=false


def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)

def _print_none(stage, step, metrics, throttle=None):
    return


def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))


class DCDIImp(InferenceMethod):
    """
    DCDI implementation

    Args:
        adata_obj: Annotated expression data object from scanpy
        output_dir: directory to save output files
        verbose: default verbosity of the algorithm implementation
        train_iterations: number of training iterations
        learning_rate: learning rate for the algorithm
        reg_coeff: regularization coefficient for the algorithm
        train_data: training data
        test_data: testing data
    """
    max_features = 60

    def __init__(
            self,
            adata_obj,
            size_factors = None,
            output_dir: str = None,
            verbose: bool = False,
            train_iterations: int = 1000000,
            learning_rate: float = 1e-3,
            reg_coeff: float = 0.1,
            num_layers: int = 2,
            hid_dim: int = 16,
            train_samples: float = 0.8,
            model: str = "DCDI-G",
            intervention_type='imperfect',
            weighted_adjacency: bool = False,    # whether to save weighted adjacency matrix
            file_label: str = "GRNBoost2",
            log_dropout: bool = False,
            sig_k: float = 1.0,
            sig_b: float = 4.0,
            lock_sigmoid: bool = False,
            use_raw: bool = False,
            use_GPU = True,
            **kwargs
    ):
        """Initialize the DCDI implementation, set all default Hyperparameters.
        """

        super(DCDIImp, self).__init__(adata_obj, output_dir, verbose,**kwargs)

        if size_factors is None:
            size_factors = np.ones(adata_obj.n_obs)
        self.size_factors = size_factors

        opt = argparse.Namespace()
        self.opt = opt
        # experiment
        opt.exp_path = output_dir  # Path to experiments
        opt.train = True            # Run `train` function, get /train folder
        opt.retrain = False         # after to-dag or pruning, retrain model
        # from scratch before reporting nll-val
        opt.dag_for_retrain = None  # path to a DAG in .npy format which will be used
        # for retrainig. e.g.  /code/stuff/DAG.npy
        opt.random_seed = 42        # Random seed for pytorch and numpy

        # data
        opt.data_path = None        # Path to data files
        opt.i_dataset = None        # dataset index
        # Number of variables
        opt.num_vars = len(self.adata_obj.var_names)
        opt.train_samples = train_samples    # Number of samples used for training
        # (default is 80% of the total size)
        opt.test_samples = None     # Number of samples used for testing
        # (default is whatever is not used for training)
        opt.num_folds = 5           # number of folds for cross-validation
        opt.fold = 0                # fold we should use for testing
        opt.train_batch_size = 64   # number of samples in a minibatch
        opt.num_train_iter = train_iterations  # number of meta gradient steps
        opt.normalize_data = False  # (x - mu) / std
        # When loading data, will remove some regimes from data set
        opt.regimes_to_ignore = None
        # When using --regimes-to-ignore, we evaluate performance
        # on new regimes never seen during training (use after retraining).
        opt.test_on_new_regimes = False

        # model
        opt.model = model           # model class (DCDI-G or DCDI-DSF)
        opt.num_layers = num_layers  # number of hidden layers
        opt.hid_dim = hid_dim       # number of hidden units per layer
        opt.nonlin = 'leaky-relu'   # leaky-relu | sigmoid
        opt.flow_num_layers = 2     # number of hidden layers of the DSF
        opt.flow_hid_dim = 16       # number of hidden units of the DSF

        # intervention
        opt.intervention = True     # Use data with intervention
        opt.dcd = False             # Use DCD (DCDI with a loss
        # not taking into account the intervention)
        # Type of intervention: perfect or imperfect
        opt.intervention_type = intervention_type
        # If the targets of the intervention are known or unknown
        opt.intervention_knowledge = "known"
        # Coefficient of the regularisation in the unknown interventions case (lambda_R)
        opt.coeff_interv_sparsity = 1e-8

        # optimization
        opt.optimizer = "rmsprop"   # sgd|rmsprop
        opt.lr = learning_rate      # Learning rate for optim
        # Learning rate for optim after first subproblem.
        opt.lr_reinit = None
        # Default mode reuses --lr.
        opt.lr_schedule = None      # Learning rate for optim, change initial lr as a
        # function of mu: None|sqrt-mu|log-mu
        opt.stop_crit_win = 100     # window size to compute stopping criterion
        # Regularization coefficient (lambda), search space is [1e-7,....,1e2]
        opt.reg_coeff = reg_coeff

        # Augmented Lagrangian options
        opt.omega_gamma = 1e-4      # Precision to declare convergence of subproblems
        opt.omega_mu = 0.9          # After subproblem solved, h should have reduced by this ratio
        opt.mu_init = 1e-8          # initial value of mu
        opt.mu_mult_factor = 2      # Multiply mu by this amount when constraint
        # not sufficiently decreasing
        opt.gamma_init = 0.         # initial value of gamma
        # Stop when |h|<X. Zero means stop AL procedure only when h==0
        opt.h_threshold = 1e-8

        # misc
        opt.patience = 10           # Early stopping patience in --retrain.
        opt.train_patience = 5      # Early stopping patience in --train after constraint
        opt.train_patience_post = 5  # Early stopping patience in --train after threshold

        # logging
        opt.plot_freq = 100       # plotting frequency
        # do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)
        opt.no_w_adjs_log = False
        # Plot density (only implemented for 2 vars)
        opt.plot_density = False

        # dropout
        self.log_dropout = log_dropout
        self.sig_k = sig_k
        self.sig_b = sig_b
        self.lock_sigmoid = lock_sigmoid
        self.use_raw = use_raw

        # device and numerical precision
        opt.gpu = use_GPU              # Use GPU
        opt.float = False           # Use Float precision

        self.plotting_callback = None
        self.weighted_adjacency = weighted_adjacency

        # Control as much randomness as possible
        torch.manual_seed(opt.random_seed)
        np.random.seed(opt.random_seed)

        if opt.lr_reinit is not None:
            assert opt.lr_schedule is None, "--lr-reinit and --lr-schedule are mutually exclusive"

        # Initialize metric logger if needed
        metrics_callback = _print_metrics

        # adjust some default hparams
        if opt.lr_reinit is None:
            opt.lr_reinit = opt.lr

        # Use GPU
        if opt.gpu:
            if opt.float:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            if opt.float:
                torch.set_default_tensor_type('torch.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.DoubleTensor')

    def get_data(self, adata_obj, set_regimes=True):
        """Extract the data from the adata object and return it in the format required by the DCDI algorithm.

        Args:
            adata_obj: Annotated expression data object from scanpy

        Returns:
            x: numpy array, shape (num_samples, num_genes), gene expression data
            mask: numpy array, shape (num_samples, num_genes), mask of the gene expression data
            regimes: list of integers, each integer represents the regime of the corresponding sample
            intervention_mask: list of lists, each list contains the targets feature index of the corresponding intervention
        """
        # 1. Define neccessary annotation objects
        perturbed_indices = convert_onehot_to_indices(adata_obj) # list of lists, each list contains the perturbed gene indices of that sample
        if not self.use_raw:
            expression_data = adata_obj.X.copy()  # numpy array, shape (num_samples, num_genes)
        else:
            expression_data = adata_obj.layers["counts"].copy()
        size_factors = self.size_factors
        geneidx_to_interventions = {}   # key: perturbation_indices - tuple, value: list of sample indices
        perturbation_to_regime = {}     # key: perturbation_indices - tuple, value: regime


        # 2. Create the dictionary of perturbation (gene) indices to sample indices
        for idx, pert in enumerate(perturbed_indices):
            pert_tuple = tuple(pert)
            if pert_tuple not in geneidx_to_interventions:
                geneidx_to_interventions[pert_tuple] = []
            geneidx_to_interventions[pert_tuple].append(idx)


        # 3. Fill up the data, intervention_mask, and regimes arrays
        intervention_mask = []
        treatment_regimes = []
        current_regime = 0
        start_idx = 0
        data = np.zeros_like(expression_data)
        size_factors_sorted = np.zeros_like(size_factors)

        for perturbation_tuple, idx_list in geneidx_to_interventions.items():
            target_genes = list(perturbation_tuple)
            # if the perturbation_tuple is empty, it means no perturbation => regime 0
            regime_label = 0 if len(perturbation_tuple) == 0 else current_regime + 1
            perturbation_to_regime[perturbation_tuple] = regime_label
            intervention_mask.extend([target_genes for _ in range(len(idx_list))])
            treatment_regimes.extend([regime_label for _ in range(len(idx_list))])

            end_idx = start_idx + len(idx_list)
            # ERROR HERE: intervention_mask and treatment_regimes are filled, but data is zero
            # Debug statements to check the state
            #print(f"start_idx: {start_idx}, end_idx: {end_idx}")
            #print(f"idx_list: {idx_list[::500]}")
            #print(f"expression_data[idx_list, :]: {expression_data[idx_list, :][::500]}")
            data[start_idx:end_idx, :] = expression_data[idx_list, :]
            size_factors_sorted[start_idx:end_idx] = size_factors[idx_list]

            #print(f"data[start_idx:end_idx, :]: {data[start_idx:end_idx, :][::500]}")
            start_idx = end_idx

            if len(perturbation_tuple) != 0:
                current_regime += 1

        regimes = np.array(treatment_regimes)
        if set_regimes:
            self.perturbation_to_regime = perturbation_to_regime

        # Debug statement to check the final state
        #print(f"data: {data[::1000]}")
        #print(f"intervention_mask: {intervention_mask[::1000]}")
        #print(f"regimes: {regimes[::1000]}")

        return data, intervention_mask, regimes, size_factors_sorted

    def convert_data(self):
        """Convert the data to the format required by the DCDI algorithm.

        regimes:    list of integers, each integer represents the regime of the corresponding sample
        intervention_mask: list of lists, each list contains the targets feature index of the corresponding intervention
        data: numpy array, shape (num_samples, num_genes), gene expression data

        gene_to_interventions: dict, key: gene name, value: list of integers, each integer represents the sample index of it's samples

        """
        opt = self.opt

        data, intervention_mask, regimes, size_fact_sorted = self.get_data(self.adata_obj)

        self.regimes = regimes
        self.intervention_mask = intervention_mask
        self.data = data

        self.train_data = DataManagerDirect(
            data,
            intervention_mask,
            regimes,
            size_factors=size_fact_sorted,
            train_samples=self.opt.train_samples,
            train=True,
            normalize=False,
            random_seed=self.opt.random_seed,
            intervention=True,
            intervention_knowledge="known",
        )
        self.test_data = DataManagerDirect(
            data,
            intervention_mask,
            regimes,
            size_factors=size_fact_sorted,
            train_samples=self.opt.train_samples,
            train=False,
            normalize=False,
            random_seed=self.opt.random_seed,
            intervention=True,
            intervention_knowledge="known",
        )

        # create learning model and ground truth model
        print('num_regimes:', self.train_data.num_regimes)
        if opt.model == "DCDI-G":
            model = LearnableModel_NonLinGaussANM(opt.num_vars,
                                                  opt.num_layers,
                                                  opt.hid_dim,
                                                  nonlin=opt.nonlin,
                                                  intervention=opt.intervention,
                                                  intervention_type=opt.intervention_type,
                                                  intervention_knowledge=opt.intervention_knowledge,
                                                  num_regimes=self.train_data.num_regimes)
        elif opt.model == "DCDI-Drop":
            model = LearnableModel_NonLinGauss_DropOut(opt.num_vars,
                                                    opt.num_layers,
                                                    opt.hid_dim,
                                                    nonlin=opt.nonlin,
                                                    intervention=opt.intervention,
                                                    intervention_type=opt.intervention_type,
                                                    intervention_knowledge=opt.intervention_knowledge,
                                                    num_regimes=self.train_data.num_regimes)
        elif opt.model == "DCDI-Drop-local-cat":
            model = LearnableModel_NonLinGauss_DropOut_cat(opt.num_vars,
                                                    opt.num_layers,
                                                    opt.hid_dim,
                                                    nonlin=opt.nonlin,
                                                    intervention=opt.intervention,
                                                    intervention_type=opt.intervention_type,
                                                    intervention_knowledge=opt.intervention_knowledge,
                                                    num_regimes=self.train_data.num_regimes)
        elif opt.model == "DCDI-Drop-global":
            model = LearnableModel_NonLinGauss_DropOut_global(opt.num_vars,
                                                          opt.num_layers,
                                                          opt.hid_dim,
                                                          nonlin=opt.nonlin,
                                                          intervention=opt.intervention,
                                                          intervention_type=opt.intervention_type,
                                                          intervention_knowledge=opt.intervention_knowledge,
                                                          num_regimes=self.train_data.num_regimes)
        elif opt.model == "DCDI-Drop-global-cat":
            model = LearnableModel_NonLinGauss_DropOut_global_cat(opt.num_vars,
                                                    opt.num_layers,
                                                    opt.hid_dim,
                                                    nonlin=opt.nonlin,
                                                    intervention=opt.intervention,
                                                    intervention_type=opt.intervention_type,
                                                    intervention_knowledge=opt.intervention_knowledge,
                                                    num_regimes=self.train_data.num_regimes)
        elif opt.model == "DCDI-Drop-sigmoid":
            model = LearnableModel_NonLinGauss_Sigmoid_Dropout(opt.num_vars,
                                                          opt.num_layers,
                                                          opt.hid_dim,
                                                          nonlin=opt.nonlin,
                                                          intervention=opt.intervention,
                                                          intervention_type=opt.intervention_type,
                                                          intervention_knowledge=opt.intervention_knowledge,
                                                          num_regimes=self.train_data.num_regimes,
                                                          log_dropout=self.log_dropout,
                                                          sig_k=self.sig_k,
                                                          sig_b=self.sig_b,
                                                          lock_sigmoid=False)
        elif opt.model == "DCDI-Drop-sigmoid-fixed":
            model = LearnableModel_NonLinGauss_Sigmoid_Dropout(opt.num_vars,
                                                          opt.num_layers,
                                                          opt.hid_dim,
                                                          nonlin=opt.nonlin,
                                                          intervention=opt.intervention,
                                                          intervention_type=opt.intervention_type,
                                                          intervention_knowledge=opt.intervention_knowledge,
                                                          num_regimes=self.train_data.num_regimes,
                                                          log_dropout=self.log_dropout,
                                                          sig_k=self.sig_k,
                                                          sig_b=self.sig_b,
                                                          lock_sigmoid=True)
        elif opt.model == "DCDI-Drop-sigmoid-cat":
            model = LearnableModel_NonLinGauss_Sigmoid_Dropout_cat(opt.num_vars,
                                                          opt.num_layers,
                                                          opt.hid_dim,
                                                          nonlin=opt.nonlin,
                                                          intervention=opt.intervention,
                                                          intervention_type=opt.intervention_type,
                                                          intervention_knowledge=opt.intervention_knowledge,
                                                          num_regimes=self.train_data.num_regimes,
                                                          log_dropout=self.log_dropout,
                                                          sig_k=self.sig_k,
                                                          sig_b=self.sig_b)
            # parameter initialization? k,b
        elif opt.model == "DCDI-NB":
            model = LearnableModel_NonLin_NegBin(opt.num_vars,
                                                  opt.num_layers,
                                                  opt.hid_dim,
                                                  nonlin=opt.nonlin,
                                                  intervention=opt.intervention,
                                                  intervention_type=opt.intervention_type,
                                                  intervention_knowledge=opt.intervention_knowledge,
                                                  num_regimes=self.train_data.num_regimes)
        elif opt.model == "DCDI-NB-sc":
            model = LearnableModel_NonLin_NegBin_sc(opt.num_vars,
                                                  opt.num_layers,
                                                  opt.hid_dim,
                                                  nonlin=opt.nonlin,
                                                  intervention=opt.intervention,
                                                  intervention_type=opt.intervention_type,
                                                  intervention_knowledge=opt.intervention_knowledge,
                                                  num_regimes=self.train_data.num_regimes,
                                                  rescale_means=True)
        elif opt.model == "DCDI-DSF":
            model = DeepSigmoidalFlowModel(num_vars=opt.num_vars,
                                           cond_n_layers=opt.num_layers,
                                           cond_hid_dim=opt.hid_dim,
                                           cond_nonlin=opt.nonlin,
                                           flow_n_layers=opt.flow_num_layers,
                                           flow_hid_dim=opt.flow_hid_dim,
                                           intervention=opt.intervention,
                                           intervention_type=opt.intervention_type,
                                           intervention_knowledge=opt.intervention_knowledge,
                                           num_regimes=self.train_data.num_regimes)
        else:
            raise ValueError("opt.model has to be in {DCDI-G, DCDI-DSF}")
        self.model = model

        # print device of samples, masks and regimes
        print("train_data.adjacency.device:", self.train_data.adjacency.device)
        # print("train_data.samples.device:", self.train_data.samples.device)

    def infer(
        self,
        save_output: bool = True,
        save_time: bool = True,
        file_label: str = "DCDI",
        print_metrics: bool = False,
        **kwargs
    ) -> np.array:
        """
        Parameters for the DCDI algorithm

        store parameters as attributes of opt
        """

        model = self.model
        train_data = self.train_data
        test_data = self.test_data
        if print_metrics:
            metrics_callback = _print_metrics
        else:
            metrics_callback = _print_none

        if self.verbose:
            print("Running DCDI")
        opt = self.opt

        # print("train_data.regimes.device:", train_data.regimes.device)

        # start timer and memory tracking
        # add GPU memory tracking? - torch.cuda.memory_allocated()
        # tracemalloc.start()
        start_time = time.time()

        # train until constraint is sufficiently close to being satisfied
        if opt.train:
            train(model, train_data.adjacency.detach().cpu().numpy(),
                  train_data.gt_interv, train_data, test_data, opt, metrics_callback,
                  self.plotting_callback)

        elif opt.retrain:
            initial_dag = np.load(opt.dag_for_retrain)
            model.adjacency[:, :] = torch.as_tensor(
                initial_dag).type(torch.Tensor)
            best_model = retrain(model, train_data, test_data, "ignored_regimes",
                                 opt, metrics_callback, self.plotting_callback)

        # end timer
        end_time = time.time()
        total_time = end_time - start_time
        self.runtime = total_time

        # save cuda max memory
        self.memory_usage = torch.cuda.max_memory_allocated()  # in bytes

        if self.verbose:
            print("DCDI finished")
            print("Time taken: ", total_time)
            print("Memory usage: ", self.memory_usage)

        if not self.weighted_adjacency:
            adjacency = model.adjacency.detach().cpu().numpy()
        else:
            adjacency = model.get_w_adj().detach().cpu().numpy()
        
        adjacency = adjacency.T

        self.adjacency_matrix = adjacency
        super(DCDIImp, self).save_output(file_label)

        return adjacency, total_time

    def compute_loss(self, adata_obj=None) -> float:
        """Compute the loss of the model on the whole validation set.

        Args:
            None

        Returns:
            float: negative log likelihood of the model
        """
        opt = self.opt
        test_data = self.test_data
        model = self.model
        weights, biases, extra_params = model.get_parameters(mode="wbx")

        # compute loss on whole validation set
        with torch.no_grad():
            # if an external dataset is provided => convert it to the right format
            if adata_obj is not None:
                data, intervention_mask, _, sf = self.get_data(
                    adata_obj, set_regimes=False)
                # print(data, intervention_mask)
                print('pert_regime_mapping: ', self.perturbation_to_regime)
                regimes = []
                for sample in intervention_mask:
                    # check if sample is in the perturbation_to_regime dict
                    if tuple(sample) in self.perturbation_to_regime:
                        regimes.append(
                            self.perturbation_to_regime[tuple(sample)])
                    else:
                        regimes.append(0)
                regimes = np.array(regimes)
                # print("regimes: ", regimes[::100])

                test_data = DataManagerDirect(
                    data,
                    intervention_mask,
                    regimes,
                    size_factors=sf,
                    train_samples=0.1,
                    train=False,
                    normalize=False,
                    random_seed=self.opt.random_seed,
                    intervention=True,
                    intervention_knowledge="known",
                )
            x, size_fact, mask, regime = test_data.sample(test_data.num_samples)
            nll_val = compute_loss(x, size_fact, mask, regime, model, weights, biases,
                                   extra_params, opt.intervention,
                                   opt.intervention_type,
                                   opt.intervention_knowledge)

        return nll_val.cpu().numpy()
    
    def compute_forward_pass(self, adata_obj=None) -> np.array:
        """Compute the forward pass of the model on a given dataset.

        Args:
            None

        Returns:
            np.array: forward pass of the model
        """
        opt = self.opt
        test_data = self.test_data
        model = self.model
        weights, biases, extra_params = model.get_parameters(mode="wbx")

        # compute loss on whole validation set
        with torch.no_grad():
            # if an external dataset is provided => convert it to the right format
            if adata_obj is not None:
                data, intervention_mask, _, size_fact_sorted = self.get_data(
                    adata_obj, set_regimes=False)
                # print(data, intervention_mask)
                print('pert_regime_mapping: ', self.perturbation_to_regime)
                regimes = []
                for sample in intervention_mask:
                    # check if sample is in the perturbation_to_regime dict
                    if tuple(sample) in self.perturbation_to_regime:
                        regimes.append(
                            self.perturbation_to_regime[tuple(sample)])
                    else:
                        regimes.append(0)
                regimes = np.array(regimes)
                # print("regimes: ", regimes[::100])

                test_data = DataManagerDirect(
                    data,
                    intervention_mask,
                    regimes,
                    size_factors=size_fact_sorted,
                    train_samples=0.1,
                    train=False,
                    normalize=False,
                    random_seed=self.opt.random_seed,
                    intervention=True,
                    intervention_knowledge="known",
                )
            x, size_fact, mask, regime = test_data.sample(test_data.num_samples)
            density_params = model.forward_given_params(
                x, weights, biases, mask=mask, regime=regime)
        
        #return param.cpu().numpy()
        return tuple(tx.cpu() for tx in density_params)


    def save_model(self, filename="dcdi_model"):
        dump(self.model, self.output_dir, filename)
        if self.lock_sigmoid:
            np.savetxt(self.output_dir + "/dcdi_sigmoid_params.txt", [self.sig_k, self.sig_b])

    def load_model(self, filename="dcdi_model.pkl"):
        """Load Model from saved file, make sure the dimensions of the saved model match the current model dimensions (num_genes, layers, hidden_dim, num_regimes)"""
        self.model = load(self.output_dir, filename)
        return self.model
