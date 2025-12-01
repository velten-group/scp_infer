"""Application of Bicycle algorithm to infer GRN from single-cell data. (local implementation)


"""

import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import argparse
import sys
import os
from pathlib import Path

from ..inference_method import InferenceMethod
from scp_infer.adata import convert_onehot_to_indices



try:
    import pytorch_lightning as pl
    import torch
    import yaml

    import scp_infer.thirdparty.bicycle_local as bicycle
    from scp_infer.thirdparty.bicycle_local.callbacks import (
        CustomModelCheckpoint,
        GenerateCallback,
        MyLoggerCallback,
    )
    from scp_infer.thirdparty.bicycle_local.dictlogger import DictLogger
    from scp_infer.thirdparty.bicycle_local.model import BICYCLE
    from scp_infer.thirdparty.bicycle_local.utils.data import (
        compute_inits,
        create_data,
        create_loaders,
        get_diagonal_mask,
    )
    from scp_infer.thirdparty.bicycle_local.utils.general import get_full_name
    from scp_infer.thirdparty.bicycle_local.utils.plotting import plot_training_results
    from pytorch_lightning.callbacks import RichProgressBar, StochasticWeightAveraging
    from pytorch_lightning.tuner.tuning import Tuner
except ImportError:
    print("Error importing Bicycle. Bicycle dependencies might not be installed.")

# import the local dcdi algorithm
# pyright: reportMissingImports=false


def _print_metrics(stage, step, metrics, throttle=None):
    for k, v in metrics.items():
        print("    %s:" % k, v)


def file_exists(prefix, suffix):
    return os.path.exists(os.path.join(prefix, suffix))


class BicycleImp(InferenceMethod):
    """
    Bicycle implementation

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
    max_features = 70

    def __init__(
            self,
            adata_obj,
            output_dir: str = None,
            verbose: bool = False,
            file_label: str = "BICYCLE",
            validation_size: float = 0.2,
            scale_kl: float = 1.0,
            scale_l1: float = 1.0,
            scale_spectral: float = 0.0,
            scale_lyapunov: float = 0.1,
            n_epochs: int = 60000,
            n_epochs_pretrain_latents: int = 10000,
            use_raw: bool = True,
    ):
        """Initialize the Bicycle implementation, set all default Hyperparameters.
        """

        super(BicycleImp, self).__init__(adata_obj, output_dir, verbose)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_PATH = Path(os.path.join(output_dir, "models"))
        self.PLOT_PATH = Path(os.path.join(output_dir, "plots"))
        self.MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self.PLOT_PATH.mkdir(parents=True, exist_ok=True)
        #
        # Settings
        #
        n_factors = 0
        intervention_type_simulation = "Cas9"
        intervention_type_inference = "Cas9"
        self.n_genes = adata_obj.n_vars  # Number of modelled genes
        self.rank_w_cov_factor = self.n_genes  # Same as dictys: #min(TFs, N_GENES-1)
        graph_type = "erdos-renyi"
        edge_assignment = "random-uniform"
        make_counts = True
        synthetic_T = 1.0
        self.n_contexts = self.n_genes + 1  # Number of contexts
        n_samples_control = 500
        n_samples_per_perturbation = 250
        self.perfect_interventions = True

        # TRAINING
        self.lr = 1e-3
        self.batch_size = 1024
        self.USE_INITS = False
        self.use_encoder = False
        self.n_epochs = n_epochs
        self.n_epochs_pretrain_latents = n_epochs_pretrain_latents
        self.early_stopping = False
        self.early_stopping_patience = 500
        self.early_stopping_min_delta = 0.01
        self.optimizer = "adam"  # "rmsprop" #"adam"
        # Faster decay for estimates of gradient and gradient squared
        self.gradient_clip_val = 1.0
        self.swa = 0 # NOTE: 250? 500?
        self.x_distribution = "Multinomial"  # "Poisson" if "random" in graph else
        self.intervention_type = "Cas9"
        self.optimizer_kwargs = {"betas": [0.5, 0.9]}
        self.plot_epoch_callback = 500
        # DATA
        self.validation_size = validation_size
        self.use_raw = use_raw  # Use raw data for training
        # MODEL
        self.lyapunov_penalty = True
        self.GPU_DEVICE = 0
        self.plot_epoch_callback = 500
        self.use_latents = True

        # HYPERPARAMETERS
        self.scale_kl = scale_kl
        self.scale_l1 = scale_l1
        self.scale_spectral = scale_spectral
        self.scale_lyapunov = scale_lyapunov # 0.1, 1, 10, 100

        LOGO = []
        # We start counting at 0
        self.train_gene_ko = [str(x) for x in set(range(0, self.n_genes)) - set(LOGO)]
        # FIXME: There might be duplicates...
        ho_perturbations = sorted(
            list(set([tuple(sorted(np.random.choice(self.n_genes, 2, replace=False)))
                 for _ in range(0, 20)]))
        )
        self.test_gene_ko = [f"{x[0]},{x[1]}" for x in ho_perturbations]

        # MODEL

        # RESULTS
        self.name_prefix = f"IFN_Epochs_{self.n_epochs}_Pretrainepochs_{self.n_epochs_pretrain_latents}_Encoder_{self.use_encoder}_{self.optimizer}_{self.batch_size}"
        self.SAVE_PLOT = True
        self.CHECKPOINTING = True
        self.VERBOSE_CHECKPOINTING = True
        self.OVERWRITE = True
        # REST
        n_samples_total = n_samples_control + \
            (len(self.train_gene_ko) + len(self.test_gene_ko)) * n_samples_per_perturbation
        self.check_val_every_n_epoch = 1
        self.log_every_n_steps = 1

        # Create Mask
        self.mask = get_diagonal_mask(self.n_genes, self.device)

        if n_factors > 0:
            self.mask = None

    def get_data(self, adata_obj, set_regimes=True):
        """Extract the data from the adata object and return it in the format required by the DCDI algorithm.

        ToDo:
            - figure out train-test split! how does bicycle use it? do interventions have to be disjoint?
            - data transformation - what if not all genes have interventions?

        Args:
            adata_obj: Annotated expression data object from scanpy

        Returns:
            Subset training and test/validation data???

        """
        #data = sp.csr_matrix.toarray(self.adata_obj.X)
        if self.use_raw:
            data = self.adata_obj.layers['counts']
        else:
            data = self.adata_obj.X

        final_genes = adata_obj.var.index

        datasets = []

        # fix adata annotation
        adata_obj.obs["targets"] = adata_obj.obs["perturbation"]

        for gene in final_genes:
            datasets.append(data[adata_obj.obs["targets"] == gene, :])

        intervention_sets = [[i] for i in range(len(final_genes))]

        # datasets = list of datasets for each gene
        # intervention_sets = list of respective gene indices
        # => concatenate

        dataset_train = np.concatenate(datasets, axis=0)
        dataset_train_targets = np.concatenate(
            [np.ones(datasets[k].shape[0]) * i for k, i in enumerate(intervention_sets)]
        )

        # dataset_train: gene exression matrix
        # dataset_train_targets: gene indices of perturbations
        samples = dataset_train
        sim_regimes = dataset_train_targets
        
        # Convert to torch
        samples = torch.from_numpy(samples).float()
        sim_regimes = torch.from_numpy(sim_regimes).long()
        
        return samples, sim_regimes

    def convert_data(self):
        """Convert the data to the format required by the Bicycle algorithm.

        samples: all samples - n_samples x n_genes
        sim_regime: integer regime ID for each sample - n_samples
        validation_size: fraction of samples to use for validation - float
        batch_size: batch size for training - int
        SEED: random seed - int
        train_gene_ko: list of genes


        """

        SEED = 1
        gt_dyn = None
        intervened_variables = None
        #gt_interv = self.adata_obj.layers['perturbed_elem_mask']
        sim_regime = None       #have to creat regime IDs
        beta = None

        samples, sim_regimes = self.get_data(self.adata_obj)
        interv = [[i] for i in range(self.n_genes)]



        # NODAGS: use perturbations only - 6 genes validation, rest training
        # Perturbations of Train and Test have to be disjoint?
        # ToDo: figure out what the test set is used for & if i can include ko of the same gene in train and test
        train_loader, validation_loader, test_loader = create_loaders(
            samples,
            sim_regimes,
            validation_size=self.validation_size,
            batch_size=1024,
            SEED=0,
            train_gene_ko=[x[0] for x in interv[:-6]],   # list of lists: indices of genes in each perturbation
            test_gene_ko=[x[0] for x in interv[-6:]],  # list of lists: indices of genes in each perturbation
            num_workers=1,
            persistent_workers=False,
            # prefetch_factor=4,
        )
        self.train_loader = train_loader
        self.validation_loader = validation_loader



        covariates = None

        gt_interv = torch.tensor(np.eye(self.n_genes), dtype=torch.float32)

        if self.USE_INITS:
            init_tensors = compute_inits(train_loader.dataset, self.rank_w_cov_factor, self.n_contexts)
        
        print(f"Number of training samples: {len(train_loader.dataset)}")
        if self.validation_size > 0:
            print(f"Number of validation samples: {len(validation_loader.dataset)}")
        print(f"Number of test samples: {len(test_loader.dataset)}")

        #gt_interv = torch.zeros((self.n_genes, self.n_genes + 1), device=device)
        gt_interv = gt_interv.to(self.device)

        self.file_dir = (
            self.name_prefix
            + f"_{SEED}_{self.gradient_clip_val}_{self.swa}_{self.USE_INITS}_{self.lr}_{self.scale_l1}_{self.scale_kl}_{self.scale_spectral}_{self.scale_lyapunov}"
        )
        print('file dir: ',self.file_dir)

        # If final plot or final model exists: do not overwrite by default
        final_file_name = os.path.join(self.MODEL_PATH, self.file_dir, "last.ckpt")
        final_plot_name = os.path.join(self.PLOT_PATH, self.file_dir, "last.png")
        if (Path(final_file_name).exists() & self.SAVE_PLOT & ~self.OVERWRITE) | (
            Path(final_plot_name).exists() & self.CHECKPOINTING & ~self.OVERWRITE
        ):
            print("Files already exists, skipping...")
            return None
        else:
            print("Files do not exist, fitting model...")
            print("Deleting dirs")
            # Delete directories of files
            if Path(final_file_name).exists():
                print(f"Deleting {final_file_name}")
                # Delete all files in os.path.join(MODEL_PATH, file_name)
                for f in os.listdir(os.path.join(self.MODEL_PATH, self.file_dir)):
                    os.remove(os.path.join(self.MODEL_PATH, self.file_dir, f))
            if Path(final_plot_name).exists():
                print(f"Deleting {final_plot_name}")
                for f in os.listdir(os.path.join(self.PLOT_PATH, self.file_dir)):
                    os.remove(os.path.join(self.PLOT_PATH, self.file_dir, f))

            print("Creating dirs")
            # Create directories
            Path(os.path.join(self.MODEL_PATH, self.file_dir)).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.PLOT_PATH, self.file_dir)).mkdir(parents=True, exist_ok=True)


        self.final_plot_name = final_plot_name
        """
        model = BICYCLE(
            self.lr,
            gt_interv,
            self.n_genes,
            n_samples_control=self.n_samples_control,
            lyapunov_penalty=self.lyapunov_penalty,
            perfect_interventions=self.perfect_interventions,
            rank_w_cov_factor=self.rank_w_cov_factor,
            init_tensors=self.init_tensors if self.USE_INITS else None,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            device=self.device,
            scale_l1=self.scale_l1,
            scale_lyapunov=self.scale_lyapunov,
            scale_spectral=self.scale_spectral,
            scale_kl=self.scale_kl,
            early_stopping=self.early_stopping,
            early_stopping_min_delta=self.early_stopping_min_delta,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_p_mode=True,
            x_distribution=self.x_distribution,
            x_distribution_kwargs=self.x_distribution_kwargs,
            mask=self.mask,
            use_encoder=self.use_encoder,
            gt_beta=self.beta,
            train_gene_ko=self.train_gene_ko,
            test_gene_ko=self.test_gene_ko,
            use_latents=self.use_latents,
            covariates=self.covariates,
            n_factors=self.n_factors,
            intervention_type=self.intervention_type_inference,
            T=self.model_T,
            learn_T=self.learn_T,
        )"""

        model = BICYCLE(
                self.lr,
                gt_interv,
                self.n_genes,
                n_samples=len(train_loader.dataset),
                lyapunov_penalty=self.lyapunov_penalty,
                perfect_interventions=self.perfect_interventions,
                rank_w_cov_factor=self.rank_w_cov_factor,
                init_tensors=self.init_tensors if self.USE_INITS else None,
                optimizer=self.optimizer,
                device=self.device,
                scale_l1=self.scale_l1,
                scale_lyapunov=self.scale_lyapunov,
                scale_spectral=self.scale_spectral,
                scale_kl=self.scale_kl,
                early_stopping=self.early_stopping,
                early_stopping_min_delta=self.early_stopping_min_delta,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_p_mode=True,
                x_distribution=self.x_distribution,
                mask=self.mask,
                use_encoder=self.use_encoder,
                gt_beta=None,
                use_latents=self.use_latents,
                intervention_type=self.intervention_type
        )
        model.to(self.device)
        self.model = model

        dlogger = DictLogger()
        self.loggers = [dlogger]

        callbacks = [
            RichProgressBar(refresh_rate=1),
            GenerateCallback(
                final_plot_name,
                plot_epoch_callback=self.plot_epoch_callback,
                true_beta=None,
                labels=self.adata_obj.var.index.values,
            ),
        ]
        if self.swa > 0:
            callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=self.swa))
        if self.CHECKPOINTING:
            Path(os.path.join(self.MODEL_PATH, self.file_dir)).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CustomModelCheckpoint(
                    dirpath=os.path.join(self.MODEL_PATH, self.file_dir),
                    filename="{epoch}",
                    save_last=True,
                    save_top_k=-1,
                    verbose=self.VERBOSE_CHECKPOINTING,
                    monitor="valid_loss", ### FIXME: valid_loss (before: train_loss)
                    mode="min",
                    save_weights_only=True,
                    start_after=0,
                    save_on_train_epoch_end=False, ### FIXME: False
                    every_n_epochs=1000,
                )
            )
            callbacks.append(MyLoggerCallback(dirpath=os.path.join(self.MODEL_PATH, self.file_dir)))
        self.callbacks = callbacks

    def infer(
        self,
        save_output: bool = True,
        save_time: bool = True,
        file_label: str = "Bicycle",
        **kwargs
    ) -> np.array:
        """
        Parameters for the Bicycle algorithm

        store parameters as attributes of opt
        """

        model = self.model

        if self.verbose:
            print("Running Bicycle")

        # print("train_data.regimes.device:", train_data.regimes.device)

        # start timer and memory tracking
        # add GPU memory tracking? - torch.cuda.memory_allocated()
        # tracemalloc.start()
        start_time = time.time()

        # train until constraint is sufficiently close to being satisfied
        if self.verbose:
            print("Training model")
        

        ######################################################
        # TRAINING
        trainer = pl.Trainer(
            max_epochs=self.n_epochs,
            accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
            logger=self.loggers,
            log_every_n_steps=self.log_every_n_steps,
            enable_model_summary=True,
            enable_progress_bar=True,
            enable_checkpointing=self.CHECKPOINTING,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            devices=[self.GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
            num_sanity_val_steps=0,
            callbacks=self.callbacks,
            gradient_clip_val=self.gradient_clip_val,
            default_root_dir=str(self.MODEL_PATH),
            gradient_clip_algorithm="value",
        )

        if self.use_latents and self.n_epochs_pretrain_latents > 0:
            pretrain_callbacks = [
                RichProgressBar(refresh_rate=1),
                GenerateCallback(
                    str(Path(self.final_plot_name).with_suffix("")) + "_pretrain",
                    plot_epoch_callback=self.plot_epoch_callback,
                    true_beta=None,
                    labels=self.adata_obj.var.index.values ## labels = gene-names ?
                ),
            ]

            if self.swa > 0:
                pretrain_callbacks.append(StochasticWeightAveraging(0.01, swa_epoch_start=self.swa))
            pretrain_callbacks.append(MyLoggerCallback(dirpath=os.path.join(self.MODEL_PATH, self.file_dir)))

            pretrainer = pl.Trainer(
                max_epochs=self.n_epochs_pretrain_latents,
                accelerator="gpu",  # if str(device).startswith("cuda") else "cpu",
                logger=self.loggers,
                log_every_n_steps=self.log_every_n_steps,
                enable_model_summary=True,
                enable_progress_bar=True,
                enable_checkpointing=self.CHECKPOINTING,
                check_val_every_n_epoch=self.check_val_every_n_epoch,
                devices=[self.GPU_DEVICE],  # if str(device).startswith("cuda") else 1,
                num_sanity_val_steps=0,
                callbacks=pretrain_callbacks,
                gradient_clip_val=self.gradient_clip_val,
                default_root_dir=str(self.MODEL_PATH),
                gradient_clip_algorithm="value"
            )

            print("PRETRAINING LATENTS!")
            start_time = time.time()
            model.train_only_likelihood = True
            pretrainer.fit(model, self.train_loader, self.validation_loader)
            end_time = time.time()
            model.train_only_likelihood = False

        #try:
        trainer.fit(model, self.train_loader, self.validation_loader)
        end_time = time.time()
        print(f"Training took {end_time - start_time:.2f} seconds")

        plot_training_results(
            trainer,
            model,
            model.beta.detach().cpu().numpy(),
            None,
            self.scale_l1,
            self.scale_kl,
            self.scale_spectral,
            self.scale_lyapunov,
            self.final_plot_name,
            callback=False,
        )
        ##############################################3

        # end timer
        end_time = time.time()
        total_time = end_time - start_time
        self.runtime = total_time

        # save cuda max memory
        self.memory_usage = torch.cuda.max_memory_allocated()  # in bytes

        if self.verbose:
            print("Bicycle finished")
            print("Time taken: ", total_time)
            print("Memory usage: ", self.memory_usage)

        adjacency = self.model.beta.detach().cpu().numpy()

        self.adjacency_matrix = adjacency
        super(BicycleImp, self).save_output(file_label)

        return self.adjacency_matrix, total_time

    def compute_loss(self, adata_obj=None) -> float:
        """Compute the loss of the model on the whole validation set.

        Args:
            None

        Returns:
            float: negative log likelihood of the model
        """
        raise NotImplementedError
    
    def save_model(self, filename="Bicycle_model"):
        raise NotImplementedError

    def load_model(self, filename="Bicycle_model.pkl"):
        """Load Model from saved file, make sure the dimensions of the saved model match the current model dimensions (num_genes, layers, hidden_dim, num_regimes)"""
        raise NotImplementedError
