"""Inherited Data Manager class (from dcdi DataManagerFile) to load data from anndata object"""

import torch
import numpy as np


from scp_infer.thirdparty.dcdi_local.data import DataManagerFile 
#from scp_infer.thirdparty.dcdi_local.utils.save import dump, load
# pyright: reportMissingImports=false 



class DataManagerDirect(DataManagerFile):
    """
    Identicial to DataManagerFile, but skips loading data from file and has it provided as arguments.
    Plus a few changes:
        - no ground truth adjacency matrix
        - convert masks to numpy array to avoid pytorch error
    """
    def __init__(
            self, 
            data, 
            masks, 
            regimes, 
            size_factors = None,
            train_samples=0.8, 
            test_samples=None, 
            train=True,
            normalize=False, 
            mean=None, 
            std=None, 
            random_seed=42, 
            intervention=False,
            intervention_knowledge="known", 
            dcd=False,
            regimes_to_ignore=None
            ):
        """
        :param str file_path: Path to the data and the DAG
        :param int i_dataset: Exemplar to use (usually in [1,10])
        :param float/int train_samples: default=0.8. If float, specifies the proportion of
            data used for training and the rest is used for testing. If an integer, specifies
            the exact number of examples to use for training.
        :param int test_samples: default=None. Specifies the number of examples to use for testing.
            The default value uses all examples that are not used for training.
        :param int random_seed: Random seed to use for data set shuffling and splitting
        :param boolean intervention: If True, use interventional data with interventional targets
        :param str intervention_knowledge: Determine if the intervention target are known or unknown
        :param boolean dcd: If True, use the baseline DCD that use interventional data, but
            with a loss that doesn't take it into account (intervention should be set to False)
        :param list regimes_to_ignore: Regimes that are ignored during training
        """
        self.random = np.random.RandomState(random_seed)
        self.dcd = dcd
        self.intervention = intervention
        if intervention_knowledge == "known":
            self.interv_known = True
        elif intervention_knowledge == "unknown":
            self.interv_known = False
        else:
            raise ValueError("intervention_knowledge should either be 'known' \
                             or 'unknown'")

        #set empty gt adjacency matrix
        adjacency = np.zeros((data.shape[1], data.shape[1]))
        self.adjacency = torch.as_tensor(adjacency).type(torch.Tensor)

        # index of all regimes, even if not used in the regimes_to_ignore case
        self.all_regimes = np.unique(regimes)

        # Remove some regimes
        if regimes_to_ignore is not None and self.intervention:
            for regime_to_ignore in regimes_to_ignore:
                if regime_to_ignore not in self.all_regimes:
                    raise ValueError(f"Regime {regime_to_ignore} is not in the possible regimes: {self.all_regimes}")
                to_keep = (np.array(regimes) != regime_to_ignore)
                data = data[to_keep]
                size_factors = size_factors[to_keep]
                masks = [mask for i, mask in enumerate(masks) if to_keep[i]]
                regimes = np.array([regime for i, regime in enumerate(regimes) if to_keep[i]])

        # Determine train/test partitioning
        if isinstance(train_samples, float):
            train_samples = int(data.shape[0] * train_samples)
        if test_samples is None:
            test_samples = data.shape[0] - train_samples
        assert train_samples + test_samples <= data.shape[0], "The number of examples to load must be " + \
            "smaller than the total size of the dataset"

        # Shuffle and filter examples
        shuffle_idx = np.arange(data.shape[0])
        self.random.shuffle(shuffle_idx)
        data = data[shuffle_idx[: train_samples + test_samples]]
        size_factors = size_factors[shuffle_idx[: train_samples + test_samples]]
        if intervention:
            masks = [masks[i] for i in shuffle_idx[: train_samples + test_samples]]
        regimes = regimes[shuffle_idx[: train_samples + test_samples]]

        # Train/test split
        if not train:
            if train_samples == data.shape[0]: # i.e. no test set
                self.dataset = None
                self.masks = None
                self.regimes = None
            else:
                self.dataset = torch.as_tensor(data[train_samples: train_samples + test_samples]).type(torch.Tensor)
                self.size_factors = torch.as_tensor(size_factors[train_samples: train_samples + test_samples]).type(torch.Tensor)
                if intervention:
                    self.masks = masks[train_samples: train_samples + test_samples]
                self.regimes = regimes[train_samples: train_samples + test_samples]
        else:
            self.dataset = torch.as_tensor(data[: train_samples]).type(torch.Tensor)
            self.size_factors = torch.as_tensor(size_factors[: train_samples]).type(torch.Tensor)
            if intervention:
                self.masks = masks[: train_samples]
            self.regimes = regimes[: train_samples]

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if self.mean is None or self.std is None:
                self.mean = torch.mean(self.dataset, 0, keepdim=True)
                self.std = torch.std(self.dataset, 0, keepdim=True)
            self.dataset = (self.dataset - self.mean) / self.std

        self.num_regimes = np.unique(self.regimes).shape[0]
        self.num_samples = self.dataset.size(0)
        self.dim = self.dataset.size(1)

        self.initialize_interv_matrix()

    def initialize_interv_matrix(self):
        """
        !TAKEN FROM DataManagerFile!
        Generate the intervention matrix I*. It is useful in the unknown case
        to compare learned target to the ground truth

        added to the original code: .cpu.numpy() to convert the tensor to numpy array
        """
        if self.intervention:
            interv_matrix = np.zeros((self.dataset.shape[1], self.num_regimes))

            regimes = np.sort(np.unique(self.regimes))
            for i, regime in enumerate(regimes):
                mask_idx = np.where(self.regimes == regime)[0][0]
                interv_matrix[:, i] = self.convert_masks(np.array([mask_idx])).cpu().numpy()

            self.gt_interv = 1 - interv_matrix
        else:
            self.gt_interv = None

    def sample(self, batch_size):
        """
        !TAKEN FROM DataManagerFile!
        Sample without replacement `batch_size` examples from the data and
        return the corresponding masks and regimes
        :param int batch_size: number of samples to sample
        :return: samples, masks, regimes

        added to the original code: .cpu() to convert the tensor to numpy array
        """
        sample_idxs = self.random.choice(np.arange(int(self.num_samples)), size=(int(batch_size),), replace=False)
        samples = self.dataset[torch.as_tensor(sample_idxs).long()]
        size_samples = self.size_factors[torch.as_tensor(sample_idxs).long()]
        if self.intervention:
            masks = self.convert_masks(sample_idxs)
            regimes = self.regimes[torch.as_tensor(sample_idxs).long().cpu()]
        else:
            masks = torch.ones_like(samples)
            regimes = None
        return samples, size_samples, masks, regimes