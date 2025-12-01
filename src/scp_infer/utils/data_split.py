"""Splitting data into train, validation, and test sets - bzw. into train and hold-out"""
import numpy as np
from anndata import AnnData
from sklearn.model_selection import StratifiedShuffleSplit


def shuffled_split(
        adata_obj: AnnData,
        # train_frac: float = 0.8,
        # val_frac: float = 0.15,
        test_frac: float = 0.2,
        seed: int = 42,
        verbose=False
) -> None:
    """Split the data into random train, validation, and test sets.

    Args
        adata_obj: AnnData object
        train_frac: float, fraction of the data to be used for training
        val_frac: float, fraction of the data to be used for validation
        test_frac: float, fraction of the data to be used for testing
        seed: int, random seed
    """
    np.random.seed(seed)
    n_obs = adata_obj.n_obs
    n_test = int(n_obs * test_frac)
    n_train = n_obs - n_test

    set_list = ['train'] * n_train + ['test'] * n_test
    np.random.shuffle(set_list)
    adata_obj.obs['set'] = set_list
    adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')
    if verbose:
        print("Train:", n_train)
        # print("Validation:", n_val)
        print("Test:", n_test)
    return None


def shuffled_split_proportion(
        adata_obj: AnnData,
        # train_frac: float = 0.8,
        # val_frac: float = 0.15,
        test_frac: float = 0.2,
        seed: int = 42,
        verbose=False
) -> None:
    """Split the data into random train, validation, and test sets.
    Enforce that the proportions of the splits are kept across the different perturbations.

    Args
        adata_obj: AnnData object
        train_frac: float, fraction of the data to be used for training
        val_frac: float, fraction of the data to be used for validation
        test_frac: float, fraction of the data to be used for testing
        seed: int, random seed
    """
    perturbations = adata_obj.obs['perturbation'].unique()
    np.random.seed(seed)
    adata_obj.obs['set'] = 'train'

    for perturbation in perturbations:
        n_obs = adata_obj[adata_obj.obs['perturbation'] == perturbation].n_obs
        n_test = int(n_obs * test_frac)
        n_train = n_obs - n_test

        set_list = ['train'] * n_train + ['test'] * n_test
        np.random.shuffle(set_list)
        adata_obj.obs.loc[adata_obj.obs['perturbation']
                          == perturbation, 'set'] = set_list

    adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')
    if verbose:
        print("Train:", n_train)
        # print("Validation:", n_val)
        print("Test:", n_test)


def gene_holdout(
        adata_obj: AnnData,
        hold_out_gene: str,
) -> None:
    """Hold out the data for a specific gene perturbation

    Stores the respective assignment in the observation annotation "set"

    Args:
        adata_obj: AnnData object
        hold_out_gene: str, gene perturbation to be held out
    """
    adata_obj.obs['set'] = 'train'
    adata_obj.obs.loc[adata_obj.obs['perturbation']
                      == hold_out_gene, 'set'] = 'test'
    adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')
    return None

def gene_holdout_stratified(
        adata_obj: AnnData,
        hold_out_gene: str,
        test_frac: float = 0.2,
        seed: int = 0,
        verbose=False
) -> None:
    """Hold out the data for specific gene perturbations and stratified portion of all other perturbations"""
    adata_obj.obs['set'] = 'train'
    np.random.seed(seed)
    n_obs = adata_obj.n_obs
    n_test = int(n_obs * test_frac)
    n_train = n_obs - n_test

    # stratified split
    if test_frac == 0.0:
        pass
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_test, random_state=seed)
        for train_index, test_index in sss.split(adata_obj.X, adata_obj.obs['perturbation']):
            adata_obj.obs.loc[adata_obj.obs.index[test_index], 'set'] = 'test'
            adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')

    # hold-out gene to test set    
    adata_obj.obs.loc[adata_obj.obs['perturbation'] == hold_out_gene, 'set'] = 'test'
    adata_obj.obs['set'] = adata_obj.obs['set'].astype('category')
    if verbose:
        print("Train:", adata_obj[adata_obj.obs['set'] == 'train'].obs['perturbation'].value_counts())
        print("Test:", adata_obj[adata_obj.obs['set'] == 'test'].obs['perturbation'].value_counts())
    return None    


def intervention_proportion_split(
        adata: AnnData,
        interv_frac: float = 0.2,
        sample_frac: float = 0.2,
        seed=0,
        verbose = False,
):
    """Create a validation train split for a given dataset.

    The Validation set is to contain:

        - complete samples for ~interv_frac (rounded) of interventions (i.e. pertubrations that aren't non-targeting)
        - sample_frac % of the samples from each regime/intervention

    Args:
        adata: Annotated expression data object from scanpy
            should be fully preprocessed and ready for inference
        interv_frac: Fraction of interventions use for validation
        sample_frac: Fraction of samples to use for validation
        seed: Random seed for reproducibility

    Returns:
        tuple: tuple containing:

            - adata_train (AnnData): Annotated expression data object for training
            - adata_val (AnnData): Annotated expression data object for validation
    """
    np.random.seed(seed)
    adata_val = adata.copy()
    adata_train = adata.copy()

    # Compute the interventions to hold out
    n_interv = len(adata.obs['perturbation'].unique())-1 # -1 to remove non-targeting
    n_val_interv = int(np.round(interv_frac * n_interv)) 
    interv_list = adata.obs['perturbation'].unique().tolist()
    np.random.shuffle(interv_list)
    pert_interv_list = [i for i in interv_list if i != 'non-targeting'] # remove non-targeting from the list
    interv_val = pert_interv_list[:n_val_interv]
    interv_train = pert_interv_list[n_val_interv:]

    # Assign the interventions to their sets
    adata.obs['set'] = 'train'
    adata.obs.loc[adata.obs['perturbation'].isin(interv_val), 'set'] = 'val'

    # Compute the samples to hold out per included intervention
    for intervention in interv_list:
        if intervention in interv_val:
            continue
        sample_locs = adata[adata.obs['perturbation'] == intervention].obs.index
        n_samples = len(sample_locs)
        n_val_samples = int(np.round(sample_frac * n_samples))
        # Randomly select the samples to hold out
        val_sample_locs = np.random.choice(sample_locs, n_val_samples, replace=False)
        adata.obs.loc[sample_locs, 'set'] = 'train'
        adata.obs.loc[val_sample_locs, 'set'] = 'val'
    
    adata.obs['set'] = adata.obs['set'].astype('category')

    # Split the data
    adata_train = adata[adata.obs['set'] == 'train']
    adata_val = adata[adata.obs['set'] == 'val']

    if verbose:
        print("Train:", adata_train.n_obs)
        print("Validation:", adata_val.n_obs)
        print("train intervention counts:", adata_train.obs['perturbation'].value_counts())
        print("val intervention counts:", adata_val.obs['perturbation'].value_counts())
    return adata_train, adata_val
