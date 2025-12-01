import sys
import pytest
import numpy as np
from anndata import AnnData

sys.path.append("src")
sys.path.append("src/algorithm_implementations")
from scp_infer.adata.filter_adata import convert_onehot_to_indices, convert_indices_to_onehot 
# pyright: reportMissingImports=false 

def adata_from_onehot(onehot):
    adata = AnnData(
            X = np.random.rand(np.shape(onehot)[0], np.shape(onehot)[1]),
            obs = {"celltype": ["A","B","C"]},
            var = {"gene": ["a","b","c"]}
    )
    adata.layers['perturbed_elem_mask'] = onehot
    return adata

def test_convert_onehot_to_indices():
    onehot = np.array([[0,0,1],[0,1,0],[1,0,0]])
    adata = adata_from_onehot(onehot)
    indices = np.array([[2],[1],[0]])
    assert np.array_equal(convert_onehot_to_indices(adata), indices)

    onehot = np.array([[0,0,0],[0,0,0],[0,0,0]])
    indices = np.array([[],[],[]])
    adata = adata_from_onehot(onehot)
    assert np.array_equal(convert_onehot_to_indices(adata), indices)

    onehot = np.array([[1,1,1],[1,1,1],[1,1,1]])
    adata = adata_from_onehot(onehot)
    indices = np.array([[0,1,2],[0,1,2],[0,1,2]])
    assert np.array_equal(convert_onehot_to_indices(adata), indices)
