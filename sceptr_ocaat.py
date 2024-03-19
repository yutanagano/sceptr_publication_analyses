import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sceptr._lib.sceptr import Sceptr
import torch
from torch import FloatTensor


class SceptrOCAAT:
    def __init__(self, sceptr: Sceptr):
        self._model = sceptr
        self._device = sceptr._device

    @property
    def name(self) -> str:
        return f"{self._model.name} - OCAAT"

    def calc_cdist_matrix(self, anchors: DataFrame, comparisons: DataFrame) -> ndarray:
        anchor_representations = self._calc_torch_representations(anchors)
        comparison_representations = self._calc_torch_representations(comparisons)
        return torch.cdist(anchor_representations, comparison_representations, p=2).cpu().numpy()
    
    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        vector_representations = self._calc_torch_representations(instances)
        return torch.pdist(vector_representations, p=2).cpu().numpy()

    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        return self._calc_torch_representations(instances).cpu().numpy()
    
    def _calc_torch_representations(self, instances: DataFrame) -> FloatTensor:
        alpha_representations = self._model.calc_vector_representations(instances[["TRAV", "CDR3A"]])
        beta_representations = self._model.calc_vector_representations(instances[["TRBV", "CDR3B"]])
        representations_concatenated = np.concatenate([alpha_representations, beta_representations], axis=1)
        return torch.tensor(representations_concatenated, device=self._device)