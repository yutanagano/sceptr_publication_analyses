import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
from paths import CACHE_DIR
import pickle
import torch
from torch import FloatTensor
from typing import Dict, Optional, Tuple


class CachedRepresentationModel():
    def __init__(self, model) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        self._model = model
        self._cache_save_path = CACHE_DIR/f"{model.name}_rep_cache.pkl"
        self._cache = self._load_cahce()
    
    def _load_cahce(self) -> Dict[Tuple[Optional[str]], FloatTensor]:
        if not self._cache_save_path.is_file():
            return dict()
        
        with open(self._cache_save_path, "rb") as f:
            return pickle.load(f)
    
    def save_cache(self) -> None:
        with open(self._cache_save_path, "wb") as f:
            pickle.dump(self._cache, f)

    @property
    def name(self) -> str:
        return self._model.name

    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        return self._calc_torch_representations(instances).cpu().numpy()
    
    def calc_cdist_matrix(self, anchors: DataFrame, comparisons: DataFrame) -> ndarray:
        anchor_representations = self._calc_torch_representations(anchors)
        comparison_representations = self._calc_torch_representations(comparisons)
        cdist_tensor = torch.cdist(anchor_representations, comparison_representations, p=2)
        return cdist_tensor.cpu().numpy()
    
    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        representations = self._calc_torch_representations(instances)
        pdist_tensor = torch.pdist(representations, p=2)
        return pdist_tensor.cpu().numpy()

    def _calc_torch_representations(self, instances: DataFrame) -> FloatTensor:
        tcr_identifiers = instances.apply(self._generate_tcr_identifier, axis=1)
        tcr_unseen = tcr_identifiers.map(lambda tcr_id: tcr_id not in self._cache)

        if tcr_unseen.sum() > 0:
            self._compute_and_cache_representations(instances[tcr_unseen])

        def fetch_torch_representation(tcr_id) -> FloatTensor:
            return torch.tensor(self._cache[tcr_id], device=self._device)

        representations = tcr_identifiers.map(lambda tcr_id: self._cache[tcr_id]).to_list()
        representations_stacked = np.stack(representations)
        return torch.tensor(representations_stacked, device=self._device)
    
    def _compute_and_cache_representations(self, instances: DataFrame) -> None:
        instances = instances.drop_duplicates(subset=["TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"])
        representations = self._model.calc_vector_representations(instances)
        tcr_ids = instances.apply(self._generate_tcr_identifier, axis="columns")

        for tcr_id, representation in zip(tcr_ids, representations):
            self._cache[tcr_id] = representation

    @staticmethod
    def _generate_tcr_identifier(row: Series) -> Tuple[Optional[str]]:
        return (row.TRAV, row.CDR3A, row.TRAJ, row.TRBV, row.CDR3B, row.TRBJ)