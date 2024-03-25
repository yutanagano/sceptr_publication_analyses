import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
import pickle
import torch
from torch import FloatTensor
from typing import Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.resolve()
CACHE_DIR = PROJECT_ROOT/".precomputed_representation_cache"
CACHE_DIR.mkdir(exist_ok=True)


class PrecomputedRepresentationModel():
    def __init__(self, model) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        self._model = model
        self._representation_cache_path = CACHE_DIR/f"{model.name}_rep_cache.pkl"
        self._test_data_representations_cache = self._get_test_data_representations()
    
    def _get_test_data_representations(self) -> Dict[Tuple[Optional[str]], FloatTensor]:
        if not self._representation_cache_path.is_file():
            representations = self._precompute_test_data_representations()
            self._save_cache(representations)
            return representations
        
        with open(self._representation_cache_path, "rb") as f:
            cache = pickle.load(f)
            return {k: torch.tensor(v, device=self._device) for k, v in cache.items()}

    def _precompute_test_data_representations(self) -> Dict[Tuple[Optional[str]], FloatTensor]:
        print(f"Precomputing {self._model.name} representations...")

        representations_cache = dict()

        test_data = pd.read_csv(PROJECT_ROOT/"tcr_data"/"preprocessed"/"benchmarking"/"vdjdb_cleaned.csv")
        tcrs = test_data[["TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"]]
        tcrs_uniqued = tcrs.drop_duplicates(ignore_index=True)
        tcr_representations = self._model.calc_vector_representations(tcrs_uniqued)
        tcr_identifiers = tcrs_uniqued.apply(self._generate_tcr_identifier, axis=1)

        for idx, tcr_identifier in tcr_identifiers.items():
            representations_cache[tcr_identifier] = torch.tensor(tcr_representations[idx], device=self._device)
        
        return representations_cache
    
    def _save_cache(self, representations) -> None:
        moved_to_cpu = {k: v.cpu() for k, v in representations.items()}

        with open(self._representation_cache_path, "wb") as f:
            pickle.dump(moved_to_cpu, f)

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
        def fetch_cached_representation(tcr_identifier: Tuple[Optional[str]]) -> ndarray:
            return self._test_data_representations_cache[tcr_identifier]
        
        tcr_identifiers = instances.apply(self._generate_tcr_identifier, axis=1)
        representations = tcr_identifiers.map(fetch_cached_representation).to_list()
        return torch.stack(representations)

    @staticmethod
    def _generate_tcr_identifier(row: Series) -> Tuple[Optional[str]]:
        return (row.TRAV, row.CDR3A, row.TRAJ, row.TRBV, row.CDR3B, row.TRBJ)