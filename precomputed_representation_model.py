import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
from tcr_representation_model import TcrRepresentationModel
from torch import FloatTensor
from typing import Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.resolve()


class PrecomputedRepresentationModel(TcrRepresentationModel):
    def __init__(self, model: TcrRepresentationModel) -> None:
        self._model = model
        self._test_data_representations_cache = self._precompute_test_data_representations()

    def _precompute_test_data_representations(self) -> Dict[Tuple[Optional[str]], FloatTensor]:
        print(f"Precomputing {self._model.name} representations...")

        representations_cache = dict()

        test_data = pd.read_csv(PROJECT_ROOT/"tcr_data"/"preprocessed"/"benchmarking"/"vdjdb_cleaned.csv")
        tcrs = test_data[["TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"]]
        tcrs_uniqued = tcrs.drop_duplicates(ignore_index=True)
        tcr_representations = self._model.calc_vector_representations(tcrs_uniqued)
        tcr_identifiers = tcrs_uniqued.apply(self._generate_tcr_identifier, axis=1)

        for idx, tcr_identifier in tcr_identifiers.items():
            representations_cache[tcr_identifier] = tcr_representations[idx]
        
        return representations_cache

    @property
    def name(self) -> str:
        return self._model.name

    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        def fetch_cached_representation(tcr_identifier: Tuple[Optional[str]]) -> ndarray:
            return self._test_data_representations_cache[tcr_identifier]
        
        tcr_identifiers = instances.apply(self._generate_tcr_identifier, axis=1)
        representations = tcr_identifiers.map(fetch_cached_representation).to_list()
        return np.stack(representations)

    @staticmethod
    def _generate_tcr_identifier(row: Series) -> Tuple[Optional[str]]:
        return (row.TRAV, row.CDR3A, row.TRAJ, row.TRBV, row.CDR3B, row.TRBJ)