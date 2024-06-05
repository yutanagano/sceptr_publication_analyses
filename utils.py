from edit_penalty import EditPenaltyCollection, EditPenaltyCollectionAnalyser
import pandas as pd
from pandas import DataFrame
from paths import RESULTS_DIR
import numpy as np
from numpy import ndarray
from typing import Optional


class ModelForAnalysis:
    def __init__(self, model_name: str, task_prefix: str, colour: Optional[str] = None, marker: str = "", linestyle: str = "-", zorder: Optional[float] = 1.5, ten_x: bool = False) -> None:
        self.name = model_name
        self.task_prefix = task_prefix
        self.colour = colour
        self.marker = marker
        self.linestyle = linestyle
        self.zorder = zorder
        self.ten_x = ten_x
    
    @property
    def style(self) -> str:
        return self.marker + self.linestyle
    
    def load_data(self, k: Optional[int] = None) -> DataFrame:
        if self.task_prefix in ("ovr_nn", "ovr_avg_dist") and k == 1:
            csv_name = f"ovr_1_shot.csv"
        elif self.task_prefix in ("ovr_unseen_epitopes_nn", "ovr_unseen_epitopes_avg_dist") and k == 1:
            csv_name = f"ovr_unseen_epitopes_1_shot.csv"
        elif k is None:
            csv_name = self.task_prefix + ".csv"
        else:
            csv_name = f"{self.task_prefix}_{k}_shot.csv"

        if self.ten_x:
            path_to_csv = RESULTS_DIR/"10x"/self.name/f"{csv_name}"
        else:
            path_to_csv = RESULTS_DIR/self.name/f"{csv_name}"
        
        return pd.read_csv(path_to_csv)
    
    def load_epc_analyser(self) -> EditPenaltyCollectionAnalyser:
        path_to_epc_state_dict = RESULTS_DIR/self.name/"epc_state_dict.pkl"
        with open(path_to_epc_state_dict, "rb") as f:
            epc = EditPenaltyCollection.from_save(f)
        return EditPenaltyCollectionAnalyser(epc)
    
    def get_num_parameters(self) -> int:
        with open(RESULTS_DIR/self.name/"model_parameter_count.txt", "r") as f:
            count = f.read()
            return int(count)

    def get_model_dimensionality(self) -> int:
        with open(RESULTS_DIR/self.name/"model_dimensionality.txt", "r") as f:
            dims = f.read()
            return int(dims)


def convert_dists_to_scores(dists: ndarray) -> ndarray:
    max_dist = np.max(dists)
    return 1 - dists / max_dist
