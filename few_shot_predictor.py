import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from pyrepseq.metric.tcr_metric import TcrMetric
from sklearn.svm import SVC
import utils


def rbf_kernel(cdist: ndarray, characteristic_length: float = 1) -> ndarray:
    return np.exp(- (cdist / characteristic_length) ** 2)


class FewShotPredictor:
    def __init__(self, metric: TcrMetric, positive_refs: DataFrame, bg_refs: DataFrame) -> None:
        self._metric = metric
        self._positive_refs = positive_refs
        self._bg_refs = bg_refs
        self._svc = self._train_svc()

    def _train_svc(self) -> SVC:
        training_data = pd.concat([self._positive_refs, self._bg_refs])
        ground_truth = [True] * len(self._positive_refs) + [False] * len(self._bg_refs)
        raw_cdist_matrix = self._metric.calc_cdist_matrix(training_data, training_data)
        kernel_cdist_matrix = rbf_kernel(raw_cdist_matrix)
        
        svc = SVC(kernel="precomputed", probability=True, class_weight="balanced")
        svc.fit(kernel_cdist_matrix, ground_truth)
        return svc
    
    def get_nn_inferences(self, queries: DataFrame) -> ndarray:
        cdist_matrix = self._metric.calc_cdist_matrix(queries, self._positive_refs)
        nn_dists = np.min(cdist_matrix, axis=1)
        return utils.convert_dists_to_scores(nn_dists)
    
    def get_avg_dist_inferences(self, queries: DataFrame) -> ndarray:
        cdist_matrix = self._metric.calc_cdist_matrix(queries, self._positive_refs)
        avg_dists = np.mean(cdist_matrix, axis=1)
        return utils.convert_dists_to_scores(avg_dists)

    def get_svc_inferences(self, queries: DataFrame) -> ndarray:
        training_data = pd.concat([self._positive_refs, self._bg_refs])
        raw_cdist_matrix = self._metric.calc_cdist_matrix(queries, training_data)
        kernel_cdist_matrix = rbf_kernel(raw_cdist_matrix)
        return self._svc.predict_proba(kernel_cdist_matrix)[:,1]
