import numpy as np
from numpy import ndarray
from pandas import DataFrame
from pyrepseq.metric.tcr_metric import TcrMetric
from sklearn.svm import SVC
import utils


class FewShotOneVsRestPredictor:
    def __init__(self, metric: TcrMetric, positive_refs: DataFrame, queries: DataFrame) -> None:
        self._cdist_matrix = metric.calc_cdist_matrix(queries, positive_refs)
    
    def get_nn_inferences(self) -> ndarray:
        nn_dists = np.min(self._cdist_matrix, axis=1)
        return utils.convert_dists_to_scores(nn_dists)
    
    def get_avg_dist_inferences(self) -> ndarray:
        avg_dists = np.mean(self._cdist_matrix, axis=1)
        return utils.convert_dists_to_scores(avg_dists)


class FewShotOneInManyPredictor:
    def __init__(self, metric: TcrMetric, positive_refs: DataFrame, queries: DataFrame) -> None:
        self._cdist_matrices = dict()

        grouped_by_epitope = positive_refs.groupby("Epitope")
        for epitope in grouped_by_epitope.groups:
            epitope_refs = grouped_by_epitope.get_group(epitope)
            cdist_matrix = metric.calc_cdist_matrix(queries, epitope_refs)
            self._cdist_matrices[epitope] = cdist_matrix
    
    def get_nn_inferences(self) -> DataFrame:
        nn_dists = dict()
        
        for epitope, cdist_matrix in self._cdist_matrices.items():
            nn_dists[epitope] = np.min(cdist_matrix, axis=1)

        nn_dists = DataFrame.from_dict(nn_dists)
        return utils.convert_dists_to_scores(nn_dists)
    
    def get_avg_dist_inferences(self) -> DataFrame:
        avg_dists = dict()

        for epitope, cdist_matrix in self._cdist_matrices.items():
            avg_dists[epitope] = np.mean(cdist_matrix, axis=1)
        
        avg_dists = DataFrame.from_dict(avg_dists)
        return utils.convert_dists_to_scores(avg_dists)