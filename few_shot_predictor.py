from numpy import ndarray
from pandas import DataFrame
from pyrepseq.metric.tcr_metric import TcrMetric


class FewShotPredictor:
    def __init__(self, metric: TcrMetric, positive_refs: DataFrame, bg_refs: DataFrame) -> None:
        self._metric = metric
        self._positive_refs = positive_refs
        self._bg_refs = bg_refs
    
    def get_nn_inferences(self, queries: DataFrame) -> ndarray:
        cdist_matrix = self._metric.calc_cdist_matrix(queries, self._positive_refs)
        nn_dists = np.min(cdist_matrix, axis=1)
        return convert_dists_to_scores(nn_dists)
    
    def get_avg_dist_inferences(self, queries: DataFrame) -> ndarray:
        cdist_matrix = self._metric.calc_cdist_matrix(queries, self._positive_refs)
        avg_dists = np.mean(cdist_matrix, axis=1)
        return convert_dists_to_scores(avg_dists)
