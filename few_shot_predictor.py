import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from paths import DATA_DIR
from pyrepseq.metric.tcr_metric import TcrMetric
from sklearn.svm import LinearSVC
import utils


BAKCGROUND_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"tanno"/"train.csv").dropna(subset=["TRAV","TRAJ","CDR3A","TRBV","TRBJ","CDR3B"])
bg_sample = BAKCGROUND_DATA.sample(n=1000, random_state=420)


class FewShotOneVsRestPredictor:
    def __init__(self, metric: TcrMetric, positive_refs: DataFrame, queries: DataFrame) -> None:
        self._cdist_matrix = metric.calc_cdist_matrix(queries, positive_refs)
    
    def get_nn_inferences(self) -> ndarray:
        nn_dists = np.min(self._cdist_matrix, axis=1)
        return utils.convert_dists_to_scores(nn_dists)
    
    def get_avg_dist_inferences(self) -> ndarray:
        avg_dists = np.mean(self._cdist_matrix, axis=1)
        return utils.convert_dists_to_scores(avg_dists)


class FewShotSVCPredictor:
    def __init__(self, model, positive_refs: DataFrame, queries: DataFrame) -> None:
        svc = LinearSVC(class_weight="balanced", dual="auto", max_iter=100_000)

        positive_reps = model.calc_vector_representations(positive_refs)
        bg_reps = model.calc_vector_representations(bg_sample)
        training_reps = np.concatenate([positive_reps, bg_reps], axis=0)
        ground_truth = np.array([1] * len(positive_refs) + [0] * len(bg_sample))
        svc.fit(training_reps, ground_truth)

        query_reps = model.calc_vector_representations(queries)
        self._scores = svc.decision_function(query_reps)
    
    def get_inferences(self) -> ndarray:
        return self._scores


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