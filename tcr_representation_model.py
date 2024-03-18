from abc import ABC, abstractmethod
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial import distance


class TcrRepresentationModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        pass

    def calc_cdist_matrix(self, anchors: DataFrame, comparisons: DataFrame) -> ndarray:
        anchor_representations = self.calc_vector_representations(anchors)
        comparison_representations = self.calc_vector_representations(comparisons)
        return distance.cdist(anchor_representations, comparison_representations, metric="euclidean")

    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        representations = self.calc_vector_representations(instances)
        return distance.pdist(representations, metric="euclidean")