from abc import ABC, abstractmethod
from numpy import ndarray
from pandas import DataFrame, Series
import torch
from torch import FloatTensor
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    BertTokenizer,
    BertModel
)


class HuggingFaceLM(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the model as a string.
        """
        pass

    def __init__(self) -> None:
        super().__init__()
        
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        self._tokeniser = self._get_tokeniser()
        self._model = self._get_model().to(device=self._device)

    @abstractmethod
    def _get_tokeniser(self) -> PreTrainedTokenizer:
        """
        Return Huggingface tokeniser.
        """
        pass

    @abstractmethod
    def _get_model(self) -> PreTrainedModel:
        """
        Return Huggingface model. (no need to worry about device as it is automatically set later)
        """
        pass

    def calc_cdist_matrix(self, anchors: DataFrame, comparisons: DataFrame) -> ndarray:
        anchor_representations = self._calc_torch_representations(anchors)
        comparison_representations = self._calc_torch_representations(comparisons)
        return torch.cdist(anchor_representations, comparison_representations, p=2).cpu().numpy()
    
    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        vector_representations = self._calc_torch_representations(instances)
        return torch.pdist(vector_representations, p=2).cpu().numpy()

    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        return self._calc_torch_representations(instances).cpu().numpy()
    
    @torch.no_grad()
    def _calc_torch_representations(self, instances: DataFrame) -> FloatTensor:
        alpha_representations = self._calc_alpha_representations(instances)
        beta_representations = self._calc_beta_representations(instances)
        representations_concatenated = torch.concatenate([alpha_representations, beta_representations], dim=1)
        return representations_concatenated
    
    @abstractmethod
    def _calc_alpha_representations(self, instances: DataFrame) -> FloatTensor:
        """
        Compute and return alpha chain representations seen in the instances df.
        torch.no_grad is already applied.
        """
        pass

    @abstractmethod
    def _calc_beta_representations(self, instances: DataFrame) -> FloatTensor:
        """
        Compute and return beta chain representations seen in the instances df.
        torch.no_grad is already applied.
        """
        pass


class TcrBert(HuggingFaceLM):
    name = "TCR BERT"

    def _get_tokeniser(self) -> PreTrainedTokenizer:
        return BertTokenizer.from_pretrained("wukevin/tcr-bert")

    def _get_model(self) -> PreTrainedModel:
        return BertModel.from_pretrained("wukevin/tcr-bert-mlm-only")

    def _calc_alpha_representations(self, instances: DataFrame) -> FloatTensor:
        return self._calc_cdr3_representations(instances.CDR3A)
    
    def _calc_beta_representations(self, instances: DataFrame) -> FloatTensor:
        return self._calc_cdr3_representations(instances.CDR3B)

    def _calc_cdr3_representations(self, cdr3s: Series) -> FloatTensor:
        cdr3s_preprocessed = cdr3s.map(lambda cdr3: " ".join(list(cdr3))).to_list()
        cdr3s_tokenised = self._tokeniser(cdr3s_preprocessed, return_tensors="pt", padding=True).to(self._device)
        return self._model(**cdr3s_tokenised).pooler_output