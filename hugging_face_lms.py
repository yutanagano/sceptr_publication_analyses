from abc import ABC, abstractmethod
from numpy import ndarray
from pandas import DataFrame, Series
import subprocess
import torch
from torch import FloatTensor, LongTensor
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
        
        self._model = self._get_model()
        self._model.to(device=self._device)
        self._model.eval()

    @abstractmethod
    def _get_tokeniser(self) -> PreTrainedTokenizer:
        """
        Return Huggingface tokeniser.
        """
        pass

    @abstractmethod
    def _get_model(self) -> PreTrainedModel:
        """
        Return Huggingface model. (no need to worry about eval mode or device as they are automatically set later)
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
        cdr3s_embedded = []

        batch_size = 128
        for idx in range(0, len(cdr3s_preprocessed), batch_size):
            batch = cdr3s_preprocessed[idx:idx+batch_size]
            cdr3s_tokenised = self._tokeniser(batch, return_tensors="pt", padding=True).to(self._device)
            token_embeddings = self._model(**cdr3s_tokenised, output_hidden_states=True).hidden_states[8]
            avg_pooled_embedding = self._average_pool(cdr3s_tokenised.attention_mask, token_embeddings)
            cdr3s_embedded.append(avg_pooled_embedding)
        
        return torch.concatenate(cdr3s_embedded, dim=0)
    
    @staticmethod
    def _average_pool(attention_mask: LongTensor, embeddings: FloatTensor):
        new_mask = attention_mask.detach().clone()

        token_sequence_lengths = attention_mask.sum(dim=1)

        new_mask[:,0] = 0
        new_mask[torch.arange(len(token_sequence_lengths)),token_sequence_lengths-1] = 0

        aa_sequence_lengths = new_mask.sum(dim=1)
        aa_embeddings = embeddings * new_mask.unsqueeze(dim=-1)
        avg_pooled_embeddings = aa_embeddings.sum(dim=1) / aa_sequence_lengths.unsqueeze(dim=-1)
        
        return avg_pooled_embeddings


class ProtBert(HuggingFaceLM):
    name = "ProtBert"

    def _get_tokeniser(self) -> PreTrainedTokenizer:
        return BertTokenizer.from_pretrained("Rostlab/prot_bert")
    
    def _get_model(self) -> PreTrainedModel:
        return BertModel.from_pretrained("Rostlab/prot_bert")
    
    def _calc_alpha_representations(self, instances: DataFrame) -> FloatTensor:
        tras = instances[["TRAV", "CDR3A", "TRAJ"]]
        tras.columns = ["v", "cdr3", "j"]
        tra_aa_seqs = tras.apply(row_to_tcr_chain, axis=1)

    def _calc_beta_representations(self, instances: DataFrame) -> FloatTensor:
        pass


def row_to_tcr_chain(row: Series) -> str:
    stitchr_output = subprocess.run(
        [
            "stitchr",
            "-v", row.v,
            "-j", row.j,
            "-cdr3", row.cdr3,
            "-m", "AA"
        ],
        capture_output=True
    )
    stdout_cleaned = stitchr_output.stdout.decode().strip()
    return stdout_cleaned