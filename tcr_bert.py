from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
import torch
from torch import FloatTensor
from transformers import BertTokenizer, BertModel
from typing import Dict


PROJECT_ROOT = Path(__file__).parent.resolve()


class TcrBert:
    name = "TCR BERT"

    def __init__(self):
        self._tokeniser = BertTokenizer.from_pretrained("wukevin/tcr-bert")
        self._model = BertModel.from_pretrained("wukevin/tcr-bert-mlm-only")

        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

    def calc_cdist_matrix(self, anchors: DataFrame, comparisons: DataFrame) -> ndarray:
        anchor_representations = self._calc_torch_representations(anchors)
        comparison_representations = self._calc_torch_representations(comparisons)
        return torch.cdist(anchor_representations, comparison_representations, p=2).cpu().numpy()
    
    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        vector_representations = self._calc_torch_representations(instances)
        return torch.pdist(vector_representations, p=2).cpu().numpy()

    def calc_vector_representations(self, instances: DataFrame) -> ndarray:
        return self._calc_torch_representations(instances).cpu().numpy()
    
    def _calc_torch_representations(self, instances: DataFrame) -> FloatTensor:
        alpha_representations = self._generate_representations(instances.CDR3A)
        beta_representations = self._generate_representations(instances.CDR3B)
        representations_concatenated = torch.concatenate([alpha_representations, beta_representations], dim=1)
        return representations_concatenated

    @torch.no_grad()
    def _generate_representations(self, cdr3s: Series) -> FloatTensor:
        cdr3s_preprocessed = cdr3s.map(lambda cdr3: " ".join(list(cdr3))).to_list()
        cdr3s_tokenised = self._tokeniser(cdr3s_preprocessed, return_tensors="pt", padding=True)
        cdr3s_tokenised = {k: v.to(self._device) for k, v in cdr3s_tokenised.items()}
        return self._model(**cdr3s_tokenised).pooler_output