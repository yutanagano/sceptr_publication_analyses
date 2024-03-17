from numpy import ndarray
from pandas import DataFrame, Series
import torch
from torch import FloatTensor
from transformers import BertTokenizer, BertModel


class TcrBert:
    def __init__(self):
        self._tokeniser = BertTokenizer.from_pretrained("wukevin/tcr-bert")
        self._alpha_model = BertModel.from_pretrained("wukevin/tcr-bert-mlm-only")
        self._beta_model = BertModel.from_pretrained("wukevin/tcr-bert")

        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            self._device = torch.device("cpu")

        self._alpha_model.to(self._device)
        self._beta_model.to(self._device)

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
        alpha_embeddings = self._generate_embeddings(instances.CDR3A, self._alpha_model)
        beta_embeddings = self._generate_embeddings(instances.CDR3B, self._beta_model)
        embeddings_stacked = torch.concatenate([alpha_embeddings, beta_embeddings], dim=1)
        return embeddings_stacked

    @torch.no_grad()
    def _generate_embeddings(self, cdr3s: Series, model: BertModel) -> FloatTensor:
        cdr3s_preprocessed = cdr3s.map(lambda cdr3: " ".join(list(cdr3))).to_list()
        cdr3s_tokenised = self._tokeniser(cdr3s_preprocessed, return_tensors="pt", padding=True)
        cdr3s_tokenised = {k: v.to(self._device) for k, v in cdr3s_tokenised.items()}
        return model(**cdr3s_tokenised).pooler_output