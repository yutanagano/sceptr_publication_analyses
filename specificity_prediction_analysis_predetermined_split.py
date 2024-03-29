from datetime import datetime
from few_shot_predictor import FewShotOneVsRestPredictor
import logging
import pandas as pd
from pandas import DataFrame
from paths import DATA_DIR, RESULTS_DIR
from cached_representation_model import CachedRepresentationModel
from pyrepseq.metric import tcr_metric
from pyrepseq.metric.tcr_metric import TcrMetric
from sceptr import variant
from sklearn import metrics
from hugging_face_lms import TcrBert, ProtBert, Esm2
from typing import Dict


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


TRAIN_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"benchmarking"/"train.csv")
TEST_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"benchmarking"/"test.csv")

MODELS = (
    tcr_metric.Cdr3Levenshtein(),
    tcr_metric.CdrLevenshtein(),
    tcr_metric.Tcrdist(),
    CachedRepresentationModel(variant.ab_sceptr()),
    CachedRepresentationModel(variant.ab_sceptr_finetuned()),
    CachedRepresentationModel(TcrBert()),
    CachedRepresentationModel(ProtBert()),
    CachedRepresentationModel(Esm2()),
)


def main() -> None:
    results_per_model = {
        model.name: get_results(model) for model in MODELS
    }

    for model_name, results in results_per_model.items():
        save_results(model_name, results)
    
    for model in MODELS:
        if isinstance(model, CachedRepresentationModel):
            model.save_cache()


def get_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing OVR for {model.name}...")

    nn_results = []
    avg_dist_results = []

    train_data_grouped_by_epitope = TRAIN_DATA.groupby("Epitope")
    for epitope, tcr_indices in train_data_grouped_by_epitope.groups.items():
        epitope_references = TRAIN_DATA.loc[tcr_indices]
        ground_truth = TEST_DATA.Epitope == epitope
        predictor = FewShotOneVsRestPredictor(model, positive_refs=epitope_references, queries=TEST_DATA)

        nn_scores = predictor.get_nn_inferences()
        avg_dist_scores = predictor.get_avg_dist_inferences()

        nn_auc = metrics.roc_auc_score(ground_truth, nn_scores)
        avg_dist_auc = metrics.roc_auc_score(ground_truth, avg_dist_scores)

        nn_results.append({"epitope": epitope, "auc": nn_auc})
        avg_dist_results.append({"epitope": epitope, "auc": avg_dist_auc})

    return {
        f"one_vs_rest_predetermined_split_nn": DataFrame.from_records(nn_results),
        f"one_vs_rest_predetermined_split_avg_dist": DataFrame.from_records(avg_dist_results),
    }


def save_results(model_name: str, results: Dict[str, DataFrame]) -> None:
    model_dir = RESULTS_DIR/model_name
    model_dir.mkdir(exist_ok=True)
    
    for benchmark_type, results_table in results.items():
        results_table.to_csv(model_dir/f"{benchmark_type}.csv", index=False)


if __name__ == "__main__":
    main()
