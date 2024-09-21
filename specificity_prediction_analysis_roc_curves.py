from datetime import datetime
from few_shot_predictor import FewShotOneVsRestPredictor
import json
import logging
import numpy as np
from numpy import ndarray
import pandas as pd
from paths import DATA_DIR, RESULTS_DIR
from cached_representation_model import CachedRepresentationModel
from pyrepseq.metric import tcr_metric
from pyrepseq.metric.tcr_metric import TcrMetric
import random
from sceptr import variant
from sklearn import metrics
from hugging_face_lms import TcrBert, ProtBert, Esm2
from tqdm import tqdm
from typing import Dict, List
import utils


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


LABELLED_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"benchmarking"/"vdjdb_cleaned.csv")
EPITOPES = LABELLED_DATA.Epitope.unique()

MODELS = (
    tcr_metric.Cdr3Levenshtein(),
    tcr_metric.Tcrdist(),
    CachedRepresentationModel(variant.default()),
    CachedRepresentationModel(TcrBert()),
    CachedRepresentationModel(ProtBert()),
    CachedRepresentationModel(Esm2()),
)

NUM_SHOTS = (2, 5, 10, 20, 50, 100, 200)
NUM_RANDOM_FOLDS = 100


def main() -> None:
    results_per_model = {
        model.name: get_results(model) for model in MODELS
    }

    for model_name, results in results_per_model.items():
        save_results(model_name, results)
    
    for model in MODELS:
        if isinstance(model, CachedRepresentationModel):
            model.save_cache()


def get_results(model: TcrMetric) -> Dict[str, dict]:
    return {
        "1_shot_rocs": get_distance_based_one_shot_results(model),
        "200_shot_rocs": get_distance_based_200_shot_results(model),
    }


def get_distance_based_one_shot_results(model: TcrMetric) -> dict:
    print(f"Catalogueing 1-shot ROCs for {model.name}...")

    results = dict()

    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = LABELLED_DATA.Epitope == epitope
        epitope_references = LABELLED_DATA[labelled_data_epitope_mask]

        if len(epitope_references) - 1 < 100:
            logging.debug(f"Not enough references for {epitope}, skipping")
            continue

        cdist_matrix = model.calc_cdist_matrix(LABELLED_DATA, epitope_references)

        tprs_collection = []

        for cdist_idx, tcr_idx in enumerate(epitope_references.index):
            dists = cdist_matrix[:,cdist_idx]
            similarities = utils.convert_dists_to_scores(dists)

            similarities = np.delete(similarities, tcr_idx)
            ground_truth = np.delete(labelled_data_epitope_mask, tcr_idx)

            fprs, tprs, _ = metrics.roc_curve(ground_truth, similarities, drop_intermediate=True)
            fixed_tprs = get_tprs_at_integer_percentage_points(fprs, tprs)
            tprs_collection.append(fixed_tprs)

        results[epitope] = generate_summary(tprs_collection)
    
    return results


def get_distance_based_200_shot_results(model: TcrMetric) -> dict:
    print(f"Catalogueing 200-shot ROCs for {model.name}...")

    results = dict()
    
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = LABELLED_DATA.Epitope == epitope
        epitope_references = LABELLED_DATA[labelled_data_epitope_mask]
        ref_index_sets = epitope_references.index.to_list()

        if len(epitope_references) - 200 < 100:
            logging.debug(f"Not enough references for {epitope} for 200 shots, skipping")
            continue

        random.seed("tcrsarecool")
        ref_index_sets = [
            random.sample(ref_index_sets, k=200) for _ in range(NUM_RANDOM_FOLDS)
        ]

        tprs_collection = []

        for ref_index_set in ref_index_sets:
            positive_refs = LABELLED_DATA.loc[ref_index_set]
            queries = LABELLED_DATA.drop(index=ref_index_set)

            ground_truth = queries.Epitope == epitope
            predictor = FewShotOneVsRestPredictor(model, positive_refs=positive_refs, queries=queries)

            scores = predictor.get_nn_inferences()
            fprs, tprs, _ = metrics.roc_curve(ground_truth, scores, drop_intermediate=True)
            fixed_tprs = get_tprs_at_integer_percentage_points(fprs, tprs)
            tprs_collection.append(fixed_tprs)

        results[epitope] = generate_summary(tprs_collection)

    return results


def get_tprs_at_integer_percentage_points(fprs: ndarray, tprs: ndarray) -> ndarray:
    query_fprs = np.linspace(0,1,101)
    result_tprs = np.empty_like(query_fprs)

    for result_idx, query_fpr in enumerate(query_fprs):
        if query_fpr == 0:
            result_tprs[0] = 0
            continue

        if query_fpr == 1:
            result_tprs[100] = 1
            continue

        result_tprs[result_idx] = np.interp(query_fpr, fprs, tprs)

    return result_tprs


def generate_summary(tprs_collection: List[ndarray]) -> Dict[str, List[float]]:
    tprs_matrix = np.stack(tprs_collection, axis=1)

    tprs_mean = np.mean(tprs_matrix, axis=1)
    tprs_std = np.std(tprs_matrix, axis=1)
    tprs_upper_quartile = np.quantile(tprs_matrix, 0.75, axis=1)
    tprs_lower_quartile = np.quantile(tprs_matrix, 0.25, axis=1)

    return {
        "tprs_mean": tprs_mean.tolist(),
        "tprs_std": tprs_std.tolist(),
        "tprs_qt_0.75": tprs_upper_quartile.tolist(),
        "tprs_qt_0.25": tprs_lower_quartile.tolist()
    }


def save_results(model_name: str, results: dict) -> None:
    model_dir = RESULTS_DIR/model_name
    model_dir.mkdir(exist_ok=True)
    
    with open(model_dir/"individual_rocs.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
