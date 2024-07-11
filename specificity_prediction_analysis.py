from datetime import datetime
from few_shot_predictor import FewShotOneVsRestPredictor, FewShotSVCPredictor
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from paths import DATA_DIR, RESULTS_DIR
from cached_representation_model import CachedRepresentationModel
from pyrepseq.metric import tcr_metric
from pyrepseq.metric.tcr_metric import TcrMetric
import random
from sceptr import variant
from sklearn import metrics
from hugging_face_lms import TcrBert, ProtBert, Esm2
from tqdm import tqdm
from typing import Dict, Iterable, List, Tuple
import utils


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


LABELLED_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"benchmarking"/"vdjdb_cleaned.csv")
EPITOPES = LABELLED_DATA.Epitope.unique()

MODELS = (
    # tcr_metric.Cdr3Levenshtein(),
    # tcr_metric.CdrLevenshtein(),
    # tcr_metric.Tcrdist(),
    # CachedRepresentationModel(variant.default()),
    # CachedRepresentationModel(variant.mlm_only()),
    # CachedRepresentationModel(variant.average_pooling()),
    # CachedRepresentationModel(variant.synthetic_data()),
    # CachedRepresentationModel(variant.shuffled_data()),
    # CachedRepresentationModel(variant.cdr3_only()),
    # CachedRepresentationModel(variant.cdr3_only_mlm_only()),
    CachedRepresentationModel(variant.left_aligned()),
    # CachedRepresentationModel(variant.dropout_noise_only()),
    # CachedRepresentationModel(TcrBert()),
    # CachedRepresentationModel(ProtBert()),
    # CachedRepresentationModel(Esm2()),
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


def get_results(model: TcrMetric) -> Dict[str, DataFrame]:
    return {
        **get_distance_based_one_shot_results(model),
        **get_distance_based_few_shot_results(model),
        # **get_support_vector_machine_results(model)
    }


def get_distance_based_one_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing OVR[1-shot] for {model.name}...")

    results = []
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = LABELLED_DATA.Epitope == epitope
        epitope_references = LABELLED_DATA[labelled_data_epitope_mask]

        if len(epitope_references) - 1 < 100:
            logging.debug(f"Not enough references for {epitope}, skipping")
            continue

        cdist_matrix = model.calc_cdist_matrix(LABELLED_DATA, epitope_references)

        aucs = []
        for cdist_idx, tcr_idx in enumerate(epitope_references.index):
            dists = cdist_matrix[:,cdist_idx]
            similarities = utils.convert_dists_to_scores(dists)

            similarities = np.delete(similarities, tcr_idx)
            ground_truth = np.delete(labelled_data_epitope_mask, tcr_idx)

            aucs.append(metrics.roc_auc_score(ground_truth, similarities))

        auc_summary = generate_summary(epitope, aucs, "auc")
        results.extend(auc_summary)
    
    return {"ovr_1_shot": DataFrame.from_records(results)}


def get_distance_based_few_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
    results = dict()
    for num_shots in NUM_SHOTS:
        k_shot_results = get_distance_based_k_shot_results(model, k=num_shots)
        results.update(k_shot_results)
    return results


def get_distance_based_k_shot_results(model: TcrMetric, k: int) -> Dict[str, DataFrame]:
    print(f"Commencing OVR[{k}-shot] for {model.name}...")

    nn_results = []
    avg_dist_results = []
    
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = LABELLED_DATA.Epitope == epitope
        epitope_references = LABELLED_DATA[labelled_data_epitope_mask]
        ref_index_sets = epitope_references.index.to_list()

        if len(epitope_references) - k < 100:
            logging.debug(f"Not enough references for {epitope} for {k} shots, skipping")
            continue

        random.seed("tcrsarecool")
        ref_index_sets = [
            random.sample(ref_index_sets, k=k) for _ in range(NUM_RANDOM_FOLDS)
        ]
        logging.info(f"{model.name}:OVR[{k}-shot]:{epitope}: {ref_index_sets[0][:2]}")

        auc_summaries = get_distance_based_k_shot_auc_summaries_for_epitope(model, epitope, ref_index_sets)

        nn_results.extend(auc_summaries["nn"])
        avg_dist_results.extend(auc_summaries["avg_dist"])

    return {
        f"ovr_nn_{k}_shot": DataFrame.from_records(nn_results),
        f"ovr_avg_dist_{k}_shot": DataFrame.from_records(avg_dist_results),
    }


def get_distance_based_k_shot_auc_summaries_for_epitope(model: TcrMetric, epitope: str, ref_index_sets: Iterable[List[int]]) -> Dict[str, List[Dict]]:
    nn_aucs = []
    avg_dist_aucs = []
    
    for ref_index_set in ref_index_sets:
        positive_refs = LABELLED_DATA.loc[ref_index_set]
        queries = LABELLED_DATA.drop(index=ref_index_set)
        ground_truth = queries.Epitope == epitope
        predictor = FewShotOneVsRestPredictor(model, positive_refs=positive_refs, queries=queries)

        nn_scores = predictor.get_nn_inferences()
        avg_dist_scores = predictor.get_avg_dist_inferences()

        nn_auc = metrics.roc_auc_score(ground_truth, nn_scores)
        avg_dist_auc = metrics.roc_auc_score(ground_truth, avg_dist_scores)

        nn_aucs.append(nn_auc)
        avg_dist_aucs.append(avg_dist_auc)
    
    return {
        "nn": generate_summary(epitope, nn_aucs, "auc"),
        "avg_dist": generate_summary(epitope, avg_dist_aucs, "auc")
    }


def get_support_vector_machine_results(model) -> Dict[str, DataFrame]:
    if "Levenshtein" in model.name or "dist" in model.name:
        print(f"Skipping OVRSVC for {model.name}.")
        return dict()

    results = dict()
    for num_shots in (1, *NUM_SHOTS):
        filename, df = get_support_vector_machine_k_shot_results(model, k=num_shots)
        results[filename] = df
    return results


def get_support_vector_machine_k_shot_results(model, k: int) -> Tuple[str, DataFrame]:
    print(f"Commencing OVRSVC[{k}-shot] for {model.name}...")

    results = []
    
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = LABELLED_DATA.Epitope == epitope
        epitope_references = LABELLED_DATA[labelled_data_epitope_mask]
        ref_indices = epitope_references.index.to_list()

        if len(epitope_references) - k < 100:
            logging.debug(f"Not enough references for {epitope} for {k} shots, skipping")
            continue
        
        if k == 1:
            ref_index_sets = [[idx] for idx in ref_indices]
        else:
            random.seed("tcrsarecool")
            ref_index_sets = [
                random.sample(ref_indices, k=k) for _ in range(NUM_RANDOM_FOLDS)
            ]
        logging.info(f"{model.name}:OVRSVC[{k}-shot]:{epitope}: {ref_index_sets[0][:2]}")

        auc_summaries = get_support_vector_machine_k_shot_results_for_epitope(model, epitope, ref_index_sets)

        results.extend(auc_summaries)

    return (f"ovr_svc_{k}_shot", DataFrame.from_records(results))


def get_support_vector_machine_k_shot_results_for_epitope(model: TcrMetric, epitope: str, ref_index_sets: Iterable[List[int]]) -> List[Dict]:
    aucs = []
    
    for ref_index_set in ref_index_sets:
        positive_refs = LABELLED_DATA.loc[ref_index_set]
        queries = LABELLED_DATA.drop(index=ref_index_set)
        ground_truth = queries.Epitope == epitope
        predictor = FewShotSVCPredictor(model, positive_refs, queries)
        
        scores = predictor.get_inferences()
        auc = metrics.roc_auc_score(ground_truth, scores)
        aucs.append(auc)
    
    return generate_summary(epitope, aucs, "auc")


def generate_summary(epitope: str, measures: Iterable[float], measure_name: str) -> List[Dict]:
    records = [
        {"epitope": epitope, "split": idx, measure_name: measure} for idx, measure in enumerate(measures)
    ]
    return records


def save_results(model_name: str, results: Dict[str, DataFrame]) -> None:
    model_dir = RESULTS_DIR/model_name
    model_dir.mkdir(exist_ok=True)
    
    # for benchmark_type, results_table in results.items():
    #     results_table.to_csv(model_dir/f"{benchmark_type}.csv", index=False)


if __name__ == "__main__":
    main()
