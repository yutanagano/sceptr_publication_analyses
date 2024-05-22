from collections import defaultdict
from datetime import datetime
from few_shot_predictor import FewShotOneVsRestPredictor, FewShotOneInManyPredictor
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from paths import DATA_DIR, RESULTS_DIR
from cached_representation_model import CachedRepresentationModel
from pyrepseq.metric import tcr_metric
from pyrepseq.metric.tcr_metric import TcrMetric
import random
from sceptr import variant
from sklearn import metrics
from hugging_face_lms import TcrBert, ProtBert, Esm2
from tqdm import tqdm
from typing import Dict, Iterable, List
import utils


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


TRAIN_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"benchmarking"/"train.csv")
TEST_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"benchmarking"/"test.csv")

TEST_DATA_DISCRIMINATION = TEST_DATA[TEST_DATA.Epitope.isin(TRAIN_DATA.Epitope.unique())].reset_index(drop=True)

TEST_DATA_UNSEEN_EPITOPES = TEST_DATA[TEST_DATA.Epitope.map(lambda ep: ep not in TRAIN_DATA.Epitope.unique())].reset_index(drop=True)
UNSEEN_EPITOPES = TEST_DATA_UNSEEN_EPITOPES.Epitope.unique()

MODELS = (
    # tcr_metric.Cdr3Levenshtein(),
    # tcr_metric.CdrLevenshtein(),
    tcr_metric.Tcrdist(),
    CachedRepresentationModel(variant.default()),
    CachedRepresentationModel(variant.finetuned()),
    # CachedRepresentationModel(variant.cdr3_only()),
    # CachedRepresentationModel(variant.mlm_only()),
    # CachedRepresentationModel(variant.cdr3_only_mlm_only()),
    # CachedRepresentationModel(variant.classic()),
    # CachedRepresentationModel(variant.olga()),
    # CachedRepresentationModel(variant.average_pooling()),
    # CachedRepresentationModel(variant.unpaired()),
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
        **get_discrimination_results(model),
        # **get_discrimination_avg_rank(model),
        # **get_detection_results(model),
        # **get_one_vs_rest_one_shot_unseen_results(model),
        # **get_one_vs_rest_few_shot_unseen_results(model)
    }


def get_discrimination_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing discrimination (predetermined split) for {model.name}...")

    nn_results = []
    avg_dist_results = []

    train_data_grouped_by_epitope = TRAIN_DATA.groupby("Epitope")
    for epitope, tcr_indices in train_data_grouped_by_epitope.groups.items():
        epitope_references = TRAIN_DATA.loc[tcr_indices]
        ground_truth = TEST_DATA_DISCRIMINATION.Epitope == epitope
        predictor = FewShotOneVsRestPredictor(model, positive_refs=epitope_references, queries=TEST_DATA_DISCRIMINATION)

        nn_scores = predictor.get_nn_inferences()
        avg_dist_scores = predictor.get_avg_dist_inferences()

        nn_auc = metrics.roc_auc_score(ground_truth, nn_scores)
        avg_dist_auc = metrics.roc_auc_score(ground_truth, avg_dist_scores)

        nn_results.append({"epitope": epitope, "auc": nn_auc})
        avg_dist_results.append({"epitope": epitope, "auc": avg_dist_auc})

    return {
        f"discrimination_ovr_predetermined_split_nn": DataFrame.from_records(nn_results),
        f"discrimination_ovr_predetermined_split_avg_dist": DataFrame.from_records(avg_dist_results),
    }


def get_discrimination_avg_rank(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing discrimination (avg rank, predetermined split) for {model.name}...")

    predictor = FewShotOneInManyPredictor(model, TRAIN_DATA, TEST_DATA_DISCRIMINATION)

    nn_scores_table = predictor.get_nn_inferences()
    avg_dist_scores_table = predictor.get_avg_dist_inferences()

    nn_avg_rank_per_epitope = get_avg_rank_per_epitope(nn_scores_table, TEST_DATA_DISCRIMINATION["Epitope"])
    avg_dist_avg_rank_per_epitope = get_avg_rank_per_epitope(avg_dist_scores_table, TEST_DATA_DISCRIMINATION["Epitope"])

    nn_df = DataFrame.from_records([
        {"epitope": epitope, "avg_rank": avg_rank} for epitope, avg_rank in nn_avg_rank_per_epitope.items()
    ])
    avg_dist_df = DataFrame.from_records([
        {"epitope": epitope, "avg_rank": avg_rank} for epitope, avg_rank in avg_dist_avg_rank_per_epitope.items()
    ])

    return {
        f"discrimination_oim_predetermined_split_nn": nn_df,
        f"discrimination_oim_predetermined_split_avg_dist": avg_dist_df
    }


def get_avg_rank_per_epitope(scores: DataFrame, true_labels: Series) -> Dict[str, float]:
    scores = scores.reset_index(drop=True)
    true_labels = true_labels.reset_index(drop=True)

    rank_of_true_label = scores.apply(
        lambda row: row.sort_values(ascending=False).index.get_loc(true_labels.loc[row.name]) + 1,
        axis=1
    )

    labels_and_ranks = DataFrame.from_dict({"true_label": true_labels, "rank": rank_of_true_label})
    avg_rank_per_epitope = labels_and_ranks.groupby("true_label").aggregate("mean")["rank"]
    return avg_rank_per_epitope.to_dict()


def get_detection_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing detection (predetermined split) for {model.name}...")

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
        f"detection_predetermined_split_nn": DataFrame.from_records(nn_results),
        f"detection_predetermined_split_avg_dist": DataFrame.from_records(avg_dist_results),
    }


def get_one_vs_rest_one_shot_unseen_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing OVR[1-shot] (unseen epitopes) for {model.name}...")

    results = []
    for epitope in tqdm(UNSEEN_EPITOPES):
        labelled_data_epitope_mask = TEST_DATA_UNSEEN_EPITOPES.Epitope == epitope
        epitope_references = TEST_DATA_UNSEEN_EPITOPES[labelled_data_epitope_mask]

        if len(epitope_references) - 1 < 100:
            logging.debug(f"Not enough references for {epitope}, skipping")
            continue

        cdist_matrix = model.calc_cdist_matrix(TEST_DATA_UNSEEN_EPITOPES, epitope_references)

        aucs = []
        for cdist_idx, tcr_idx in enumerate(epitope_references.index):
            dists = cdist_matrix[:,cdist_idx]
            similarities = utils.convert_dists_to_scores(dists)

            similarities = np.delete(similarities, tcr_idx)
            ground_truth = np.delete(labelled_data_epitope_mask, tcr_idx)

            aucs.append(metrics.roc_auc_score(ground_truth, similarities))

        auc_summary = generate_summary(epitope, aucs, "auc")
        results.extend(auc_summary)
    
    return {"ovr_unseen_epitopes_1_shot": DataFrame.from_records(results)}


def get_one_vs_rest_few_shot_unseen_results(model: TcrMetric) -> Dict[str, DataFrame]:
    results = dict()
    for num_shots in NUM_SHOTS:
        k_shot_results = get_one_vs_rest_k_shot_unseen_results(model, k=num_shots)
        results.update(k_shot_results)
    return results


def get_one_vs_rest_k_shot_unseen_results(model: TcrMetric, k: int) -> Dict[str, DataFrame]:
    print(f"Commencing OVR[{k}-shot] for {model.name}...")

    nn_results = []
    avg_dist_results = []
    
    for epitope in tqdm(UNSEEN_EPITOPES):
        labelled_data_epitope_mask = TEST_DATA_UNSEEN_EPITOPES.Epitope == epitope
        epitope_references = TEST_DATA_UNSEEN_EPITOPES[labelled_data_epitope_mask]
        ref_index_sets = epitope_references.index.to_list()

        if len(epitope_references) - k < 100:
            logging.debug(f"Not enough references for {epitope} for {k} shots, skipping")
            continue

        random.seed("tcrsarecool")
        ref_index_sets = [
            random.sample(ref_index_sets, k=k) for _ in range(NUM_RANDOM_FOLDS)
        ]
        logging.info(f"{model.name}:OVR[{k}-shot]:{epitope}: {ref_index_sets[0][:2]}")

        auc_summaries = get_one_vs_rest_k_shot_auc_summaries_for_epitope(model, epitope, ref_index_sets)

        nn_results.extend(auc_summaries["nn"])
        avg_dist_results.extend(auc_summaries["avg_dist"])

    return {
        f"ovr_unseen_epitopes_nn_{k}_shot": DataFrame.from_records(nn_results),
        f"ovr_unseen_epitopes_avg_dist_{k}_shot": DataFrame.from_records(avg_dist_results),
    }


def get_one_vs_rest_k_shot_auc_summaries_for_epitope(model: TcrMetric, epitope: str, ref_index_sets: Iterable[List[int]]) -> Dict[str, List[Dict]]:
    nn_aucs = []
    avg_dist_aucs = []
    
    for ref_index_set in ref_index_sets:
        positive_refs = TEST_DATA_UNSEEN_EPITOPES.loc[ref_index_set]
        queries = TEST_DATA_UNSEEN_EPITOPES.drop(index=ref_index_set)
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


def generate_summary(epitope: str, measures: Iterable[float], measure_name: str) -> List[Dict]:
    records = [
        {"epitope": epitope, "split": idx, measure_name: measure} for idx, measure in enumerate(measures)
    ]
    return records


def save_results(model_name: str, results: Dict[str, DataFrame]) -> None:
    model_dir = RESULTS_DIR/model_name
    model_dir.mkdir(exist_ok=True)
    
    for benchmark_type, results_table in results.items():
        results_table.to_csv(model_dir/f"{benchmark_type}.csv", index=False)


if __name__ == "__main__":
    main()
