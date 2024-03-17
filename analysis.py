from collections import defaultdict
from datetime import datetime
from few_shot_predictor import FewShotOneVsRestPredictor, FewShotOneInManyPredictor
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pathlib import Path
from pyrepseq.metric import tcr_metric
from pyrepseq.metric.tcr_metric import TcrMetric
import random
from sceptr import variant
from sklearn import metrics
from tqdm import tqdm
from typing import Dict, Iterable, List
import utils


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


PROJECT_ROOT = Path(__file__).parent.resolve()
BENCHMARKS_DIR = PROJECT_ROOT/"benchmarks"

LABELLED_DATA = pd.read_csv("tcr_data/preprocessed/benchmarking/vdjdb_cleaned.csv")
EPITOPES = LABELLED_DATA.Epitope.unique()

MODELS = (
    # tcr_metric.Cdr3Levenshtein(),
    # tcr_metric.Tcrdist(),
    # variant.ab_sceptr(),
    variant.ab_sceptr_blosum(),
    # variant.ab_sceptr_large(),
)

NUM_SHOTS = (10, 100, 200)
NUM_RANDOM_FOLDS = 100


def main() -> None:
    BENCHMARKS_DIR.mkdir(exist_ok=True)

    results_per_model = {
        model.name: get_results(model) for model in MODELS
    }

    for model_name, results in results_per_model.items():
        save_results(model_name, results)


def get_results(model: TcrMetric) -> Dict[str, DataFrame]:
    return {
        **get_one_vs_rest_one_shot_results(model),
        **get_one_vs_rest_few_shot_results(model),
        **get_one_in_many_results(model)
    }


def get_one_vs_rest_one_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing OVR[1-shot] for {model.name}...")

    results = []
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = LABELLED_DATA.Epitope == epitope
        epitope_references = LABELLED_DATA[labelled_data_epitope_mask]
        cdist_matrix = model.calc_cdist_matrix(LABELLED_DATA, epitope_references)

        aucs = []
        for cdist_idx, tcr_idx in enumerate(epitope_references.index):
            dists = cdist_matrix[:,cdist_idx]
            similarities = utils.convert_dists_to_scores(dists)

            similarities = np.delete(similarities, tcr_idx)
            ground_truth = np.delete(labelled_data_epitope_mask, tcr_idx)

            aucs.append(metrics.roc_auc_score(ground_truth, similarities))

        auc_summary = generate_summary(epitope, aucs, "auc")
        results.append(auc_summary)
    
    return {"one_vs_rest_1_shot": DataFrame.from_records(results)}


def get_one_vs_rest_few_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
    results = dict()
    for num_shots in NUM_SHOTS:
        k_shot_results = get_one_vs_rest_k_shot_results(model, k=num_shots)
        results.update(k_shot_results)
    return results


def get_one_vs_rest_k_shot_results(model: TcrMetric, k: int) -> Dict[str, DataFrame]:
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

        auc_summaries = get_one_vs_rest_k_shot_auc_summaries_for_epitope(model, epitope, ref_index_sets)

        nn_results.append(auc_summaries["nn"])
        avg_dist_results.append(auc_summaries["avg_dist"])

    return {
        f"one_vs_rest_{k}_shot_nn": DataFrame.from_records(nn_results),
        f"one_vs_rest_{k}_shot_avg_dist": DataFrame.from_records(avg_dist_results),
    }


def get_one_vs_rest_k_shot_auc_summaries_for_epitope(model: TcrMetric, epitope: str, ref_index_sets: Iterable[List[int]]) -> Dict[str, Dict[str, float]]:
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


def get_one_in_many_results(model: TcrMetric) -> Dict[str, DataFrame]:
    results = dict()
    for num_shots in NUM_SHOTS:
        k_shot_results = get_one_in_many_k_shot_results(model, k=num_shots)
        results.update(k_shot_results)
    return results


def get_one_in_many_k_shot_results(model: TcrMetric, k: int) -> Dict[str, DataFrame]:
    print(f"Commencing OIM[{k}-shot] for {model.name}...")

    nn_avg_ranks = defaultdict(list)
    avg_dist_avg_ranks = defaultdict(list)

    labelled_data_with_large_enough_epitope_groups = LABELLED_DATA.groupby("Epitope").filter(lambda group: len(group) - k >= 100)
    labelled_data_grouped_by_epitope = labelled_data_with_large_enough_epitope_groups.groupby("Epitope")

    for random_seed in tqdm(range(NUM_RANDOM_FOLDS)):
        positive_refs = labelled_data_grouped_by_epitope.sample(n=k, replace=False, random_state=random_seed)
        queries = labelled_data_with_large_enough_epitope_groups[
            ~labelled_data_with_large_enough_epitope_groups.index.isin(positive_refs.index)
        ]
        assert len(set(positive_refs.index).intersection(set(queries.index))) == 0

        if random_seed == 0:
            log_sample_indices(positive_refs, k, model)

            positive_refs_summary = positive_refs.groupby("Epitope").size()
            queries_summary = queries.groupby("Epitope").size()

            logging.info(f"\n{positive_refs_summary}")
            logging.info(f"\n{queries_summary}")

        predictor = FewShotOneInManyPredictor(model, positive_refs, queries)

        nn_scores_table = predictor.get_nn_inferences()
        avg_dist_scores_table = predictor.get_avg_dist_inferences()

        nn_avg_rank_per_epitope = get_avg_rank_per_epitope(nn_scores_table, queries["Epitope"])
        avg_dist_avg_rank_per_epitope = get_avg_rank_per_epitope(avg_dist_scores_table, queries["Epitope"])

        for epitope, avg_rank in nn_avg_rank_per_epitope.items():
            nn_avg_ranks[epitope].append(avg_rank)
        
        for epitope, avg_rank in avg_dist_avg_rank_per_epitope.items():
            avg_dist_avg_ranks[epitope].append(avg_rank)
    
    nn_summaries = [
        generate_summary(epitope, avg_ranks, "avg_rank") for epitope, avg_ranks in nn_avg_ranks.items()
    ]
    avg_dist_summaries = [
        generate_summary(epitope, avg_ranks, "avg_rank") for epitope, avg_ranks in avg_dist_avg_ranks.items()
    ]

    return {
        f"one_in_many_{k}_shot_nn": DataFrame.from_records(nn_summaries),
        f"one_in_many_{k}_shot_avg_dist": DataFrame.from_records(avg_dist_summaries)
    }


def generate_summary(epitope: str, measures: Iterable[float], measure_name: str) -> Dict[str, float]:
    return {
        "epitope": epitope,
        f"mean_{measure_name}": np.mean(measures),
        f"std_{measure_name}": np.std(measures),
        f"median_{measure_name}": np.median(measures),
        f"upper_quartile_{measure_name}": np.quantile(measures, q=0.75),
        f"lower_quartile_{measure_name}": np.quantile(measures, q=0.25),
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


def log_sample_indices(df: DataFrame, k: int, model: TcrMetric) -> None:
    grouped_by_epitope = df.groupby("Epitope")
    for epitope, indices in grouped_by_epitope.groups.items():
        logging.info(f"{model.name}:OIM[{k}-shot]:{epitope}: {indices[:2].to_list()}")


def save_results(model_name: str, results: Dict[str, DataFrame]) -> None:
    model_dir = BENCHMARKS_DIR/model_name
    model_dir.mkdir(exist_ok=True)
    
    for benchmark_type, results_table in results.items():
        results_table.to_csv(model_dir/f"{benchmark_type}.csv", index=False)


if __name__ == "__main__":
    main()
