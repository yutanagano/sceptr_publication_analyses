from datetime import datetime
from few_shot_predictor import FewShotPredictor
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
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
BACKGROUND_DATA = pd.read_csv("tcr_data/preprocessed/tanno/test.csv")
BACKGROUND_SAMPLE = BACKGROUND_DATA.sample(n=100,random_state=420)

EPITOPES = LABELLED_DATA.Epitope.unique()

MODELS = (
    tcr_metric.Cdr3Levenshtein(),
    tcr_metric.Tcrdist(),
    variant.ab_sceptr()
)


def main() -> None:
    BENCHMARKS_DIR.mkdir(exist_ok=True)

    results_per_model = {
        model.name: get_results(model) for model in MODELS
    }

    for model_name, results in results_per_model.items():
        save_results(model_name, results)


def get_results(model: TcrMetric) -> Dict[str, DataFrame]:
    one_shot_results = get_one_shot_results(model)
    few_shot_results = get_few_shot_results(model)
    collated_results = {**one_shot_results, **few_shot_results}
    return collated_results


def get_one_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing one-shot benchmarking for {model.name}...")

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

        auc_summary = get_auc_summary(epitope, aucs)
        results.append(auc_summary)
    
    return {"one_shot": DataFrame.from_records(results)}


def get_auc_summary(epitope: str, aucs: Iterable[float]) -> Dict[str, float]:
    return {
        "epitope": epitope,
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "median_auc": np.median(aucs),
        f"upper_quartile": np.quantile(aucs, q=0.75),
        f"lower_quartile": np.quantile(aucs, q=0.25),
    }


def get_few_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing few-shot benchmarking for {model.name}...")

    results = dict()
    for num_shots in (10,100,200):
        k_shot_results = get_k_shot_results(model, k=num_shots)
        results.update(k_shot_results)
    return results


def get_k_shot_results(model: TcrMetric, k: int) -> Dict[str, DataFrame]:
    nn_results = []
    avg_dist_results = []
    # svc_results = []
    # mlp_results = []
    
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = LABELLED_DATA.Epitope == epitope
        epitope_references = LABELLED_DATA[labelled_data_epitope_mask]
        ref_index_sets = epitope_references.index.to_list()

        if len(epitope_references) - k < 100:
            logging.debug(f"Not enough references for {epitope} for {k} shots, skipping")
            continue

        random.seed("tcrsarecool")
        ref_index_sets = [
            random.sample(ref_index_sets, k=k) for _ in range(100)
        ]
        logging.info(f"The first ten indices in the first {k}-shot reference set for {epitope} for {model.name} are: {ref_index_sets[0][:10]}")

        auc_summaries = get_k_shot_auc_summaries_for_epitope(model, epitope, ref_index_sets)

        nn_results.append(auc_summaries["nn"])
        avg_dist_results.append(auc_summaries["avg_dist"])
        # svc_results.append(auc_summaries["svc"])

        # if "mlp" in auc_summaries:
        #     mlp_results.append(auc_summaries["mlp"])

    return {
        f"{k}_shot_nn": DataFrame.from_records(nn_results),
        f"{k}_shot_avg_dist": DataFrame.from_records(avg_dist_results),
        # f"{k}_shot_svc": DataFrame.from_records(svc_results),
        # f"{k}_shot_mlp": DataFrame.from_records(mlp_results)
    }


def get_k_shot_auc_summaries_for_epitope(model: TcrMetric, epitope: str, ref_index_sets: Iterable[List[int]]) -> Dict[str, Dict[str, float]]:
    nn_aucs = []
    avg_dist_aucs = []
    svc_aucs = []
    
    for ref_index_set in ref_index_sets:
        positive_refs = LABELLED_DATA.loc[ref_index_set]
        queries = LABELLED_DATA.drop(index=ref_index_set)
        ground_truth = queries.Epitope == epitope
        predictor = FewShotPredictor(model, positive_refs=positive_refs, bg_refs=BACKGROUND_SAMPLE)

        nn_scores = predictor.get_nn_inferences(queries)
        avg_dist_scores = predictor.get_avg_dist_inferences(queries)
        # svc_scores = predictor.get_svc_inferences(queries)

        nn_auc = metrics.roc_auc_score(ground_truth, nn_scores)
        avg_dist_auc = metrics.roc_auc_score(ground_truth, avg_dist_scores)
        # svc_auc = metrics.roc_auc_score(ground_truth, svc_scores)

        nn_aucs.append(nn_auc)
        avg_dist_aucs.append(avg_dist_auc)
        # svc_aucs.append(svc_auc)
    
    return {
        "nn": get_auc_summary(epitope, nn_aucs),
        "avg_dist": get_auc_summary(epitope, avg_dist_aucs),
        # "svc": get_auc_summary(epitope, svc_aucs)
    }


def save_results(model_name: str, results: Dict[str, DataFrame]) -> None:
    model_dir = BENCHMARKS_DIR/model_name
    model_dir.mkdir(exist_ok=True)
    
    for benchmark_type, results_table in results.items():
        results_table.to_csv(model_dir/f"{benchmark_type}.csv", index=False)


if __name__ == "__main__":
    main()
