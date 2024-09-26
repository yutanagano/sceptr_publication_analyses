from datetime import datetime
from few_shot_predictor import FewShotOneVsRestPredictor
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
from tqdm import tqdm
from typing import Dict, Iterable, List, Literal
import utils


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


LABELLED_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"benchmarking"/"vdjdb_cleaned.csv")
EPITOPES = LABELLED_DATA.Epitope.unique()

SCEPTR_VARIANTS = (
    CachedRepresentationModel(variant.default()),
    CachedRepresentationModel(variant.dropout_noise_only()),
)

NUM_SHOTS = (2, 5, 10, 20, 50, 100, 200)
NUM_RANDOM_FOLDS = 100


def main() -> None:
    results_per_model = dict()
    # results_per_model["TCRdist"] = get_results(tcr_metric.AlphaTcrdist(), "a")
    # results_per_model["TCRdist"].update(get_results(tcr_metric.BetaTcrdist(), "b"))
    # results_per_model["SCEPTR"] = get_results(CachedRepresentationModel(variant.default()), "a")
    # results_per_model["SCEPTR"].update(get_results(CachedRepresentationModel(variant.default()), "b"))
    results_per_model["SCEPTR (dropout noise only)"] = get_results(CachedRepresentationModel(variant.dropout_noise_only()), "a")
    results_per_model["SCEPTR (dropout noise only)"].update(get_results(CachedRepresentationModel(variant.dropout_noise_only()), "b"))

    for model_name, results in results_per_model.items():
        save_results(model_name, results)
    

def get_results(model: TcrMetric, chain: Literal["a", "b"]) -> Dict[str, DataFrame]:
    return {
        **get_one_shot_results(model, chain),
        **get_few_shot_results(model, chain),
    }


def get_one_shot_results(model: TcrMetric, chain: Literal["a", "b"]) -> Dict[str, DataFrame]:
    print(f"Commencing OVR {chain} chain [1-shot] for {model.name}...")

    if chain == "a":
        single_chain_labelled_data = LABELLED_DATA[["TRAV","CDR3A","TRAJ","Epitope"]].copy()
        single_chain_labelled_data[["TRBV","CDR3B","TRBJ"]] = None
    elif chain == "b":
        single_chain_labelled_data = LABELLED_DATA[["TRBV","CDR3B","TRBJ","Epitope"]].copy()
        single_chain_labelled_data[["TRAV","CDR3A","TRAJ"]] = None

    results = []
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = single_chain_labelled_data.Epitope == epitope
        epitope_references = single_chain_labelled_data[labelled_data_epitope_mask]

        if len(epitope_references) - 1 < 100:
            logging.debug(f"Not enough references for {epitope}, skipping")
            continue

        cdist_matrix = model.calc_cdist_matrix(single_chain_labelled_data, epitope_references)

        aucs = []
        for cdist_idx, tcr_idx in enumerate(epitope_references.index):
            dists = cdist_matrix[:,cdist_idx]
            similarities = utils.convert_dists_to_scores(dists)

            similarities = np.delete(similarities, tcr_idx)
            ground_truth = np.delete(labelled_data_epitope_mask, tcr_idx)

            aucs.append(metrics.roc_auc_score(ground_truth, similarities))

        auc_summary = generate_summary(epitope, aucs, "auc")
        results.extend(auc_summary)
    
    return {
        f"ovr_{chain}_1_shot": DataFrame.from_records(results)
    }


def get_few_shot_results(model: TcrMetric, chain: Literal["a", "b"]) -> Dict[str, DataFrame]:
    results = dict()
    for num_shots in NUM_SHOTS:
        k_shot_results = get_distance_based_k_shot_results(model, chain, k=num_shots)
        results.update(k_shot_results)
    return results


def get_distance_based_k_shot_results(model: TcrMetric, chain: Literal["a","b"], k: int) -> Dict[str, DataFrame]:
    print(f"Commencing OVR {chain} chain [{k}-shot] for {model.name}...")

    if chain == "a":
        single_chain_labelled_data = LABELLED_DATA[["TRAV","CDR3A","TRAJ","Epitope"]].copy()
        single_chain_labelled_data[["TRBV","CDR3B","TRBJ"]] = None
    elif chain == "b":
        single_chain_labelled_data = LABELLED_DATA[["TRBV","CDR3B","TRBJ","Epitope"]].copy()
        single_chain_labelled_data[["TRAV","CDR3A","TRAJ"]] = None

    results = []
    
    for epitope in tqdm(EPITOPES):
        labelled_data_epitope_mask = single_chain_labelled_data.Epitope == epitope
        epitope_references = single_chain_labelled_data[labelled_data_epitope_mask]
        ref_index_sets = epitope_references.index.to_list()

        if len(epitope_references) - k < 100:
            logging.debug(f"Not enough references for {epitope} for {k} shots, skipping")
            continue

        random.seed("tcrsarecool")
        ref_index_sets = [
            random.sample(ref_index_sets, k=k) for _ in range(NUM_RANDOM_FOLDS)
        ]

        aucs = []
        for ref_index_set in ref_index_sets:
            positive_refs = single_chain_labelled_data.loc[ref_index_set]
            queries = single_chain_labelled_data.drop(index=ref_index_set)
            ground_truth = queries.Epitope == epitope
            predictor = FewShotOneVsRestPredictor(model, positive_refs=positive_refs, queries=queries)

            scores = predictor.get_nn_inferences()
            auc = metrics.roc_auc_score(ground_truth, scores)

            aucs.append(auc)

        results.extend(generate_summary(epitope, aucs, "auc"))

    return {
        f"ovr_nn_{chain}_{k}_shot": DataFrame.from_records(results),
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