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
from hugging_face_lms import TcrBert
import tidytcells as tt
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
    tcr_metric.Cdr3Levenshtein(),
    tcr_metric.Tcrdist(),
    CachedRepresentationModel(variant.default()),
    CachedRepresentationModel(variant.finetuned()),
    CachedRepresentationModel(TcrBert()),
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
        **get_seen_pmhc_results(model),
        **get_seen_pmhc_results_filtered(model),
        **get_unseen_pmhc_one_shot_results(model),
        **get_unseen_pmhc_few_shot_results(model)
    }


def get_seen_pmhc_results(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing ovr (predetermined split) for {model.name}...")

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
        f"ovr_predetermined_split_nn": DataFrame.from_records(nn_results),
        f"ovr_predetermined_split_avg_dist": DataFrame.from_records(avg_dist_results),
    }


def get_seen_pmhc_results_filtered(model: TcrMetric) -> Dict[str, DataFrame]:
    print(f"Commencing ovr (predetermined split, filtered) for {model.name}...")

    test_data_filtered = filter_for_sequence_similarity(TEST_DATA_DISCRIMINATION, TRAIN_DATA)

    nn_results = []
    avg_dist_results = []

    train_data_grouped_by_epitope = TRAIN_DATA.groupby("Epitope")
    for epitope, tcr_indices in train_data_grouped_by_epitope.groups.items():
        epitope_references = TRAIN_DATA.loc[tcr_indices]
        ground_truth = test_data_filtered.Epitope == epitope
        predictor = FewShotOneVsRestPredictor(model, positive_refs=epitope_references, queries=test_data_filtered)

        nn_scores = predictor.get_nn_inferences()
        avg_dist_scores = predictor.get_avg_dist_inferences()

        nn_auc = metrics.roc_auc_score(ground_truth, nn_scores)
        avg_dist_auc = metrics.roc_auc_score(ground_truth, avg_dist_scores)

        nn_results.append({"epitope": epitope, "auc": nn_auc})
        avg_dist_results.append({"epitope": epitope, "auc": avg_dist_auc})

    return {
        f"ovr_predetermined_split_filtered_nn": DataFrame.from_records(nn_results),
        f"ovr_predetermined_split_filtered_avg_dist": DataFrame.from_records(avg_dist_results),
    }


def get_unseen_pmhc_one_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
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


def get_unseen_pmhc_few_shot_results(model: TcrMetric) -> Dict[str, DataFrame]:
    results = dict()
    for num_shots in NUM_SHOTS:
        k_shot_results = get_unseen_pmhc_k_shot_results(model, k=num_shots)
        results.update(k_shot_results)
    return results


def get_unseen_pmhc_k_shot_results(model: TcrMetric, k: int) -> Dict[str, DataFrame]:
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

        auc_summaries = get_unseen_pmhc_k_shot_auc_summaries_for_epitope(model, epitope, ref_index_sets)

        nn_results.extend(auc_summaries["nn"])
        avg_dist_results.extend(auc_summaries["avg_dist"])

    return {
        f"ovr_unseen_epitopes_nn_{k}_shot": DataFrame.from_records(nn_results),
        f"ovr_unseen_epitopes_avg_dist_{k}_shot": DataFrame.from_records(avg_dist_results),
    }


def get_unseen_pmhc_k_shot_auc_summaries_for_epitope(model: TcrMetric, epitope: str, ref_index_sets: Iterable[List[int]]) -> Dict[str, List[Dict]]:
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


def filter_for_sequence_similarity(test_df: DataFrame, train_df: DataFrame) -> DataFrame:    
    # Get cdist
    cdr3_levenshtein = tcr_metric.Cdr3Levenshtein()
    cdist_matrix = cdr3_levenshtein.calc_cdist_matrix(test_df, train_df)

    # Preclude TCRs with different V genes
    test_travs = test_df.TRAV.map(lambda x: tt.tr.standardize(x, precision="gene"))
    test_trbvs = test_df.TRBV.map(lambda x: tt.tr.standardize(x, precision="gene")) 
    train_travs = train_df.TRAV.map(lambda x: tt.tr.standardize(x, precision="gene"))
    train_trbvs = train_df.TRBV.map(lambda x: tt.tr.standardize(x, precision="gene")) 

    same_trav = np.empty_like(cdist_matrix)
    for i, anch_trav in enumerate(test_travs):
        for j, comp_trav in enumerate(train_travs):
            same_trav[i,j] = anch_trav == comp_trav
    
    same_trbv = np.empty_like(cdist_matrix)
    for i, anch_trbv in enumerate(test_trbvs):
        for j, comp_trbv in enumerate(train_trbvs):
            same_trbv[i,j] = anch_trbv == comp_trbv

    different_v_genes = 1 - (same_trav * same_trbv)

    updated_cdist = cdist_matrix + 99999 * different_v_genes

    # Get nearest neighbour distances
    nn_dists = np.min(updated_cdist, axis=1)

    # Calculate combined CDR3 lengths
    combined_cdr3_length = test_df.apply(
        lambda row: len(row.CDR3A) + len(row.CDR3B),
        axis='columns'
    ).to_numpy()

    # Calculate sequence identity
    seq_identity = 1 - (nn_dists/combined_cdr3_length)
    legal_test_seq_mask = seq_identity < 0.95

    # Return filtered df
    return test_df[legal_test_seq_mask]


if __name__ == "__main__":
    main()
