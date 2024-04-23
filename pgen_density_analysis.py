from datetime import datetime
import logging
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from paths import RESULTS_DIR, DATA_DIR
from pyrepseq.metric import tcr_metric
from cached_representation_model import CachedRepresentationModel
from sceptr import variant
from tqdm import tqdm


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


bg_data = pd.read_csv(DATA_DIR/"preprocessed"/"tanno"/"test.csv")
BACKGROUND_SAMPLE = bg_data.sample(n=10_000, random_state=420)

MODELS = (
    CachedRepresentationModel(variant.default()),
    tcr_metric.Tcrdist(),
)


def main():
    results_per_model = {
        model.name: get_results(model) for model in MODELS
    }

    for model_name, results in results_per_model.items():
        save_results(model_name, results)

    for model in MODELS:
        if isinstance(model, CachedRepresentationModel):
            model.save_cache()


def get_results(model) -> DataFrame:
    print(f"Computing pgen density estimates for {model.name}...")

    results = DataFrame()
    results["pgen"] = BACKGROUND_SAMPLE.apply(
        lambda row: row["alpha_pgen"] * row["beta_pgen"],
        axis="columns"
    )

    batch_size = 100
    k = 10
    all_nn_dists = []
    for idx in tqdm(range(0, len(BACKGROUND_SAMPLE), batch_size)):
        batch = BACKGROUND_SAMPLE.iloc[idx:idx+batch_size]
        dists: ndarray = model.calc_cdist_matrix(batch, BACKGROUND_SAMPLE)
        nn_dists = np.partition(dists, kth=k, axis=1)[:,:k]
        nn_dists_averaged = nn_dists.mean(axis=1)
        all_nn_dists.append(nn_dists_averaged)
    results["nn_dist"] = np.concatenate(all_nn_dists)

    return results


def save_results(model_name: str, results: DataFrame) -> None:
    model_dir = RESULTS_DIR/model_name
    model_dir.mkdir(exist_ok=True)
    
    results.to_csv(model_dir/f"pgen_nn_dists.csv", index=False)


if __name__ == "__main__":
    main()