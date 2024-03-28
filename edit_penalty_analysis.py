from cached_representation_model import CachedRepresentationModel
from datetime import datetime
import edit_penalty
from edit_penalty import EditPenaltyCollection
from hugging_face_lms import TcrBert
import logging
import pandas as pd
from paths import DATA_DIR, RESULTS_DIR
from sceptr import variant


BACKGROUND_DATA = pd.read_csv(DATA_DIR/"preprocessed"/"tanno"/"test.csv")

MODELS = (
    variant.ab_sceptr(),
    TcrBert(),
)


current_time = datetime.now().isoformat()
logging.basicConfig(filename=f"log_{current_time}.txt", level=logging.INFO)


def main():
    epc_per_model = {
        model.name: explore_edit_penalties(model) for model in MODELS
    }

    for model_name, edit_penalty_collection in epc_per_model.items():
        model_dir = RESULTS_DIR/model_name
        model_dir.mkdir(exist_ok=True)
        with open(model_dir/"epc_state_dict.pkl", "wb") as f:
            edit_penalty_collection.save(f)
    
    for model in MODELS:
        if isinstance(model, CachedRepresentationModel):
            model.save_cache()


def explore_edit_penalties(model) -> EditPenaltyCollection:
    print(f"Exploring edit penalties for {model.name}...")

    edit_penalty_collection = EditPenaltyCollection()
    num_tcrs_processed = 0

    while not (edit_penalty_collection.has_sufficient_coverage() and num_tcrs_processed >= 1000):
        tcr = BACKGROUND_DATA.sample(n=1)
        tcr_variants = edit_penalty.get_all_tcr_variants(tcr)
        distances = model.calc_cdist_matrix(tcr, tcr_variants).squeeze()
        edits_and_resulting_distances = zip(tcr_variants["edit"], distances)

        for edit, distance in edits_and_resulting_distances:
            edit_penalty_collection.update_edit_record(edit, distance)
        
        num_tcrs_processed += 1

        if (num_tcrs_processed % 10) == 0:
            print(f"{num_tcrs_processed} TCRs processed...")
            edit_penalty_collection.print_current_estimation_coverage()
    
    return edit_penalty_collection


if __name__ == "__main__":
    main()