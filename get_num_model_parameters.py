from pandas import DataFrame
from paths import RESULTS_DIR
import hugging_face_lms
from sceptr import variant as sceptr_variant
from torch.nn import Module


dummy_tcr = DataFrame(
    {
        "TRAV": "TRAV1-1",
        "CDR3A": "CASQYF",
        "TRAJ": "TRAJ1",
        "TRBV": "TRBV2",
        "CDR3B": "CASQYF",
        "TRBJ": "TRBJ1-1"
    },
    index=[0]
)


def main():
    sceptr = sceptr_variant.default()
    tcr_bert = hugging_face_lms.TcrBert()
    esm2 = hugging_face_lms.Esm2()
    protbert = hugging_face_lms.ProtBert()

    save_complexity(sceptr)
    save_complexity(tcr_bert)
    save_complexity(esm2)
    save_complexity(protbert)


def save_complexity(model):
    if model.name == "SCEPTR":
        model_parameter_count = get_num_parameters(model._bert)
    else:
        model_parameter_count = get_num_parameters(model._model)
    
    model_dimensionality = get_model_dimensionality(model)

    model_dir = RESULTS_DIR/model.name
    model_dir.mkdir(exist_ok=True)

    with open(model_dir/"model_parameter_count.txt", "w") as f:
        f.write(str(model_parameter_count))
    
    with open(model_dir/"model_dimensionality.txt", "w") as f:
        f.write(str(model_dimensionality))


def get_num_parameters(module: Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_model_dimensionality(model) -> int:
    rep = model.calc_vector_representations(dummy_tcr)
    return rep.shape[1]


if __name__ == "__main__":
    main()