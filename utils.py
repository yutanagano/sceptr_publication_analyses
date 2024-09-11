from edit_penalty import EditPenaltyCollection, EditPenaltyCollectionAnalyser
from matplotlib.axes import Axes
import pandas as pd
from pandas import DataFrame, Series
from paths import RESULTS_DIR
import numpy as np
from numpy import ndarray
from typing import Optional, Iterable


class ModelForAnalysis:
    def __init__(self, model_name: str, task_prefix: str, colour: Optional[str] = None, marker: str = "", linestyle: str = "-", zorder: Optional[float] = 1.5, ten_x: bool = False) -> None:
        self.name = model_name
        self.task_prefix = task_prefix
        self.colour = colour
        self.marker = marker
        self.linestyle = linestyle
        self.zorder = zorder
        self.ten_x = ten_x
    
    @property
    def style(self) -> str:
        return self.marker + self.linestyle
    
    def load_data(self, k: Optional[int] = None) -> DataFrame:
        if self.task_prefix in ("ovr_nn", "ovr_avg_dist") and k == 1:
            csv_name = f"ovr_1_shot.csv"
        elif self.task_prefix in ("ovr_unseen_epitopes_nn", "ovr_unseen_epitopes_avg_dist") and k == 1:
            csv_name = f"ovr_unseen_epitopes_1_shot.csv"
        elif k is None:
            csv_name = self.task_prefix + ".csv"
        else:
            csv_name = f"{self.task_prefix}_{k}_shot.csv"

        if self.ten_x:
            path_to_csv = RESULTS_DIR/"10x"/self.name/f"{csv_name}"
        else:
            path_to_csv = RESULTS_DIR/self.name/f"{csv_name}"
        
        return pd.read_csv(path_to_csv)
    
    def load_epc_analyser(self) -> EditPenaltyCollectionAnalyser:
        path_to_epc_state_dict = RESULTS_DIR/self.name/"epc_state_dict.pkl"
        with open(path_to_epc_state_dict, "rb") as f:
            epc = EditPenaltyCollection.from_save(f)
        return EditPenaltyCollectionAnalyser(epc)
    
    def get_num_parameters(self) -> int:
        with open(RESULTS_DIR/self.name/"model_parameter_count.txt", "r") as f:
            count = f.read()
            return int(count)

    def get_model_dimensionality(self) -> int:
        with open(RESULTS_DIR/self.name/"model_dimensionality.txt", "r") as f:
            dims = f.read()
            return int(dims)


def convert_dists_to_scores(dists: ndarray) -> ndarray:
    max_dist = np.max(dists)
    return 1 - dists / max_dist


def get_benchmark_summary_with_errorbars(
    models: Iterable[ModelForAnalysis],
    ks: Iterable[int],
    epitopes: Iterable[str],
):
    mean_std_collection = []

    for k in ks:
        model_dfs = [(model.name, model.load_data(k)) for model in models]
        summary_df = pd.DataFrame()
        summary_df["epitope"] = model_dfs[0][1]["epitope"]
        summary_df["split"] = model_dfs[0][1]["split"]

        for model_name, model_df in model_dfs:
            summary_df[model_name] = model_df["auc"]
        
        summary_df = summary_df[summary_df["epitope"].map(lambda ep: ep in epitopes)]

        # get average performance across epitopes per model
        avg_performance_df = summary_df.groupby("epitope").aggregate({model.name: "mean" for model in models})
        avg_performances = avg_performance_df.mean()

        # get error bars across epitopes per model
        model_averages = summary_df.apply(
            lambda row: np.mean(row.iloc[2:]),
            axis="columns"
        )

        delta_df = summary_df.copy()
        for model in models:
            delta_df[model.name] = delta_df[model.name] - model_averages

        variance_by_epitope = delta_df.groupby("epitope").apply(
            lambda df: Series(data=(df[model.name].var() for model in models), index=(model.name for model in models)),
            include_groups=False
        )
        stds = np.sqrt(variance_by_epitope.sum()) / len(epitopes)

        # append to mean_std collection
        mean_std_df = pd.DataFrame(data=(avg_performances, stds), index=("mean", "std"))
        mean_std_collection.append(mean_std_df.T.stack())

    return pd.DataFrame(mean_std_collection, index=ks)


def plot_performance_curves(models: Iterable[ModelForAnalysis], ks: Iterable[int], epitopes: Iterable[str], ax: Axes):
    benchmark_summary = get_benchmark_summary_with_errorbars(models, ks, epitopes)

    for model in models:
        mean_stds_for_model = benchmark_summary[model.name]
        ax.errorbar(
            x=range(len(ks)),
            y=mean_stds_for_model["mean"],
            yerr=mean_stds_for_model["std"],
            fmt=model.style,
            markersize=5,
            c=model.colour,
            label=model.name,
            zorder=model.zorder,
            capsize=5
        )

    ax.set_ylabel("Mean AUROC")
    ax.set_xlabel("Number of Reference TCRs")
    ax.set_xticks(range(len(ks)), ks)

    return ax