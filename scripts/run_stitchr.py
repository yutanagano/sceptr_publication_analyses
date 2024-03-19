import subprocess
import pandas as pd
from pandas import Series
from pathlib import Path


PROJECT_ROOT = Path.cwd().parent.resolve()
BENCHMARKING_DATA_DIR = PROJECT_ROOT/"tcr_data"/"preprocessed"/"benchmarking"


def main():
    vdjdb_cleaned = pd.read_csv(BENCHMARKING_DATA_DIR/"vdjdb_cleaned.csv")

    tras = vdjdb_cleaned[["TRAV", "CDR3A", "TRAJ"]].drop_duplicates(ignore_index=True)
    trbs = vdjdb_cleaned[["TRBV", "CDR3B", "TRBJ"]].drop_duplicates(ignore_index=True)

    tras.columns = ["v", "cdr3", "j"]
    trbs.columns = ["v", "cdr3", "j"]

    tcr_chains = pd.concat([tras, trbs], ignore_index=True)
    tcr_chains["aa"] = tcr_chains.apply(row_to_tcr_chain, axis=1)
    tcr_chains = tcr_chains.dropna() # drop TCR chains for which stitchr failed

    tcr_chains.to_csv(BENCHMARKING_DATA_DIR/"vdjdb_cleaned_tcr_chain_aa_seqs.csv", index=False)


def row_to_tcr_chain(row: Series) -> str:
    stitchr_output = subprocess.run(
        [
            "stitchr",
            "-v", row.v,
            "-j", row.j,
            "-cdr3", row.cdr3,
            "-m", "AA"
        ],
        capture_output=True
    )
    stdout_cleaned = stitchr_output.stdout.decode().strip()
    return stdout_cleaned


if __name__ == "__main__":
    main()