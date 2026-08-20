#!/usr/bin/env python3

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from sklearn.ensemble import IsolationForest


# Must be at the top level for ProcessPoolExecutor to pickle it correctly.
def _count_contigs_single_file(file_info):
    """Count contigs in a single FASTA file.

    Parameters
    ----------
    file_info : tuple
        ``(genome_id, filepath)`` pair.

    Returns
    -------
    tuple
        ``(genome_id, contig_count)``; the count is -1 if the file cannot be
        read.
    """
    gid, filepath = file_info
    try:
        with open(filepath, "r") as f:
            count = sum(1 for line in f if line.startswith(">"))
        return gid, count
    except Exception as e:
        print(f"Error reading {gid}: {e}")
        return gid, -1


class GenomeFilter:
    def __init__(self, fasta_dir, extension=".fna"):
        self.fasta_dir = fasta_dir
        self.ext = extension
        self.files = [f for f in os.listdir(fasta_dir) if f.endswith(extension)]
        self.genome_ids = [os.path.splitext(f)[0] for f in self.files]
        print(f"Detected {len(self.genome_ids)} genomes in {fasta_dir}")

    def filter_by_contigs_parallel(self, max_contigs=500, threads=8):
        """Return the IDs of genomes with 1 to ``max_contigs`` contigs."""
        print(f"Counting contigs using {threads} threads...")

        tasks = []
        for f in self.files:
            gid = os.path.splitext(f)[0]
            path = os.path.join(self.fasta_dir, f)
            tasks.append((gid, path))

        keep_ids = set()

        with ProcessPoolExecutor(max_workers=threads) as executor:
            results = executor.map(_count_contigs_single_file, tasks)

            for gid, count in results:
                if 0 < count <= max_contigs:
                    keep_ids.add(gid)

        print(
            f"[Contig Filter] Kept {len(keep_ids)}/{len(self.genome_ids)} genomes (Max Contigs: {max_contigs})"
        )
        return keep_ids

    def run_checkm2_and_filter(
        self,
        valid_ids,
        output_dir="checkm2_out",
        threads=8,
        min_compl=90.0,
        max_contam=5.0,
    ):
        """Run CheckM2 and return the IDs passing the quality thresholds.

        CheckM2 parallelizes internally via its ``--threads`` flag.
        """
        report_file = os.path.join(output_dir, "quality_report.tsv")

        if not os.path.exists(report_file):
            print(
                f"Running CheckM2 on {threads} threads... (This involves ML prediction and may take time)"
            )

            cmd = [
                "checkm2",
                "predict",
                "--threads",
                str(threads),
                "-x",
                self.ext,
                "--input",
                self.fasta_dir,
                "--output-directory",
                output_dir,
                "--force",  # overwrites an existing output directory
            ]

            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError:
                raise RuntimeError(
                    "CheckM2 not found. Is it installed and in your PATH?"
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"CheckM2 failed with error: {e}")
        else:
            print("Found existing CheckM2 report. Skipping run.")

        # CheckM2 report columns: Name, Completeness, Contamination, and others.
        try:
            df = pd.read_csv(report_file, sep="\t")

            # CheckM2 uses 'Name' instead of 'Bin Id'.
            id_col = "Name" if "Name" in df.columns else "Bin Id"

            clean_df = df[
                (df["Completeness"] >= min_compl)
                & (df["Contamination"] <= max_contam)
                & (df[id_col].isin(valid_ids))  # enforce the contig filter
            ]

            final_ids = set(clean_df[id_col].values)
            print(
                f"[CheckM2 Filter] Kept {len(final_ids)}/{len(valid_ids)} genomes (Compl>{min_compl}, Contam<{max_contam})"
            )
            return final_ids

        except Exception as e:
            print(f"Error parsing CheckM2 report: {e}")
            return valid_ids

    def filter_outliers_isoforest(self, feature_matrix, valid_ids, contamination=0.05):
        """Drop feature-matrix outliers with an Isolation Forest.

        Returns the retained rows and the list of outlier genome IDs.
        """
        # Restrict to the genomes that passed the earlier QC steps.
        common_ids = [gid for gid in feature_matrix.index if gid in valid_ids]
        X_clean = feature_matrix.loc[common_ids]

        print(f"Running Isolation Forest on {len(X_clean)} genomes...")

        iso = IsolationForest(
            n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1
        )

        preds = iso.fit_predict(X_clean)

        # IsolationForest labels: 1 = inlier, -1 = outlier.
        X_final = X_clean[preds == 1]
        outliers = X_clean[preds == -1].index.tolist()

        print(f"[IsoForest] Removed {len(outliers)} outliers.")
        print(f"Final High-Quality Dataset: {len(X_final)} genomes.")

        return X_final, outliers


if __name__ == "__main__":
    FASTA_DIR = "./genomes"
    THREADS = 16

    qc_pipeline = GenomeFilter(FASTA_DIR, extension=".fna")

    # Drops highly fragmented assemblies before CheckM2.
    ids_passed_contigs = qc_pipeline.filter_by_contigs_parallel(
        max_contigs=500, threads=THREADS
    )

    # CheckM2 sees the full directory; its results are then filtered by
    # ids_passed_contigs so the fragmented assemblies are ignored.
    ids_passed_checkm = qc_pipeline.run_checkm2_and_filter(
        valid_ids=ids_passed_contigs,
        output_dir="checkm2_results",
        threads=THREADS,
        min_compl=90.0,
        max_contam=5.0,
    )

    # Optional Isolation Forest step, on a KEGG KO matrix (rows=genomes, cols=KOs):
    # df_features = pd.read_csv("kegg_matrix.csv", index_col=0)

    # X_final, outliers = qc_pipeline.filter_outliers_isoforest(
    #    feature_matrix=df_features,
    #    valid_ids=ids_passed_checkm
    # )

    # X_final.to_csv("clean_training_data.csv")
