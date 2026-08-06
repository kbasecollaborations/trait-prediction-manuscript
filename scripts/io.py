"""
Utilities for reading data from files
"""

from collections.abc import Iterable
from pathlib import Path

from trait_prediction.main import (
    FeatureIndex,
    FeatureInput,
    FeatureSet,
    PhenotypeIndex,
    PhenotypeInput,
    PhenotypeSet,
)


def index_format_func(x):
    return (
        x.strip()
        .split("?")[-1]
        .removesuffix(".contigs")
        .removesuffix(".RAST")
        .removesuffix(".fna")
        .removesuffix("_assembly")
        .removeprefix("g")
    )


def read_features(feature_files: Iterable[Path], ftype: str = "binary") -> FeatureSet:
    """Read feature files and create a FeatureSet.

    Parameters
    ----------
    feature_files : Iterable[Path]
        Iterable of paths to feature files to read

    Returns
    -------
    FeatureSet
        Feature set containing the data from all input files
    """
    feature_inputs: list[FeatureInput] = []
    for feature_file in feature_files:
        feature_name = f"{feature_file.parent.stem}_{feature_file.stem}"
        if ftype == "binary":
            dtype = "uint8"
        elif ftype == "count":
            dtype = "uint32"
        elif ftype == "float":
            dtype = "float32"
        elif ftype == "int":
            dtype = "int32"
        else:
            raise ValueError(f"Unknown feature type: {ftype}")
        findex = FeatureIndex(feature_name, ftype=ftype, dtype=dtype)
        finput = FeatureInput(feature_file, findex, index_format_func)
        feature_inputs.append(finput)
    feature_set = FeatureSet.read_data(feature_inputs)
    return feature_set


def read_phenotypes(phenotype_files: Iterable[Path]) -> PhenotypeSet:
    """Read phenotype files and create a PhenotypeSet.

    Parameters
    ----------
    phenotype_files : Iterable[Path]
        Iterable of paths to phenotype files to read

    Returns
    -------
    PhenotypeSet
        Phenotype set containing the data from all input files
    """
    phenotype_inputs: list[PhenotypeInput] = []
    for phenotype_file in phenotype_files:
        phenotype_name = phenotype_file.stem
        phenotype_category = phenotype_file.parent.stem
        pindex = PhenotypeIndex(phenotype_name, phenotype_category)
        pinput = PhenotypeInput(phenotype_file, pindex, index_format_func)
        phenotype_inputs.append(pinput)
    phenotype_set = PhenotypeSet.read_data(phenotype_inputs)
    return phenotype_set


def cache_is_fresh(cache: Path, *inputs: Path) -> bool:
    """Return whether a cached artefact is newer than every input it derives from.

    Compares modification times, since existence alone would silently reuse a
    cache whose inputs have changed.

    Parameters
    ----------
    cache
        Path to the cached artefact.
    *inputs
        Files or directories the cache derives from. Directories are searched
        recursively. Missing inputs are ignored.

    Returns
    -------
    bool
        ``True`` when the cache exists and post-dates every input, otherwise
        ``False`` (meaning the caller should recompute).
    """
    if not cache.exists():
        return False
    cache_mtime = cache.stat().st_mtime
    for source in inputs:
        if not source.exists():
            continue
        if source.is_dir():
            newest = max(
                (f.stat().st_mtime for f in source.rglob("*") if f.is_file()),
                default=0.0,
            )
        else:
            newest = source.stat().st_mtime
        if newest > cache_mtime:
            return False
    return True
