#!/usr/bin/env python3
"""Compatibility entry point for the merged supplementary feature table."""

from pathlib import Path

from scripts.tables.table1 import create_feature_table

if __name__ == "__main__":
    create_feature_table("Histidine", Path("sections/table_feature_comparison.tex"))
