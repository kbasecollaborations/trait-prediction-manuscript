# README for data and scripts

How to run scripts:

```python
python -m scripts.figure1.figure1c_data
```

> [!note]
> Do not include the `.py` extension when running the script as a module.

## Figure 1 (Dataset characteristics)

### Figure 1A (Data collection illustration)

- Inputs
  - None (illustration only)
- Outputs
  - PDF file from Biorender to include in manuscript
- Scripts:
  - None

### Figure 1B (Microbial information: Phylogenetic tree)

- Inputs
  - Phylogenetic tree file (Newick format)
- Outputs
  - ITOL format file for tree visualization
  - Tree visualization from ITOL to include in manuscript
- Scripts:
  - Script to create the file for ITOL import

### Figure 1C (Phenotype information: Phenotype distribution)

- Inputs
  - Phenotype data file containing positive and negatives for all 4 datasets (CSV format)
- Outputs
  - Plot showing phenotype distribution across datasets
- Scripts:
  - Script to create the data file of phenotype distribution
  - Script to create the plot for phenotype distribution

## Figure 2 (Phenotype prediction baselines)

### Figure 2A (GapMind performance)

- Inputs
  - Phenotype data file containing positive and negatives for all 4 datasets (CSV format)
  - GapMind prediction results for all 4 datasets (CSV format)
- Outputs
  - Plot showing GapMind performance across datasets
- Scripts:
  - Script to create the data file for GapMind performance
  - Script to create the plot for GapMind performance

### Figure 2B (Phylogeney-based performance - Nearest Neighbor)

- Inputs
  - Phenotype data file containing positive and negatives for all 4 datasets (CSV format)
  - Phylogenetic tree file (Newick format)
- Outputs
  - Plot showing Nearest Neighbor performance across datasets
- Scripts:
  - Script to create the data file for Nearest Neighbor performance
  - Script to create the plot for Nearest Neighbor performance
