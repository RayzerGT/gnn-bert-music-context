# gnn-bert-music-context

This repository implements the GNN + BERT music context understanding project described in `Project Details/425 project outline.pdf`.

## Project structure

- `data/raw/` - raw dataset downloads for FMA, MagnaTagATune, MusicCaps, DEAM, etc.
- `data/processed/` - preprocessed audio features, graphs, BERT caches, and derived data.
- `data/splits/` - train/validation/test split files and metadata.
- `notebooks/` - exploratory data analysis and demo inference notebooks.
- `src/` - source code for feature extraction, graph construction, model definitions, training, and evaluation.
- `results/` - experiment outputs, metrics, plots, and retrieval examples.
- `report/` - final report and submission artifacts.

## Task 1 outputs

The Task 1 training script now produces a complete baseline artifact bundle in `results/task1/`:

- `test_metrics.json` - final micro/macro F1 on the held-out test split
- `training_history.json` and `training_history.csv` - per-epoch loss and validation F1 values
- `f1_curves.png` - plot of validation micro/macro F1 across epochs
- `example_predictions.json` - five example predictions with true vs predicted tags
- `model/` and `tokenizer/` - saved trained model and tokenizer weights
- `summary.json` - high-level experiment summary
