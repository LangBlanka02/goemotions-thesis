# Problem-Driven Analysis of Textual Emotion Classification

This repository contains the source code and reproducibility artefacts for the license thesis:

**Problem-Driven Analysis of Textual Emotion Classification: Baselines and Class-Imbalance Interventions on a Single-Label Reformulation of the GoEmotions Dataset**

Author: Blanka-Maria Lang

## Project overview

The project evaluates textual emotion classification on a single-label reformulation of the GoEmotions dataset. It follows a four-phase experimental design:

1. Phase 1: TF-IDF + Logistic Regression on the original 28-label setting.
2. Phase 2: TF-IDF + Logistic Regression on the Ekman-7 setting.
3. Phase 3: Standard DistilBERT fine-tuning on the Ekman-7 setting.
4. Phase 4: DistilBERT with class-imbalance interventions:
   - class-weighted cross-entropy,
   - focal loss,
   - back-translation augmentation.

The final selected model is the Phase 3 standard DistilBERT baseline, because it obtains the best validation macro-F1 under the predefined model-selection protocol.

## Repository structure

- `utils.py`: shared constants, label lists, project paths and seed utilities.
- `preprocessing.py`: GoEmotions loading, single-label filtering and preprocessing.
- `evaluate.py`: shared metric and evaluation functions.
- `train_phase3.py`: reusable DistilBERT training components.
- `train_classical.ipynb`: Phase 1 and Phase 2 classical baselines.
- `train_phase3.ipynb`: Phase 3 standard DistilBERT baseline.
- `train_phase4.ipynb`: Phase 4 imbalance-handling experiments.
- `phase4_class_weight_controls.ipynb`: additional class-weighting controls.
- `phase4_alpha_focal_controls.ipynb`: alpha-balanced focal-loss controls.
- `phase4_backtranslation_controls.ipynb`: augmentation controls.
- `inference.ipynb`: inference demo for testing new English sentences.
- `data/`: processed data splits used in the experiments.
- `results/`: saved metrics, reports and statistical verification results.
- `reproducibility/`: runtime and reproducibility metadata.

## Running the project

Install the required packages:

bash
pip install -r requirements.txt