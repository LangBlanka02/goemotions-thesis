from pathlib import Path
import random
import numpy as np

PROJECT_ROOT = Path("/content/drive/MyDrive/Thesis")
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"
for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
ID_TO_LABEL = {i: l for i, l in enumerate(GOEMOTIONS_LABELS)}
LABEL_TO_ID = {l: i for i, l in enumerate(GOEMOTIONS_LABELS)}

EKMAN_MAP = {
    "admiration":     "joy",
    "amusement":      "joy",
    "approval":       "joy",
    "caring":         "joy",
    "desire":         "joy",
    "excitement":     "joy",
    "gratitude":      "joy",
    "joy":            "joy",
    "love":           "joy",
    "optimism":       "joy",
    "pride":          "joy",
    "relief":         "joy",
    "anger":          "anger",
    "annoyance":      "anger",
    "disapproval":    "anger",
    "disgust":        "disgust",
    "fear":           "fear",
    "nervousness":    "fear",
    "sadness":        "sadness",
    "disappointment": "sadness",
    "embarrassment":  "sadness",
    "grief":          "sadness",
    "remorse":        "sadness",
    "surprise":       "surprise",
    "confusion":      "surprise",
    "curiosity":      "surprise",
    "realization":    "surprise",
    "neutral":        "neutral",
}
EKMAN_LABELS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EKMAN_LABEL_TO_ID = {l: i for i, l in enumerate(EKMAN_LABELS)}

def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass