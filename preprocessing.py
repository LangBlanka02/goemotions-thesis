import sys
sys.path.append('/content/drive/MyDrive/Thesis/models')

import re
from typing import Iterable
import pandas as pd
from datasets import load_dataset

from utils import DATA_DIR, ID_TO_LABEL, EKMAN_MAP, EKMAN_LABEL_TO_ID

import nltk
for resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text_heavy(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\[name\]", " ", text)
    text = re.sub(r"/?r/\w+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def clean_text_light(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\[NAME\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def load_single_label_split(split_name: str) -> pd.DataFrame:
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split_name)
    df = ds.to_pandas()
    df = df[df["labels"].apply(len) == 1].copy()
    df["label"]      = df["labels"].apply(lambda x: x[0])
    df["label_name"] = df["label"].apply(ID_TO_LABEL.get)
    df["ekman"]      = df["label_name"].apply(EKMAN_MAP.get)
    df["ekman_id"]   = df["ekman"].apply(EKMAN_LABEL_TO_ID.get)
    return df.reset_index(drop=True)

def build_classical_partitions() -> None:
    for split in ("train", "validation", "test"):
        df = load_single_label_split(split)
        df["text_clean"] = df["text"].apply(clean_text_heavy)
        df = df[df["text_clean"].str.len() > 0]
        out = DATA_DIR / f"{split}_single_clean.parquet"
        df.to_parquet(out, index=False)
        print(f"Saved {len(df):>6} rows -> {out}")

def build_transformer_partitions() -> None:
    for split in ("train", "validation", "test"):
        df = load_single_label_split(split)
        df["text_clean"] = df["text"].apply(clean_text_light)
        df = df[df["text_clean"].str.len() > 0]
        out = DATA_DIR / f"{split}_single_light.parquet"
        df.to_parquet(out, index=False)
        print(f"Saved {len(df):>6} rows -> {out}")