from google.colab import drive
drive.mount("/content/drive")

import sys
import importlib

TARGET_PATH = (
    "/content/drive/MyDrive/Thesis/models"
)

importlib.invalidate_caches()

if TARGET_PATH not in sys.path:
    sys.path.insert(
        0,
        TARGET_PATH,
    )


from pathlib import Path
from datetime import datetime
import json
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from utils import (
    DATA_DIR,
    EKMAN_LABELS,
    set_all_seeds,
)

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 32
N_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

NUM_LABELS = len(
    EKMAN_LABELS
)

EXPECTED_TRAIN_SIZE = 36_302
EXPECTED_VALIDATION_SIZE = 4_547
EXPECTED_TEST_SIZE = 4_590

class GoEmotionsDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=MAX_LENGTH,
    ):
        self.texts = list(texts)
        self.labels = [
            int(label)
            for label in labels
        ]

        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(self.texts) != len(
            self.labels
        ):
            raise ValueError(
                "The number of texts and labels does not match."
            )

    def __len__(self):
        return len(
            self.labels
        )

    def __getitem__(
        self,
        index,
    ):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        encoded["labels"] = (
            self.labels[index]
        )

        return encoded

def validate_dataframe(
    dataframe,
    split_name,
    expected_size,
):
    required_columns = {
        "id",
        "text_clean",
        "ekman_id",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(dataframe) != expected_size:
        raise ValueError(
            f"Expected {expected_size:,} rows in "
            f"{split_name}, but found "
            f"{len(dataframe):,}."
        )

    if dataframe["id"].duplicated().any():
        raise ValueError(
            f"{split_name} contains duplicate IDs."
        )

    if dataframe["text_clean"].isna().any():
        raise ValueError(
            f"{split_name} contains null cleaned text."
        )

    empty_text = (
        dataframe["text_clean"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_text.any():
        raise ValueError(
            f"{split_name} contains "
            f"{int(empty_text.sum())} empty texts."
        )

    labels = (
        dataframe["ekman_id"]
        .astype(int)
    )

    invalid_labels = dataframe[
        ~labels.isin(
            range(NUM_LABELS)
        )
    ]

    if len(invalid_labels) > 0:
        raise ValueError(
            f"{split_name} contains invalid Ekman IDs."
        )

    observed_labels = set(
        labels.unique()
    )

    expected_labels = set(
        range(NUM_LABELS)
    )

    if observed_labels != expected_labels:
        raise ValueError(
            f"{split_name} does not contain all seven "
            f"Ekman classes. Observed: "
            f"{sorted(observed_labels)}"
        )


def load_data():
    train_dataframe = pd.read_parquet(
        DATA_DIR
        / "train_single_light.parquet"
    )

    validation_dataframe = pd.read_parquet(
        DATA_DIR
        / "validation_single_light.parquet"
    )

    test_dataframe = pd.read_parquet(
        DATA_DIR
        / "test_single_light.parquet"
    )

    validate_dataframe(
        train_dataframe,
        "training",
        EXPECTED_TRAIN_SIZE,
    )

    validate_dataframe(
        validation_dataframe,
        "validation",
        EXPECTED_VALIDATION_SIZE,
    )

    validate_dataframe(
        test_dataframe,
        "test",
        EXPECTED_TEST_SIZE,
    )

    train_ids = set(
        train_dataframe["id"]
    )

    validation_ids = set(
        validation_dataframe["id"]
    )

    test_ids = set(
        test_dataframe["id"]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Training and validation IDs overlap."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Training and test IDs overlap."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test IDs overlap."
        )

    return (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )

def hf_compute_metrics(
    evaluation_prediction,
):
    logits, true_labels = (
        evaluation_prediction
    )

    predicted_labels = np.argmax(
        logits,
        axis=-1,
    )

    label_ids = list(
        range(NUM_LABELS)
    )

    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=label_ids,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=label_ids,
            average="weighted",
            zero_division=0,
        ),
    }


def calculate_complete_metrics(
    true_labels,
    predicted_labels,
):
    label_ids = list(
        range(NUM_LABELS)
    )

    per_class_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=label_ids,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=label_ids,
            average="weighted",
            zero_division=0,
        ),
        "per_class_f1":
            per_class_f1.tolist(),
        "report": classification_report(
            true_labels,
            predicted_labels,
            labels=label_ids,
            target_names=EKMAN_LABELS,
            zero_division=0,
            output_dict=True,
        ),
    }

def make_json_safe(
    value,
):
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def save_confusion_matrix(
    true_labels,
    predicted_labels,
    output_path,
    title,
):
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(
            range(NUM_LABELS)
        ),
    )

    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(
            NUM_LABELS
        ),
        yticks=np.arange(
            NUM_LABELS
        ),
        xticklabels=EKMAN_LABELS,
        yticklabels=EKMAN_LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
    )

    threshold = (
        matrix.max() / 2
        if matrix.size
        else 0
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return matrix


def capture_environment():
    environment = {
        "python_version":
            sys.version,
        "platform":
            platform.platform(),
        "torch_version":
            torch.__version__,
        "transformers_version":
            transformers.__version__,
        "sklearn_version":
            sklearn.__version__,
        "pandas_version":
            pd.__version__,
        "numpy_version":
            np.__version__,
        "cuda_available":
            torch.cuda.is_available(),
        "torch_cuda_version":
            torch.version.cuda,
    }

    if torch.cuda.is_available():
        environment["gpu_name"] = (
            torch.cuda.get_device_name(0)
        )

        environment["cudnn_version"] = (
            torch.backends.cudnn.version()
        )

    return environment

def run_phase3(
    seed,
    output_root,
    require_gpu=True,
):
    if require_gpu and not (
        torch.cuda.is_available()
    ):
        raise RuntimeError(
            "A CUDA GPU is not active. In Colab, select "
            "Runtime → Change runtime type → T4 GPU."
        )

    set_all_seeds(
        seed
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_name = (
        f"phase3_distilbert_vanilla_"
        f"seed{seed}_{timestamp}"
    )

    output_root = Path(
        output_root
    )

    run_directory = (
        output_root
        / run_name
    )

    if run_directory.exists():
        raise FileExistsError(
            f"The output directory already exists:\n"
            f"{run_directory}"
        )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoint_directory = (
        run_directory
        / "checkpoints"
    )

    best_model_directory = (
        run_directory
        / "best_model"
    )

    print("=" * 70)
    print("PHASE 3: VANILLA DISTILBERT")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Output: {run_directory}")
    print(f"Environment: {capture_environment()}")
    print("=" * 70)

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = load_data()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label={
                index: label
                for index, label
                in enumerate(
                    EKMAN_LABELS
                )
            },
            label2id={
                label: index
                for index, label
                in enumerate(
                    EKMAN_LABELS
                )
            },
        )
    )

    train_dataset = GoEmotionsDataset(
        train_dataframe["text_clean"],
        train_dataframe["ekman_id"],
        tokenizer,
    )

    validation_dataset = (
        GoEmotionsDataset(
            validation_dataframe[
                "text_clean"
            ],
            validation_dataframe[
                "ekman_id"
            ],
            tokenizer,
        )
    )

    test_dataset = GoEmotionsDataset(
        test_dataframe["text_clean"],
        test_dataframe["ekman_id"],
        tokenizer,
    )

    training_arguments = (
        TrainingArguments(
            output_dir=str(
                checkpoint_directory
            ),
            learning_rate=(
                LEARNING_RATE
            ),
            weight_decay=(
                WEIGHT_DECAY
            ),
            per_device_train_batch_size=(
                BATCH_SIZE
            ),
            per_device_eval_batch_size=(
                BATCH_SIZE
            ),
            num_train_epochs=(
                N_EPOCHS
            ),
            warmup_steps=(
                WARMUP_STEPS
            ),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=(
                "macro_f1"
            ),
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=100,
            seed=seed,
            data_seed=seed,
            report_to="none",
        )
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=(
            validation_dataset
        ),
        processing_class=tokenizer,
        data_collator=(
            DataCollatorWithPadding(
                tokenizer=tokenizer
            )
        ),
        compute_metrics=(
            hf_compute_metrics
        ),
    )

    print("\nStarting training...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_start = time.time()

    training_result = trainer.train()

    training_seconds = (
        time.time()
        - training_start
    )

    peak_gpu_memory_gb = None

    if torch.cuda.is_available():
        peak_gpu_memory_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1e9
        )

    # Because load_best_model_at_end=True, trainer.model now
    # contains the validation-selected checkpoint.
    trainer.save_model(
        str(best_model_directory)
    )

    tokenizer.save_pretrained(
        str(best_model_directory)
    )

    def evaluate_split(
        dataset,
        dataframe,
        split_name,
    ):
        prediction_output = (
            trainer.predict(
                dataset,
                metric_key_prefix=(
                    split_name
                ),
            )
        )

        logits = (
            prediction_output.predictions
        )

        if isinstance(
            logits,
            tuple,
        ):
            logits = logits[0]

        true_labels = (
            prediction_output
            .label_ids
            .astype(int)
        )

        predicted_labels = (
            np.argmax(
                logits,
                axis=-1,
            )
            .astype(int)
        )

        metrics = (
            calculate_complete_metrics(
                true_labels,
                predicted_labels,
            )
        )

        prediction_dataframe = (
            pd.DataFrame({
                "id":
                    dataframe["id"].tolist(),
                "text_clean":
                    dataframe[
                        "text_clean"
                    ].astype(str).tolist(),
                "true_id":
                    true_labels,
                "true_label": [
                    EKMAN_LABELS[value]
                    for value
                    in true_labels
                ],
                "predicted_id":
                    predicted_labels,
                "predicted_label": [
                    EKMAN_LABELS[value]
                    for value
                    in predicted_labels
                ],
                "correct": (
                    true_labels
                    == predicted_labels
                ),
            })
        )

        prediction_dataframe.to_csv(
            run_directory
            / f"{split_name}_predictions.csv",
            index=False,
        )

        matrix = save_confusion_matrix(
            true_labels,
            predicted_labels,
            run_directory
            / (
                f"{split_name}_"
                f"confusion_matrix.png"
            ),
            (
                f"Phase 3 {split_name} "
                f"confusion matrix"
            ),
        )

        np.savetxt(
            run_directory
            / (
                f"{split_name}_"
                f"confusion_matrix.csv"
            ),
            matrix,
            delimiter=",",
            fmt="%d",
        )

        with open(
            run_directory
            / f"{split_name}_metrics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                make_json_safe(
                    metrics
                ),
                file,
                indent=2,
                ensure_ascii=False,
            )

        return metrics

    validation_metrics = (
        evaluate_split(
            validation_dataset,
            validation_dataframe,
            "validation",
        )
    )

    # Test evaluation occurs only after validation-based
    # checkpoint selection.
    test_metrics = evaluate_split(
        test_dataset,
        test_dataframe,
        "test",
    )

    environment = capture_environment()

    complete_results = {
        "run_name": run_name,
        "seed": seed,
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "number_of_epochs":
            N_EPOCHS,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "warmup_steps":
            WARMUP_STEPS,
        "training_size":
            len(train_dataframe),
        "validation_size":
            len(validation_dataframe),
        "test_size":
            len(test_dataframe),
        "best_checkpoint":
            trainer.state
            .best_model_checkpoint,
        "best_validation_macro_f1":
            trainer.state.best_metric,
        "training_time_seconds":
            training_seconds,
        "peak_gpu_memory_gb":
            peak_gpu_memory_gb,
        "environment":
            environment,
        "validation":
            validation_metrics,
        "test":
            test_metrics,
    }

    with open(
        run_directory
        / "complete_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(
                complete_results
            ),
            file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        run_directory
        / "training_log_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(
                trainer.state.log_history
            ),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("FINAL PHASE 3 RESULTS")
    print("=" * 70)

    print("\nValidation metrics:")
    print(
        json.dumps(
            make_json_safe(
                validation_metrics
            ),
            indent=2,
        )
    )

    print("\nTest metrics:")
    print(
        json.dumps(
            make_json_safe(
                test_metrics
            ),
            indent=2,
        )
    )

    print("\nBest checkpoint:")
    print(
        trainer.state
        .best_model_checkpoint
    )

    print("\nAll results saved in:")
    print(
        run_directory
    )

    print("=" * 70)

    return complete_results


