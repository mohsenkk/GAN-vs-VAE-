#!/usr/bin/env python3
"""Stage 09A: one full 128/4/8 PDBbind Variant C Kaggle training run.

Attach this file and the DCGAN-DTA repository as Kaggle inputs, then run this
file as a script. No baseline source or input data is edited. A parent process
monitors one worker and writes evidence under /kaggle/working/stage09a_full/.
Training and selection use shipped training fold 0 for validation. The held-out
test fold is opened only after best-checkpoint restoration, for one final pass.
"""

import hashlib
import csv
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback


INPUT_ROOT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/stage09a_full")
WORK_SOURCE = OUTPUT / "source"
WALL_LIMIT_SECONDS = 5 * 60 * 60
HOST_RSS_LIMIT_BYTES = 12 * 1024**3
GPU_HEADROOM_MIB = 512
POLL_SECONDS = 1.0
MAX_EPOCHS = 300
PATIENCE = 75
PAPER_REFERENCE = {"ci": 0.768, "aupr": 0.787, "rm2": 0.451, "mse": 1.907}
MILESTONES = {1, 10, 25, 50, 75, 100, 150, 200, 250, 300}

# Fingerprints of the inspected local baseline and shipped PDBbind inputs.
# Fail before launching the worker if a different Kaggle input was attached.
EXPECTED_CODE_SHA256 = {
    "run_experiments.py": "c05b14bd62fd0fb90419868e64e50bfbd9628401013e817001058eccf28a7058",
    "arguments.py": "ce144580cf422ce7ec2d55d3f30b7afd450f6797933b5f36f304f6f0ce18e738",
    "datahelper.py": "9b5dd382c7b84c7ae18fee736e1dcfbab127c69a6cb9cf003639ef9fa6da8d07",
    "dataset.py": "6b50c930f5a4030905d7ecea77e84405255760d3b7a0d4b82808d696b4ffffdd",
    "emetrics.py": "6208f8b7bb0a18cff725ac2561e3f82f32b3ff9d46359e5989aa6fa08bc01ae8",
}
EXPECTED_INPUT_SHA256 = {
    "Y": "53f8f201c70d7bffd2c72238949f387e6483086bd6a12de5825facc6e2aa5660",
    "ligands.txt": "a1bc45f4e8479ea29c4b6bce4476de13aa2013ee5e2aae824e0bbb324dbffa1e",
    "proteins.txt": "3ef813e2a6d3dbfcbd1fcf040cbbda26dcddff1e128cecca385d180618776be1",
    "ligands_train.txt": "9ed95a24a48483da10482f9d002a963a260e29c138bfda15f8e0d3c8d2dd69b6",
    "proteins_train.txt": "71584263eb205f4366af1c54a7b01c69ed2aa1acccde5267d783b0853fec4c1a",
    "protein_feature_vec.json": "af8d5e2c2aa00ab605948cc2440f8be508266f37fef46b7e8c32b7ce9b87d044",
    "protein_feature_vecblsm.json": "69c09580f891c498a1ae82d4895a54a00688d8818980c494f49ca495c1189adc",
    "folds/train_fold_setting1.txt": "c50d2eec98eae8958218124cdf71bf0c48aa2cafc7831685392a2714c3c07c47",
}
EXPECTED_TEST_FOLD_SHA256 = "224462551b2edf67413d0bd7a9558adb24e4a13606a368bebf17ed965325eec3"

# Kaggle's verified compatibility configuration must precede TensorFlow import.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(kind, **data):
    event = {"kind": kind, "epoch_seconds": time.time(), **data}
    with (OUTPUT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")


def read_events():
    path = OUTPUT / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A worker killed during a write can leave an incomplete last line.
    return events


def recover_progress(summary, events):
    """Keep milestone evidence even when a safety kill prevents worker cleanup."""
    for event in events:
        kind = event["kind"]
        if kind == "environment_ready":
            summary["environment"] = event["environment"]
        elif kind == "data_ready":
            summary["dataset"] = event["dataset"]
            summary["fold_identity"] = event["fold_identity"]
            summary["fold_sizes"] = event["fold_sizes"]
        elif kind == "memory_boundary":
            summary.setdefault("memory_at_boundaries", {})[event["label"]] = event["sample"]
        elif kind == "gan_started":
            summary.setdefault("gan", {})["calls"] = 1
        elif kind == "gan_completed":
            summary.setdefault("gan", {}).update({
                "completed": True, "runtime_seconds": event["runtime_seconds"]
            })
        elif kind == "variant_c_constructed":
            summary.setdefault("dta", {}).update({
                "constructed": True, "parameters": event["parameters"]
            })
        elif kind == "dta_epoch_completed":
            summary.setdefault("dta", {}).update({
                "epochs_completed": event["epoch"],
                "optimizer_updates": event["optimizer_updates"],
                "best_epoch": event["best_epoch"],
                "best_val_native_ci": event["best_val_native_ci"],
            })
        elif kind == "best_validation_completed":
            summary["validation"] = event["validation"]
        elif kind == "test_fold_opened":
            summary["test_set_used"] = True
            summary.setdefault("fold_identity", {})["heldout_test_fold_file_read"] = True
        elif kind == "test_evaluation_started":
            summary["test_evaluation_count"] = 1
        elif kind == "test_evaluation_completed":
            summary["test_prediction_count"] = event["prediction_count"]
            summary["test"] = {"metrics": event["metrics"]}
        elif kind == "worker_exception":
            summary["exception"] = event["exception"]
    return summary


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def discover_source():
    matches = [
        candidate.parent
        for candidate in INPUT_ROOT.rglob("run_experiments.py")
        if (candidate.parent / "dataset.py").is_file()
        and (candidate.parent / "data/pdb/Y").is_file()
        and (candidate.parent / "data/pdb/folds/train_fold_setting1.txt").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one DCGAN-DTA source under {INPUT_ROOT}; found {matches}")
    return matches[0]


def prepare_source(source):
    WORK_SOURCE.mkdir(parents=True, exist_ok=False)
    files = ("run_experiments.py", "arguments.py", "datahelper.py", "dataset.py", "emetrics.py")
    fingerprints = {}
    for name in files:
        original = source / name
        if not original.is_file():
            raise FileNotFoundError(original)
        fingerprints[name] = sha256(original)
        shutil.copy2(original, WORK_SOURCE / name)
        if sha256(WORK_SOURCE / name) != fingerprints[name]:
            raise RuntimeError(f"Working copy differs from Kaggle input: {name}")
    # Data remain the read-only Kaggle input. The original relative data/pdb path works.
    (WORK_SOURCE / "data").symlink_to(source / "data", target_is_directory=True)
    for name in ("figures", "logs", "result"):
        (WORK_SOURCE / name).mkdir(exist_ok=True)
    return fingerprints


def finite_metrics(values):
    import numpy as np

    result = {}
    for key, value in (values or {}).items():
        number = float(np.asarray(value).reshape(()))
        if not np.isfinite(number):
            raise FloatingPointError(f"Non-finite native {key}: {number}")
        result[key] = number
    return result


def classify_failure(phase, error):
    detail = (type(error).__name__ + " " + str(error)).lower()
    if "out of memory" in detail or "resourceexhausted" in detail or "resource limit" in detail:
        return "resource failure"
    if "monitor unavailable" in detail or "nvidia-smi" in detail:
        return "harness failure"
    return {
        "environment": "compatibility failure",
        "data": "harness failure",
        "gan_training": "model construction failure",
        "model_construction": "model construction failure",
        "dta_training": "training failure",
        "final_validation": "validation failure",
        "test_evaluation": "test evaluation failure",
    }.get(phase, "harness failure")


def sample_boundary(label, result, enforce=True):
    """Take a worker-local boundary reading in addition to parent polling."""
    rss = process_rss_bytes(os.getpid())
    gpu = gpu_memory()
    if rss is None:
        raise RuntimeError("Worker RSS monitor unavailable at " + label)
    sample = {
        "epoch_seconds": time.time(),
        "worker_rss_bytes": rss,
        "gpu_used_mib": gpu["used_mib"],
        "gpu_total_mib": gpu["total_mib"],
    }
    result.setdefault("memory_at_boundaries", {})[label] = sample
    emit("memory_boundary", label=label, sample=sample)
    if enforce and (rss > HOST_RSS_LIMIT_BYTES or
                    gpu["used_mib"] >= gpu["total_mib"] - GPU_HEADROOM_MIB):
        raise RuntimeError("Resource limit crossed at " + label)
    return sample


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_weights(model, path):
    temporary = path.with_name(path.stem + ".tmp.weights.h5")
    model.save_weights(str(temporary))
    os.replace(temporary, path)


def save_array(path, values):
    import numpy as np

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(values))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def full_vector_metrics(native, targets, predictions):
    import numpy as np
    from emetrics import get_cindex

    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    p = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise FloatingPointError("Non-finite or mismatched full-vector predictions")
    observed = {
        "global_ci": float(get_cindex(y, p)),
        "mse": float(np.mean((y - p) ** 2)),
        "aupr": float(native.average_precision_score(y > 7, p)),
        "rm2": float(np.asarray(native.get_rm2(y, p.reshape(-1, 1))).reshape(())),
        "mae_diagnostic": float(np.mean(np.abs(y - p))),
        "prediction_count": int(len(p)),
    }
    finite_metrics(observed)
    return observed


def paper_gaps(metrics):
    return {
        "ci_gap_to_paper": PAPER_REFERENCE["ci"] - metrics["global_ci"],
        "aupr_gap_to_paper": PAPER_REFERENCE["aupr"] - metrics["aupr"],
        "rm2_gap_to_paper": PAPER_REFERENCE["rm2"] - metrics["rm2"],
        "mse_gap_to_paper": metrics["mse"] - PAPER_REFERENCE["mse"],
    }


def reconstructed_native_ci(native, targets, predictions, batch_size):
    """One prediction pass; apply the source metric to each native batch.

    Keras MeanMetricWrapper weighs the batch metric by batch sample count.
    This is a post-hoc reconstruction, not a second model evaluation.
    """
    import numpy as np
    import tensorflow as tf

    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    p = np.asarray(predictions, dtype=np.float32).reshape(-1, 1)
    total = 0.0
    for start in range(0, len(y), batch_size):
        stop = min(start + batch_size, len(y))
        score = float(native.cindex_score(tf.constant(y[start:stop]),
                                          tf.constant(p[start:stop])).numpy())
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite reconstructed native CI")
        total += score * (stop - start)
    return total / len(y)


def full_worker():
    started = time.monotonic()
    phase = "environment"
    result = {
        "status": "running", "execution_status": "running",
        "convergence_status": None,
        "numerical_comparison_status": "descriptive_only_pending_protocol_alignment",
        "resource_status": "within_limits", "stopping_reason": None,
        "failure_classification": None, "exception": None,
        "test_set_used": False, "test_evaluation_count": 0,
        "test_prediction_count": 0, "environment": {}, "dataset": {},
        "fold_identity": {}, "fold_sizes": {}, "configuration": {},
        "gan": {"calls": 0, "completed": False, "native_schedule":
                {"iterations": 5000, "batch_size": 5, "log_interval": 500}},
        "dta": {"constructed": False, "epochs_completed": 0,
                "optimizer_updates": 0, "best_epoch": None,
                "best_val_native_ci": None},
        "validation": {}, "test": {}, "finite": True,
        "paper_reference": PAPER_REFERENCE,
        "seeds": {"native_numpy": 1, "native_python_random": 1,
                  "native_tensorflow_v1": 0,
                  "deterministic_gpu_execution_claimed": False},
        "paper_comparison_caveat": (
            "Contextual only: paper split aggregation, metric estimator, seed, "
            "and model-selection protocol are under-specified."),
        "resume": {"exact_resume_supported": False,
                   "reason": "weights alone omit optimizer, callback and RNG states; restart fresh"},
        "timings_seconds": {},
        "controlled_deviations": [
            "fixed paper-reported 128/4/8 configuration; no README grid search",
            "shipped training fold 0 used for validation; other four for training",
            "native validation-CI EarlyStopping, patience 75, maximum 300 epochs",
            "read-only Kaggle input accessed through a writable source copy",
            "standalone keras imports mapped to verified tf_keras 2.20 runtime",
            "unused native np.mat(Y copy) allocation omitted (NumPy 2 removed np.mat)",
            "additional full-vector validation diagnostics do not affect model selection",
            "one held-out test prediction pass after best checkpoint restoration",
        ],
    }

    live = {
        "phase": phase, "current_epoch": 0, "max_epochs": MAX_EPOCHS,
        "best_epoch": None, "best_val_native_ci_so_far": None,
        "epochs_since_improvement": 0, "patience": PATIENCE,
        "patience_remaining": PATIENCE, "latest_train_loss": None,
        "latest_train_ci": None, "latest_val_loss": None,
        "latest_val_native_ci": None, "latest_val_global_ci": None,
        "latest_val_mse": None, "latest_val_aupr": None,
        "latest_val_rm2": None, "latest_diagnostic_epoch": None,
        "paper_reference": PAPER_REFERENCE, "metric_gaps": None,
        "elapsed_seconds": 0.0, "estimated_remaining_seconds": None,
        "worker_rss_bytes": None, "gpu_used_mib": None,
        "gpu_total_mib": None, "warnings": [], "test_evaluation_count": 0,
    }

    def publish(new_phase=None, **changes):
        nonlocal phase
        if new_phase is not None:
            phase = new_phase
            live["phase"] = phase
        live.update(changes)
        live["elapsed_seconds"] = time.monotonic() - started
        live["last_update_timestamp"] = time.time()
        write_json(OUTPUT / "live_status.json", live)

    try:
        publish()
        import numpy as np
        import tensorflow as tf

        physical_gpus = tf.config.list_physical_devices("GPU")
        if len(physical_gpus) != 1:
            raise RuntimeError(f"Expected exactly one visible T4; TensorFlow sees {physical_gpus}")
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)
        import tf_keras

        if sys.version_info[:2] != (3, 12):
            raise RuntimeError(f"Kaggle Python differs from verified 3.12: {sys.version}")
        if not tf.__version__.startswith("2.20.") or not tf_keras.__version__.startswith("2.20."):
            raise RuntimeError("TensorFlow/tf_keras differs from verified 2.20")
        if np.__version__ != "2.0.2":
            raise RuntimeError(f"NumPy differs from verified 2.0.2: {np.__version__}")
        for suffix in ("backend", "layers", "models", "callbacks", "utils"):
            sys.modules["keras." + suffix] = importlib.import_module("tf_keras." + suffix)
        sys.modules["keras"] = tf_keras
        backend = tf.compat.v1.keras.backend
        if not all(callable(getattr(backend, name, None))
                   for name in ("get_session", "set_session")):
            raise RuntimeError("Legacy Keras session API unavailable")
        if not all(callable(getattr(tf_keras.Model, name, None))
                   for name in ("fit_generator", "predict_generator")):
            raise RuntimeError("Legacy generator API unavailable")
        result["environment"] = {
            "python": sys.version, "tensorflow": tf.__version__,
            "tf_keras": tf_keras.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_use_legacy_keras": os.environ["TF_USE_LEGACY_KERAS"],
            "gpu_details": tf.config.experimental.get_device_details(physical_gpus[0]),
            "memory_growth": tf.config.experimental.get_memory_growth(physical_gpus[0]),
        }
        emit("environment_ready", environment=result["environment"])
        os.chdir(WORK_SOURCE)
        sys.path.insert(0, str(WORK_SOURCE))
        sys.argv = [
            "run_experiments.py", "--dataset_path", "data/pdb/",
            "--problem_type", "1", "--is_log", "0", "--model", "C",
            "--num_windows", "128", "--smi_window_lengths", "4",
            "--seq_window_lengths", "8", "--batch_size", "256",
            "--num_epoch", str(MAX_EPOCHS), "--max_smi_len", "200",
            "--max_seq_len", "2000", "--learning_rate", "0.001",
            "--log_dir", str(OUTPUT),
        ]
        import run_experiments as native
        from arguments import argparser
        from datahelper import DataSet
        import pandas
        import sklearn
        import matplotlib

        result["environment"].update({
            "pandas": pandas.__version__, "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        })
        flags = argparser()
        if not (flags.model == "C" and flags.num_windows == [128]
                and flags.smi_window_lengths == [4]
                and flags.seq_window_lengths == [8]
                and flags.problem_type == 1 and flags.is_log == 0
                and flags.batch_size == 256 and flags.num_epoch == MAX_EPOCHS
                and flags.max_smi_len == 200 and flags.max_seq_len == 2000
                and flags.learning_rate == 0.001):
            raise RuntimeError("Parsed configuration differs from Stage 09A full protocol")
        result["configuration"] = {
            "dataset_path": flags.dataset_path, "problem_type": flags.problem_type,
            "is_log": flags.is_log, "model": flags.model,
            "num_filters": 128, "drug_kernel": 4, "protein_kernel": 8,
            "batch_size": flags.batch_size, "max_epochs": MAX_EPOCHS,
            "early_stopping_monitor": "val_cindex_score",
            "early_stopping_mode": "max", "early_stopping_patience": PATIENCE,
            "requested_learning_rate": flags.learning_rate,
            "native_compile_optimizer": "adam default (source ignores learning_rate flag)",
            "max_smi_len": flags.max_smi_len, "max_seq_len": flags.max_seq_len,
        }

        publish("data")
        load_start = time.monotonic()
        dataset = DataSet(flags.dataset_path, flags.problem_type,
                          flags.max_seq_len, flags.max_smi_len, need_shuffle=False)
        flags.charseqset_size = dataset.charseqset_size
        flags.charsmiset_size = dataset.charsmiset_size
        XD, XT, Y, XD_t, XT_t = dataset.parse_data(flags)
        XD, XT, Y, XD_t = map(np.asarray, (XD, XT, Y, XD_t))
        del XT_t
        rows, cols = np.where(np.isnan(Y) == False)
        # Loading the shipped test-fold indices is deliberately deferred.
        with Path(flags.dataset_path, "folds/train_fold_setting1.txt").open(
                encoding="utf-8") as stream:
            folds = json.load(stream)
        if len(folds) != 5:
            raise RuntimeError("Expected five shipped training folds")
        valinds = folds[0]
        traininds = [index for fold in folds[1:] for index in fold]
        pool = [index for fold in folds for index in fold]
        if (len(pool) != len(set(pool)) or
                any(index < 0 or index >= len(rows) for index in pool)):
            raise RuntimeError("Training folds overlap or have invalid indices")
        heldout_inferred = len(rows) - len(pool)
        if (len(rows), len(traininds), len(valinds), heldout_inferred) != (
                5014, 3344, 836, 834):
            raise RuntimeError("Shipped fold/data sizes differ from Stage 08C")
        if list(XD_t.shape) != [50068, 200]:
            raise RuntimeError("GAN drug corpus differs from Stage 08B/08C")
        result["dataset"] = {
            "name": "PDBbind", "XD_shape": list(XD.shape),
            "XT_shape": list(XT.shape), "Y_shape": list(Y.shape),
            "XD_t_shape": list(XD_t.shape), "observed_pairs": len(rows),
            "native_parse_data_loads_full_affinity_matrix": True,
        }
        result["fold_identity"] = {
            "validation_fold_zero_based": 0,
            "training_folds_zero_based": [1, 2, 3, 4],
            "heldout_test_fold_file_read": False,
            "heldout_pair_count_inferred_from_remainder": heldout_inferred,
        }
        result["fold_sizes"] = {
            "train": len(traininds), "validation": len(valinds),
            "heldout_inferred_not_loaded": heldout_inferred,
        }
        trrows, trcols = rows[traininds], cols[traininds]
        varows, vacols = rows[valinds], cols[valinds]
        train_drugs, train_prots, train_Y = native.prepare_interaction_pairs(
            XD, XT, Y, trrows, trcols)
        val_drugs, val_prots, val_Y = native.prepare_interaction_pairs(
            XD, XT, Y, varows, vacols)
        with Path(flags.dataset_path, "protein_feature_vecblsm.json").open(
                encoding="utf-8") as stream:
            pro2vec = json.load(stream)
        index2char = {index: char for char, index in native.CHARPROTSET.items()}
        train_gen = native.DataGenerator(
            train_drugs, train_prots, train_Y, pro2vec, flags.batch_size,
            flags.max_seq_len, index2char, shuffle=False)
        val_gen = native.DataGenerator(
            val_drugs, val_prots, val_Y, pro2vec, flags.batch_size,
            flags.max_seq_len, index2char, shuffle=False)
        _, first_labels = train_gen[0]
        if len(first_labels) != 256 or len(train_gen) != 14 or len(val_gen) != 4:
            raise RuntimeError("Native generators differ from preflight")
        result["fold_sizes"].update({
            "train_batches": len(train_gen), "validation_batches": len(val_gen),
        })
        result["timings_seconds"]["dataset_loading"] = time.monotonic() - load_start
        emit("data_ready", dataset=result["dataset"],
             fold_identity=result["fold_identity"], fold_sizes=result["fold_sizes"])
        sample_boundary("before_gan", result)

        publish("gan_training")
        construct_start = time.monotonic()
        gan_start = None
        gan_code = native.ganForDrug.__code__

        def gan_profile(frame, event, ignored):
            nonlocal gan_start
            if frame.f_code is gan_code and event == "call":
                result["gan"]["calls"] += 1
                if result["gan"]["calls"] != 1:
                    raise RuntimeError("Unexpected second drug-GAN call")
                gan_start = time.monotonic()
                emit("gan_started")
            elif frame.f_code is gan_code and event == "return" and gan_start is not None:
                result["gan"]["runtime_seconds"] = time.monotonic() - gan_start
                result["gan"]["completed"] = True
                emit("gan_completed", runtime_seconds=result["gan"]["runtime_seconds"])
                publish("model_construction")
                sample_boundary("after_gan_before_dta_construction", result)

        sys.setprofile(gan_profile)
        try:
            model = native.build_GAN_C(
                flags, XD_t, flags.num_windows[0],
                flags.smi_window_lengths[0], flags.seq_window_lengths[0])
        finally:
            sys.setprofile(None)
        if result["gan"]["calls"] != 1 or not result["gan"]["completed"]:
            raise RuntimeError("Exactly one fresh native drug GAN did not complete")
        result["dta"].update({
            "constructed": True, "parameters": int(model.count_params()),
            "optimizer_updates": int(model.optimizer.iterations.numpy()),
        })
        if result["dta"]["parameters"] <= 0 or result["dta"]["optimizer_updates"] != 0:
            raise RuntimeError("DTA construction or initial optimizer state invalid")
        result["timings_seconds"]["variant_c_construction"] = (
            time.monotonic() - construct_start - result["gan"]["runtime_seconds"])
        emit("variant_c_constructed", parameters=result["dta"]["parameters"])
        sample_boundary("after_variant_c_construction", result)

        publish("dta_training")
        fit_start = time.monotonic()
        fields = [
            "epoch", "train_loss", "train_native_ci", "val_loss", "val_native_ci",
            "val_global_ci", "val_mse", "val_aupr", "val_rm2", "val_mae_diagnostic",
            "ci_gap_to_paper", "aupr_gap_to_paper", "rm2_gap_to_paper",
            "mse_gap_to_paper",
            "best_val_native_ci_so_far", "best_epoch_so_far", "epochs_since_improvement",
            "elapsed_seconds", "epoch_seconds", "estimated_remaining_seconds",
            "worker_rss_bytes", "gpu_used_mib", "gpu_total_mib",
            "train_loss_finite", "train_native_ci_finite", "val_loss_finite",
            "val_native_ci_finite", "val_full_metrics_finite", "diagnostic_epoch",
        ]
        epoch_records = []
        latest_full = None
        best_epoch = None
        best_ci = -math.inf
        epoch_start = None

        class Observe(tf_keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                nonlocal epoch_start
                epoch_start = time.monotonic()

            def on_train_batch_end(self, batch, logs=None):
                finite_metrics(logs)
                result["dta"]["optimizer_updates"] = int(
                    self.model.optimizer.iterations.numpy())

            def on_test_batch_end(self, batch, logs=None):
                finite_metrics(logs)

            def on_epoch_end(self, epoch, logs=None):
                nonlocal latest_full, best_epoch, best_ci
                number = int(epoch) + 1
                native_logs = finite_metrics(logs)
                ci = native_logs["val_cindex_score"]
                improved = ci > best_ci
                if improved:
                    best_ci, best_epoch = ci, number
                    atomic_weights(self.model, OUTPUT / "best_model.weights.h5")
                    emit("best_checkpoint", epoch=number, val_native_ci=ci)
                # Snapshot the actual post-epoch weights before EarlyStopping
                # can restore its selected state, regardless of Keras version.
                atomic_weights(self.model, OUTPUT / "latest_model.weights.h5")
                diagnostic = (number == 1 or number % 5 == 0 or improved
                              or number == MAX_EPOCHS)
                full = None
                if diagnostic:
                    predictions = self.model.predict_generator(val_gen, verbose=0)
                    full = full_vector_metrics(native, val_Y, predictions)
                    latest_full = {"epoch": number, **full, "gaps": paper_gaps(full)}
                sample = sample_boundary(f"epoch_{number}", result, enforce=False)
                epoch_seconds = time.monotonic() - epoch_start
                elapsed = time.monotonic() - started
                no_improvement = number - best_epoch
                mean_epoch = (time.monotonic() - fit_start) / number
                estimated = mean_epoch * (MAX_EPOCHS - number)
                row = {
                    "epoch": number, "train_loss": native_logs["loss"],
                    "train_native_ci": native_logs["cindex_score"],
                    "val_loss": native_logs["val_loss"], "val_native_ci": ci,
                    "val_global_ci": full["global_ci"] if full else None,
                    "val_mse": full["mse"] if full else None,
                    "val_aupr": full["aupr"] if full else None,
                    "val_rm2": full["rm2"] if full else None,
                    "val_mae_diagnostic": full["mae_diagnostic"] if full else None,
                    "ci_gap_to_paper": paper_gaps(full)["ci_gap_to_paper"] if full else None,
                    "aupr_gap_to_paper": paper_gaps(full)["aupr_gap_to_paper"] if full else None,
                    "rm2_gap_to_paper": paper_gaps(full)["rm2_gap_to_paper"] if full else None,
                    "mse_gap_to_paper": paper_gaps(full)["mse_gap_to_paper"] if full else None,
                    "best_val_native_ci_so_far": best_ci,
                    "best_epoch_so_far": best_epoch,
                    "epochs_since_improvement": no_improvement,
                    "elapsed_seconds": elapsed, "epoch_seconds": epoch_seconds,
                    "estimated_remaining_seconds": estimated,
                    **{key: sample[key] for key in (
                        "worker_rss_bytes", "gpu_used_mib", "gpu_total_mib")},
                    "train_loss_finite": True, "train_native_ci_finite": True,
                    "val_loss_finite": True, "val_native_ci_finite": True,
                    "val_full_metrics_finite": True if full else None,
                    "diagnostic_epoch": diagnostic,
                }
                warnings = []
                if row["train_loss"] > 100:
                    warnings.append("train loss above 100")
                if no_improvement >= 50:
                    warnings.append("validation CI plateau of at least 50 epochs")
                if row["train_native_ci"] - ci > 0.15:
                    warnings.append("train/validation native CI gap above 0.15")
                if sample["gpu_used_mib"] >= sample["gpu_total_mib"] - GPU_HEADROOM_MIB - 256:
                    warnings.append("GPU within 256 MiB of safety threshold")
                if sample["worker_rss_bytes"] >= HOST_RSS_LIMIT_BYTES - 1024**3:
                    warnings.append("RSS within 1 GiB of safety threshold")
                epoch_records.append(row)
                with (OUTPUT / "history.csv").open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writerow(row)
                    stream.flush()
                    os.fsync(stream.fileno())
                append_jsonl(OUTPUT / "history.jsonl", row)
                result["dta"].update({
                    "epochs_completed": number,
                    "optimizer_updates": int(self.model.optimizer.iterations.numpy()),
                    "best_epoch": best_epoch, "best_val_native_ci": best_ci,
                    "last_epoch_metrics": native_logs,
                })
                emit("dta_epoch_completed", epoch=number,
                     optimizer_updates=result["dta"]["optimizer_updates"],
                     best_epoch=best_epoch, best_val_native_ci=best_ci)
                if number in MILESTONES:
                    milestone = {
                        "epoch": number, "best_epoch": best_epoch,
                        "best_val_native_ci": best_ci,
                        "latest_full_validation": latest_full,
                        "metric_gaps_to_paper": latest_full["gaps"] if latest_full else None,
                        "resource": sample, "elapsed_seconds": elapsed,
                    }
                    append_jsonl(OUTPUT / "milestones.jsonl", milestone)
                    emit("milestone", epoch=number)
                publish(
                    current_epoch=number, best_epoch=best_epoch,
                    best_val_native_ci_so_far=best_ci,
                    epochs_since_improvement=no_improvement,
                    patience_remaining=max(0, PATIENCE - no_improvement),
                    latest_train_loss=row["train_loss"],
                    latest_train_ci=row["train_native_ci"],
                    latest_val_loss=row["val_loss"],
                    latest_val_native_ci=ci,
                    latest_val_global_ci=latest_full["global_ci"] if latest_full else None,
                    latest_val_mse=latest_full["mse"] if latest_full else None,
                    latest_val_aupr=latest_full["aupr"] if latest_full else None,
                    latest_val_rm2=latest_full["rm2"] if latest_full else None,
                    latest_diagnostic_epoch=latest_full["epoch"] if latest_full else None,
                    metric_gaps=latest_full["gaps"] if latest_full else None,
                    estimated_remaining_seconds=estimated,
                    worker_rss_bytes=sample["worker_rss_bytes"],
                    gpu_used_mib=sample["gpu_used_mib"],
                    gpu_total_mib=sample["gpu_total_mib"], warnings=warnings,
                )
                print(f"Epoch {number}/{MAX_EPOCHS}: val native CI={ci:.6f}; "
                      f"best={best_ci:.6f} at {best_epoch}; patience={no_improvement}/{PATIENCE}",
                      flush=True)
                if (sample["worker_rss_bytes"] > HOST_RSS_LIMIT_BYTES or
                        sample["gpu_used_mib"] >=
                        sample["gpu_total_mib"] - GPU_HEADROOM_MIB):
                    raise RuntimeError(f"Resource limit crossed at epoch {number}")

        with (OUTPUT / "history.csv").open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
            stream.flush()
            os.fsync(stream.fileno())
        for filename in ("history.jsonl", "milestones.jsonl"):
            (OUTPUT / filename).touch(exist_ok=False)
        es = native.EarlyStopping(
            monitor="val_cindex_score", mode="max", verbose=1,
            patience=PATIENCE, restore_best_weights=True)
        emit("dta_fit_started", planned_epochs=MAX_EPOCHS,
             planned_batches_per_epoch=len(train_gen))
        model.fit_generator(
            generator=train_gen, validation_data=val_gen,
            epochs=MAX_EPOCHS, callbacks=[Observe(), es])
        result["timings_seconds"]["fit_including_validation"] = time.monotonic() - fit_start
        completed = result["dta"]["epochs_completed"]
        if completed < 1 or completed > MAX_EPOCHS:
            raise RuntimeError("No complete DTA epoch or epoch count exceeded bound")
        if result["dta"]["optimizer_updates"] != completed * len(train_gen):
            raise RuntimeError("DTA optimizer-update count differs from full native epochs")
        latest_path = OUTPUT / "latest_model.weights.h5"
        if not latest_path.is_file():
            raise RuntimeError("Last completed epoch weights were not saved")
        os.replace(latest_path, OUTPUT / "final_model.weights.h5")
        result["convergence_status"] = (
            "early_stopped_after_plateau" if completed < MAX_EPOCHS else "max_epochs_reached")
        result["stopping_reason"] = result["convergence_status"]

        publish("final_validation", estimated_remaining_seconds=None)
        final_validation_start = time.monotonic()
        # The final-epoch state and selected best state are separately inspected.
        model.load_weights(str(OUTPUT / "final_model.weights.h5"))
        final_predictions = model.predict_generator(val_gen, verbose=0)
        final_metrics = full_vector_metrics(native, val_Y, final_predictions)
        model.load_weights(str(OUTPUT / "best_model.weights.h5"))
        best_predictions = model.predict_generator(val_gen, verbose=0)
        best_metrics = full_vector_metrics(native, val_Y, best_predictions)
        save_array(OUTPUT / "validation_predictions_best.npy", best_predictions)
        save_array(OUTPUT / "validation_targets.npy", val_Y)
        result["validation"] = {
            "final_epoch": completed, "final_epoch_metrics": final_metrics,
            "best_epoch": best_epoch, "best_val_native_ci": best_ci,
            "best_full_vector_metrics": best_metrics,
            "best_metric_gaps_to_paper": paper_gaps(best_metrics),
            "native_ci_selection_metric": "Keras val_cindex_score",
            "global_ci_metric": "repository emetrics.get_cindex on full vector",
            "best_checkpoint_restored": True,
        }
        publish(latest_val_global_ci=best_metrics["global_ci"],
                latest_val_mse=best_metrics["mse"],
                latest_val_aupr=best_metrics["aupr"],
                latest_val_rm2=best_metrics["rm2"],
                latest_diagnostic_epoch=best_epoch,
                metric_gaps=paper_gaps(best_metrics))
        emit("best_validation_completed", validation=result["validation"])
        sample_boundary("after_best_checkpoint_restoration", result)
        result["timings_seconds"]["final_validation"] = (
            time.monotonic() - final_validation_start)

        # First and only access to the shipped held-out fold is below this gate.
        if (result["dta"]["epochs_completed"] < 1
                or result["validation"]["best_checkpoint_restored"] is not True):
            raise RuntimeError("Test isolation gate not satisfied")
        publish("test_evaluation")
        test_start = time.monotonic()
        test_fold_path = Path(flags.dataset_path, "folds/test_fold_setting1.txt")
        result["fold_identity"]["heldout_test_fold_sha256"] = sha256(test_fold_path)
        result["fold_identity"]["heldout_test_fold_file_read"] = True
        result["test_set_used"] = True
        emit("test_fold_opened")
        if result["fold_identity"]["heldout_test_fold_sha256"] != EXPECTED_TEST_FOLD_SHA256:
            raise RuntimeError("Held-out test-fold hash differs from inspected local file")
        with test_fold_path.open(
                encoding="utf-8") as stream:
            testinds = json.load(stream)
        if (len(testinds) != 834 or len(set(testinds)) != 834
                or set(testinds).intersection(pool)
                or any(index < 0 or index >= len(rows) for index in testinds)
                or len(set(testinds).union(pool)) != len(rows)):
            raise RuntimeError("Held-out test fold identity/coverage invalid")
        test_drugs, test_prots, test_Y = native.prepare_interaction_pairs(
            XD, XT, Y, rows[testinds], cols[testinds])
        test_gen = native.DataGenerator(
            test_drugs, test_prots, test_Y, pro2vec, flags.batch_size,
            flags.max_seq_len, index2char, shuffle=False)
        if len(test_gen) != 4:
            raise RuntimeError("Held-out test generator size invalid")
        result["test_evaluation_count"] = 1
        publish(test_evaluation_count=1)
        emit("test_evaluation_started", count=1)
        test_predictions = model.predict_generator(test_gen, verbose=0)
        test_metrics = full_vector_metrics(native, test_Y, test_predictions)
        if test_metrics["prediction_count"] != 834:
            raise RuntimeError("Held-out test prediction count differs from 834")
        test_metrics["native_ci"] = reconstructed_native_ci(
            native, test_Y, test_predictions, flags.batch_size)
        test_metrics["native_ci_reconstructed_from_single_pass"] = True
        test_metrics["native_ci_method"] = (
            "source cindex_score on each native batch, sample-count weighted; "
            "reconstructed from the single prediction pass")
        save_array(OUTPUT / "test_predictions.npy", test_predictions)
        save_array(OUTPUT / "test_targets.npy", test_Y)
        result["test"] = {
            "metrics": test_metrics,
            "metric_gaps_to_paper": paper_gaps(test_metrics),
            "absolute_differences_to_paper": {
                name: abs(test_metrics[key] - PAPER_REFERENCE[name])
                for name, key in (("ci", "global_ci"), ("mse", "mse"),
                                  ("aupr", "aupr"), ("rm2", "rm2"))},
        }
        result["test_prediction_count"] = 834
        emit("test_evaluation_completed", prediction_count=834,
             metrics=test_metrics)
        sample_boundary("after_test_evaluation", result)
        result["timings_seconds"]["test_evaluation"] = time.monotonic() - test_start
        result["execution_status"] = "completed"
        result["status"] = "completed"
        publish("completed", current_epoch=completed,
                estimated_remaining_seconds=0.0)
    except BaseException as error:
        result["status"] = "failed"
        result["execution_status"] = "failed"
        result["stopping_reason"] = phase + " exception"
        result["failure_classification"] = classify_failure(phase, error)
        if result["failure_classification"] == "resource failure":
            result["resource_status"] = "limit_or_oom"
        result["exception"] = {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "phase": phase,
        }
        result["finite"] = not isinstance(error, FloatingPointError)
        emit("worker_exception", exception=result["exception"])
        publish("failed", error=result["exception"],
                test_evaluation_count=result["test_evaluation_count"])
        traceback.print_exc()
    finally:
        result["timings_seconds"]["worker_total"] = time.monotonic() - started
        write_json(OUTPUT / "worker_summary.json", result)
    return 0 if result["status"] == "completed" else 1


def process_rss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def gpu_memory():
    command = [
        "nvidia-smi", "-i", "0",
        "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"Expected one physical GPU 0 record: {lines}")
    index, name, total, used = [value.strip() for value in lines[0].split(",")]
    if index != "0":
        raise RuntimeError(f"Expected GPU index 0: {index}")
    return {"index": 0, "name": name, "total_mib": int(total), "used_mib": int(used)}


def terminate_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def parent():
    try:
        OUTPUT.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Output directory already exists: {OUTPUT}. No run or overwrite was attempted.")
        return 2
    # Keep the marker after completion to prevent an accidental second experiment.
    lock = OUTPUT / "experiment_started.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print(f"Experiment already started. Read {OUTPUT / 'summary.json'}; no retry was made.")
        return 2
    with os.fdopen(descriptor, "w") as stream:
        stream.write(f"started_utc_epoch={time.time()}\n")
    (OUTPUT / "training.log").touch(exist_ok=True)
    (OUTPUT / "memory.jsonl").touch(exist_ok=True)
    (OUTPUT / "events.jsonl").touch(exist_ok=True)

    start = time.monotonic()
    summary = {
        "status": "preparing", "execution_status": "preparing",
        "convergence_status": None,
        "numerical_comparison_status": "descriptive_only_pending_protocol_alignment",
        "resource_status": "within_limits",
        "stopping_reason": None, "exception": None,
        "failure_classification": None, "test_set_used": False,
        "test_evaluation_count": 0, "test_prediction_count": 0,
        "environment": {}, "GPU": {}, "dataset": {}, "fold_identity": {},
        "fold_sizes": {},
        "configuration": {}, "gan": {}, "dta": {}, "validation": {}, "test": {},
        "finite": None, "timings_seconds": {}, "host_memory": {},
        "gpu_memory": {}, "artifacts": {},
        "paper_reference": PAPER_REFERENCE,
        "resume": {"exact_resume_supported": False,
                   "reason": "weights alone omit optimizer, callback and RNG states; restart fresh"},
        "monitor_limits": {"wall_seconds": WALL_LIMIT_SECONDS,
                           "host_rss_bytes": HOST_RSS_LIMIT_BYTES,
                           "gpu_headroom_mib": GPU_HEADROOM_MIB,
                           "poll_seconds": POLL_SECONDS},
    }
    write_json(OUTPUT / "summary.json", summary)
    proc = None
    samples = []
    monitoring_error = None
    fingerprints = None
    source = None
    try:
        source = discover_source()
        fingerprints = prepare_source(source)
        if fingerprints != EXPECTED_CODE_SHA256:
            raise RuntimeError("Kaggle baseline source hashes differ from inspected local source")
        summary["source"] = {
            "kaggle_input": str(source), "working_copy": str(WORK_SOURCE),
            "source_sha256": fingerprints,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "local_nested_HEAD_at_preparation": "453fe16a3c6279dff3ddd4bd28ee63f50465b799",
            "local_root_HEAD_at_preparation": "ac5809b8fa4204f5c44c4faf3f5f5bbeb9539c28",
            "input_git_identity_verified": False,
            "heldout_test_fold_file_read_or_hashed_before_training": False,
        }
        data_files = (
            "Y", "ligands.txt", "proteins.txt", "ligands_train.txt",
            "proteins_train.txt", "protein_feature_vec.json",
            "protein_feature_vecblsm.json", "folds/train_fold_setting1.txt",
        )
        summary["source"]["pdb_data_sha256"] = {
            name: sha256(source / "data/pdb" / name) for name in data_files
        }
        if summary["source"]["pdb_data_sha256"] != EXPECTED_INPUT_SHA256:
            raise RuntimeError("Kaggle PDBbind input or training-fold hashes differ from inspected local files")
        first_gpu = gpu_memory()  # Fail before training if the GPU monitor is unavailable.
        if "T4" not in first_gpu["name"]:
            raise RuntimeError(f"Expected a Tesla T4 at physical GPU 0: {first_gpu['name']}")
        if first_gpu["total_mib"] <= GPU_HEADROOM_MIB:
            raise RuntimeError("GPU memory capacity is too small for the monitor boundary")
        gpu_limit = first_gpu["total_mib"] - GPU_HEADROOM_MIB
        if first_gpu["used_mib"] >= gpu_limit:
            raise RuntimeError("GPU 0 is already above the memory safety boundary")
        summary["GPU"] = {**first_gpu, "memory_limit_mib": gpu_limit}
        summary["status"] = "running"
        summary["execution_status"] = "running"
        write_json(OUTPUT / "summary.json", summary)

        environment = dict(os.environ)
        environment.update({
            "CUDA_VISIBLE_DEVICES": "0",
            "TF_USE_LEGACY_KERAS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        with (OUTPUT / "training.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                cwd=WORK_SOURCE, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            print(f"Worker PID {proc.pid}; source {source}; output {OUTPUT}", flush=True)
            last_gpu = first_gpu
            next_gpu_sample = 0.0
            announced = set()
            with (OUTPUT / "memory.jsonl").open("w", encoding="utf-8") as memory_log:
                while proc.poll() is None:
                    elapsed = time.monotonic() - start
                    rss = process_rss_bytes(proc.pid)
                    if rss is None and proc.poll() is None:
                        # VmRSS may disappear while a completed worker is exiting.
                        try:
                            proc.wait(timeout=POLL_SECONDS)
                        except subprocess.TimeoutExpired:
                            summary["stopping_reason"] = "worker RSS monitor unavailable"
                    if elapsed >= next_gpu_sample and proc.poll() is None:
                        try:
                            last_gpu = gpu_memory()
                            monitoring_error = None
                        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
                            monitoring_error = repr(error)
                            summary["stopping_reason"] = "GPU memory monitor failed: " + monitoring_error
                        next_gpu_sample = elapsed + 5.0
                    sample = {
                        "epoch_seconds": time.time(), "parent_elapsed_seconds": elapsed,
                        "worker_pid": proc.pid, "worker_rss_bytes": rss,
                        "gpu_used_mib": last_gpu["used_mib"],
                        "gpu_total_mib": last_gpu["total_mib"],
                    }
                    samples.append(sample)
                    memory_log.write(json.dumps(sample) + "\n")
                    memory_log.flush()
                    for event in read_events():
                        kind = event["kind"]
                        if kind in {"data_ready", "gan_completed", "variant_c_constructed",
                                    "dta_epoch_completed", "best_validation_completed",
                                    "test_evaluation_completed"} and (
                                        kind != "dta_epoch_completed" or
                                        event.get("epoch") in MILESTONES) and (
                                        kind, event.get("epoch")) not in announced:
                            print(f"{kind} at {elapsed:.1f}s", flush=True)
                            announced.add((kind, event.get("epoch")))
                    if proc.poll() is None:
                        if elapsed > WALL_LIMIT_SECONDS:
                            summary["stopping_reason"] = (
                                f"total wall time exceeded {WALL_LIMIT_SECONDS} seconds")
                        elif rss is not None and rss > HOST_RSS_LIMIT_BYTES:
                            summary["stopping_reason"] = "worker RSS exceeded 12 GiB"
                        elif last_gpu["used_mib"] >= gpu_limit:
                            summary["stopping_reason"] = "GPU 0 memory reached capacity minus 512 MiB"
                    if summary["stopping_reason"] is not None:
                        print("Safety stop:", summary["stopping_reason"], flush=True)
                        terminate_group(proc)
                        break
                    time.sleep(POLL_SECONDS)
            proc.wait()

        parent_stop_reason = summary["stopping_reason"]
        worker_file = OUTPUT / "worker_summary.json"
        if worker_file.exists():
            worker_result = json.loads(worker_file.read_text(encoding="utf-8"))
            summary.update(worker_result)
        else:
            recover_progress(summary, read_events())
        if parent_stop_reason is not None:
            summary["stopping_reason"] = parent_stop_reason
            summary["status"] = "stopped_by_safety_limit"
            summary["execution_status"] = "stopped_by_safety_limit"
            summary["resource_status"] = "safety_limit_or_monitor_failure"
            summary["failure_classification"] = (
                "harness failure" if "monitor" in parent_stop_reason.lower()
                else "resource failure"
            )
        elif proc.returncode != 0:
            summary["status"] = "failed"
            summary["execution_status"] = "failed"
            summary["stopping_reason"] = summary["stopping_reason"] or f"worker exit {proc.returncode}"
            summary["failure_classification"] = summary.get("failure_classification") or "harness failure"
        elif summary.get("status") != "completed":
            summary["status"] = "failed"
            summary["execution_status"] = "failed"
            summary["stopping_reason"] = "worker exited without complete held-out test evidence"
            summary["failure_classification"] = "harness failure"
        summary["worker_returncode"] = proc.returncode
        summary["worker_pid"] = proc.pid
    except BaseException as error:
        summary["status"] = "failed"
        summary["execution_status"] = "failed"
        summary["stopping_reason"] = summary["stopping_reason"] or "parent exception"
        summary["failure_classification"] = "harness failure"
        summary["exception"] = {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "phase": "parent",
        }
        traceback.print_exc()
    finally:
        if proc is not None and proc.poll() is None:
            try:
                terminate_group(proc)
            except BaseException as error:
                summary["termination_error"] = repr(error)
        if source is not None and fingerprints is not None:
            try:
                summary.setdefault("source", {})["unchanged_after_run"] = all(
                    sha256(source / name) == digest for name, digest in fingerprints.items()
                )
            except OSError as error:
                summary.setdefault("source", {})["integrity_check_error"] = repr(error)
        if samples:
            rss_values = [sample["worker_rss_bytes"] for sample in samples
                          if sample["worker_rss_bytes"] is not None]
            gpu_values = [sample["gpu_used_mib"] for sample in samples]
            summary["host_memory"] = {
                "first_rss_bytes": rss_values[0] if rss_values else None,
                "max_observed_rss_bytes": max(rss_values) if rss_values else None,
                "last_observed_rss_bytes": rss_values[-1] if rss_values else None,
                "sample_count": len(samples), "limit_bytes": HOST_RSS_LIMIT_BYTES,
            }
            summary["gpu_memory"] = {
                "first_used_mib": gpu_values[0], "max_observed_used_mib": max(gpu_values),
                "last_used_mib": gpu_values[-1], "limit_mib": summary.get("GPU", {}).get("memory_limit_mib"),
                "scope": "physical GPU 0, including any other processes",
            }
            boundary_names = (
                "environment_ready", "data_ready", "gan_started", "gan_completed",
                "variant_c_constructed", "dta_fit_started", "dta_epoch_completed",
                "best_validation_completed", "test_evaluation_completed",
            )
            boundaries = {}
            for event in read_events():
                if event["kind"] in boundary_names:
                    nearest = min(samples, key=lambda item: abs(
                        item["epoch_seconds"] - event["epoch_seconds"]
                    ))
                    boundaries[event["kind"]] = {
                        "event_epoch_seconds": event["epoch_seconds"],
                        "sample_offset_seconds": nearest["epoch_seconds"] - event["epoch_seconds"],
                        "worker_rss_bytes": nearest["worker_rss_bytes"],
                        "gpu_used_mib": nearest["gpu_used_mib"],
                    }
            summary["sampled_memory_near_events"] = boundaries
        summary.setdefault("timings_seconds", {})["total_parent"] = time.monotonic() - start
        summary["artifacts"] = {
            "summary": str(OUTPUT / "summary.json"),
            "training_log": str(OUTPUT / "training.log"),
            "memory_log": str(OUTPUT / "memory.jsonl"),
            "events": str(OUTPUT / "events.jsonl"),
            "history_csv": str(OUTPUT / "history.csv"),
            "history_jsonl": str(OUTPUT / "history.jsonl"),
            "live_status": str(OUTPUT / "live_status.json"),
            "milestones": str(OUTPUT / "milestones.jsonl"),
            "best_checkpoint": str(OUTPUT / "best_model.weights.h5"),
            "final_checkpoint": str(OUTPUT / "final_model.weights.h5"),
            "validation_predictions_best": str(OUTPUT / "validation_predictions_best.npy"),
            "validation_targets": str(OUTPUT / "validation_targets.npy"),
            "test_predictions": str(OUTPUT / "test_predictions.npy"),
            "test_targets": str(OUTPUT / "test_targets.npy"),
        }
        config = summary.get("configuration", {})
        gan = summary.get("gan", {})
        dta = summary.get("dta", {})
        validation = summary.get("validation", {})
        test = summary.get("test", {})
        observed_gpu = summary.get("gpu_memory", {}).get("max_observed_used_mib")
        observed_rss = summary.get("host_memory", {}).get("max_observed_rss_bytes")
        gpu_limit = summary.get("GPU", {}).get("memory_limit_mib")
        completion_evidence = {
            "fixed_configuration": (
                config.get("model") == "C" and config.get("num_filters") == 128
                and config.get("drug_kernel") == 4 and config.get("protein_kernel") == 8
                and config.get("batch_size") == 256 and config.get("max_epochs") == MAX_EPOCHS
                and config.get("problem_type") == 1 and config.get("is_log") == 0),
            "one_fresh_native_drug_gan": (
                gan.get("calls") == 1 and gan.get("completed") is True),
            "bounded_complete_dta_epochs": (
                1 <= dta.get("epochs_completed", 0) <= MAX_EPOCHS
                and dta.get("optimizer_updates") ==
                dta.get("epochs_completed", 0) * 14),
            "best_validation_checkpoint_restored": (
                validation.get("best_checkpoint_restored") is True),
            "exactly_one_heldout_test_pass": (
                summary.get("test_evaluation_count") == 1
                and summary.get("test_prediction_count") == 834
                and test.get("metrics", {}).get("prediction_count") == 834),
            "sampled_resources_within_limits": (
                observed_rss is not None and observed_rss <= HOST_RSS_LIMIT_BYTES
                and observed_gpu is not None and gpu_limit is not None
                and observed_gpu < gpu_limit
                and summary["timings_seconds"]["total_parent"] <= WALL_LIMIT_SECONDS),
            "source_copy_unchanged": (
                summary.get("source", {}).get("unchanged_after_run") is True),
            "worker_exit_zero": proc is not None and proc.returncode == 0,
        }
        summary["completion_evidence"] = completion_evidence
        if summary["status"] == "completed" and not all(completion_evidence.values()):
            summary["status"] = "failed"
            summary["execution_status"] = "failed"
            summary["stopping_reason"] = "completion evidence incomplete"
            if not completion_evidence["sampled_resources_within_limits"]:
                summary["resource_status"] = "safety_limit_exceeded"
                summary["failure_classification"] = "resource failure"
            else:
                summary["failure_classification"] = "harness failure"
        if summary["status"] != "completed":
            live_path = OUTPUT / "live_status.json"
            try:
                live = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else {}
            except (OSError, json.JSONDecodeError):
                live = {}
            live.update({
                "phase": "failed", "error": summary.get("stopping_reason"),
                "last_update_timestamp": time.time(),
                "paper_reference": PAPER_REFERENCE,
                "test_evaluation_count": summary.get("test_evaluation_count", 0),
            })
            write_json(live_path, live)
        write_json(OUTPUT / "summary.json", summary)
        print(f"Final status: {summary['status']}; reason: {summary['stopping_reason']}", flush=True)
        print(f"Summary: {OUTPUT / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        sys.exit(full_worker())
    if sys.argv[1:]:
        raise SystemExit("Usage: python stage09a_dcgan_full_kaggle.py")
    sys.exit(parent())
