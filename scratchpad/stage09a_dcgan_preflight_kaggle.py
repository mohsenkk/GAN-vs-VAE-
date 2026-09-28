#!/usr/bin/env python3
"""Stage 09A: one bounded 128/4/8 PDBbind Variant C Kaggle preflight.

Attach this file and the DCGAN-DTA repository as Kaggle inputs, then run this
file as a script. No baseline source or input data is edited. A parent process
monitors one worker and writes evidence under /kaggle/working/stage09a_preflight/.
Only shipped training fold 0 is validation; the held-out test fold is not read.
"""

import hashlib
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
OUTPUT = Path("/kaggle/working/stage09a_preflight")
WORK_SOURCE = OUTPUT / "source"
WALL_LIMIT_SECONDS = 30 * 60
HOST_RSS_LIMIT_BYTES = 12 * 1024**3
GPU_HEADROOM_MIB = 512
POLL_SECONDS = 1.0

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
        elif kind == "dta_batch_completed":
            dta = summary.setdefault("dta", {})
            dta["optimizer_updates"] = event["optimizer_updates"]
            dta.setdefault("batch_metrics", []).append({
                "batch_zero_based": event["batch_zero_based"],
                "optimizer_updates": event["optimizer_updates"],
                "metrics": event["metrics"],
            })
        elif kind == "dta_epoch_completed":
            summary.setdefault("dta", {}).update({
                "epoch_completed": True,
                "optimizer_updates": event["optimizer_updates"],
                "history": event["history"],
            })
        elif kind == "fit_validation_started":
            summary.setdefault("validation", {})["fit_reached"] = True
        elif kind == "fit_validation_batch":
            validation = summary.setdefault("validation", {})
            validation["fit_batch_count"] = validation.get("fit_batch_count", 0) + 1
        elif kind == "native_validation_completed":
            summary["validation"] = event["validation"]
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


def measured_metrics(values):
    """Preserve NaN/Inf as JSON strings so failed validation remains inspectable."""
    import numpy as np

    observed = {}
    for key, value in values.items():
        number = float(np.asarray(value).reshape(()))
        if math.isnan(number):
            observed[key] = "NaN"
        elif math.isinf(number):
            observed[key] = "+Inf" if number > 0 else "-Inf"
        else:
            observed[key] = number
    return observed


def classify_failure(phase, error):
    detail = (type(error).__name__ + " " + str(error)).lower()
    if "out of memory" in detail or "resourceexhausted" in detail or "resource limit" in detail:
        return "resource failure"
    if "monitor unavailable" in detail or "nvidia-smi" in detail:
        return "harness failure"
    return {
        "environment": "compatibility failure",
        "data_loading": "harness failure",
        "gan_and_construction": "model construction failure",
        "dta_training": "training failure",
        "native_validation": "validation failure",
    }.get(phase, "harness failure")


def sample_boundary(label, result):
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
    if rss > HOST_RSS_LIMIT_BYTES or gpu["used_mib"] >= gpu["total_mib"] - GPU_HEADROOM_MIB:
        raise RuntimeError("Resource limit crossed at " + label)
    return sample


def worker():
    started = time.monotonic()
    result = {
        "status": "running",
        "stopping_reason": None,
        "exception": None,
        "failure_classification": None,
        "test_set_used": False,
        "environment": {},
        "dataset": {},
        "fold_identity": {},
        "fold_sizes": {},
        "configuration": {},
        "gan": {"completed": False, "calls": 0, "runtime_seconds": None,
                "native_schedule": {"iterations": 5000, "batch_size": 5, "log_interval": 500}},
        "dta": {
            "constructed": False,
            "parameters": None,
            "optimizer_updates": 0,
            "epoch_completed": False,
            "training_metrics": {},
            "batch_metrics": [],
            "first_training_batch_size": None,
        },
        "validation": {
            "fit_reached": False, "fit_batch_count": 0,
            "reached": False, "metrics": {},
        },
        "finite": True,
        "timings_seconds": {},
        "controlled_deviations": [
            "fixed paper-reported 128/4/8 configuration; no README grid search",
            "one first shipped training fold used for validation",
            "one DTA epoch",
            "no held-out test-fold read or evaluation; no second pass or plots",
            "read-only Kaggle input accessed through a writable source copy",
            "standalone keras imports mapped to verified tf_keras 2.20 runtime",
            "unused native np.mat(Y copy) allocation omitted (NumPy 2 removed np.mat)",
        ],
    }
    phase = "environment"
    try:
        import numpy as np
        import tensorflow as tf

        physical_gpus = tf.config.list_physical_devices("GPU")
        if len(physical_gpus) != 1:
            raise RuntimeError(f"Expected exactly one visible T4; TensorFlow sees {physical_gpus}")
        # The selected path does not call reset_keras(); growth avoids preallocation.
        tf.config.experimental.set_memory_growth(physical_gpus[0], True)

        import tf_keras

        if sys.version_info[:2] != (3, 12):
            raise RuntimeError(f"Kaggle Python differs from verified 3.12: {sys.version}")
        if not tf.__version__.startswith("2.20."):
            raise RuntimeError(f"TensorFlow differs from verified 2.20: {tf.__version__}")
        if not tf_keras.__version__.startswith("2.20."):
            raise RuntimeError(f"tf_keras differs from verified 2.20: {tf_keras.__version__}")
        if np.__version__ != "2.0.2":
            raise RuntimeError(f"NumPy differs from verified 2.0.2: {np.__version__}")

        # Map the original standalone `keras` imports to the verified legacy API.
        for suffix in ("backend", "layers", "models", "callbacks", "utils"):
            sys.modules["keras." + suffix] = importlib.import_module("tf_keras." + suffix)
        sys.modules["keras"] = tf_keras

        legacy_backend = tf.compat.v1.keras.backend
        get_session = callable(getattr(legacy_backend, "get_session", None))
        set_session = callable(getattr(legacy_backend, "set_session", None))
        if not (get_session and set_session):
            raise RuntimeError("Expected legacy get_session/set_session are unavailable")
        if not callable(getattr(tf_keras.Model, "fit_generator", None)):
            raise RuntimeError("Legacy Model.fit_generator is unavailable")
        if not callable(getattr(tf_keras.Model, "predict_generator", None)):
            raise RuntimeError("Legacy Model.predict_generator is unavailable")

        details = tf.config.experimental.get_device_details(physical_gpus[0])
        result["environment"] = {
            "python": sys.version,
            "tensorflow": tf.__version__,
            "tf_keras": tf_keras.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_use_legacy_keras": os.environ["TF_USE_LEGACY_KERAS"],
            "physical_gpus": [str(gpu) for gpu in physical_gpus],
            "logical_gpus": [str(gpu) for gpu in tf.config.list_logical_devices("GPU")],
            "gpu_details": details,
            "legacy_get_session": get_session,
            "legacy_set_session": set_session,
            "memory_growth": tf.config.experimental.get_memory_growth(physical_gpus[0]),
        }
        print("Environment verified:", result["environment"], flush=True)
        emit("environment_ready", environment=result["environment"])
        result["timings_seconds"]["environment_initialization"] = time.monotonic() - started

        os.chdir(WORK_SOURCE)
        sys.path.insert(0, str(WORK_SOURCE))
        # The native parser supplies other defaults. This is one fixed point.
        sys.argv = [
            "run_experiments.py",
            "--dataset_path", "data/pdb/",
            "--problem_type", "1", "--is_log", "0", "--model", "C",
            "--num_windows", "128", "--smi_window_lengths", "4",
            "--seq_window_lengths", "8", "--batch_size", "256",
            "--num_epoch", "1", "--max_smi_len", "200",
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
        if not (
            flags.model == "C" and flags.num_windows == [128]
            and flags.smi_window_lengths == [4] and flags.seq_window_lengths == [8]
            and flags.problem_type == 1 and flags.is_log == 0
            and flags.batch_size == 256 and flags.num_epoch == 1
            and flags.max_smi_len == 200 and flags.max_seq_len == 2000
            and flags.learning_rate == 0.001
        ):
            raise RuntimeError("Parsed configuration differs from Stage 09A 128/4/8 preflight")
        result["configuration"] = {
            "dataset_path": flags.dataset_path,
            "problem_type": flags.problem_type,
            "is_log": flags.is_log,
            "model": flags.model,
            "num_filters": flags.num_windows[0],
            "drug_kernel": flags.smi_window_lengths[0],
            "protein_kernel": flags.seq_window_lengths[0],
            "batch_size": flags.batch_size,
            "epochs": flags.num_epoch,
            "requested_learning_rate": flags.learning_rate,
            "native_compile_optimizer": "adam default (source does not consume learning_rate flag)",
            "max_smi_len": flags.max_smi_len,
            "max_seq_len": flags.max_seq_len,
        }

        phase = "data_loading"
        load_start = time.monotonic()
        dataset = DataSet(
            fpath=flags.dataset_path,
            setting_no=flags.problem_type,
            seqlen=flags.max_seq_len,
            smilen=flags.max_smi_len,
            need_shuffle=False,
        )
        flags.charseqset_size = dataset.charseqset_size
        flags.charsmiset_size = dataset.charsmiset_size
        XD, XT, Y, XD_t, XT_t = dataset.parse_data(flags)
        XD, XT, Y, XD_t = map(np.asarray, (XD, XT, Y, XD_t))
        del XT_t  # Variant C never consumes target-GAN training data.
        rows, cols = np.where(np.isnan(Y) == False)
        # read_sets() would open the held-out test-fold file. Read only the
        # shipped training-fold file; no test indices enter this process.
        train_fold_path = Path(flags.dataset_path, "folds/train_fold_setting1.txt")
        with train_fold_path.open(encoding="utf-8") as stream:
            outer_train_sets = json.load(stream)
        if len(outer_train_sets) != 5:
            raise RuntimeError("Expected exactly five shipped training folds")
        valinds = outer_train_sets[0]
        traininds = [index for fold in outer_train_sets[1:] for index in fold]
        training_pool = [index for fold in outer_train_sets for index in fold]
        if (len(training_pool) != len(set(training_pool))
                or any(index < 0 or index >= len(rows) for index in training_pool)):
            raise RuntimeError("Shipped training folds overlap or contain invalid pair indices")
        heldout_count_inferred = len(rows) - len(training_pool)
        result["dataset"] = {
            "name": "PDBbind", "XD_shape": list(XD.shape),
            "XT_shape": list(XT.shape), "Y_shape": list(Y.shape),
            "XD_t_shape": list(XD_t.shape),
            "observed_pairs": len(rows),
            "split_source": str(train_fold_path),
            "native_parse_data_loads_full_affinity_matrix": True,
        }
        result["fold_identity"] = {
            "validation_fold_zero_based": 0,
            "training_folds_zero_based": [1, 2, 3, 4],
            "heldout_test_fold_file_read": False,
            "heldout_pair_count_inferred_from_remainder": heldout_count_inferred,
        }
        result["fold_sizes"] = {
            "fold_index_zero_based": 0, "train": len(traininds),
            "validation": len(valinds),
            "heldout_test_pairs_inferred_not_loaded": heldout_count_inferred,
        }
        if (len(rows), len(traininds), len(valinds), heldout_count_inferred) != (5014, 3344, 836, 834):
            raise RuntimeError("Shipped fold/data sizes differ from Stage 08C; stop before training")
        if list(XD_t.shape) != [50068, 200]:
            raise RuntimeError("The GAN drug corpus differs from Stage 08B/08C")

        # This reproduces general_nfold_cv's first-fold pair preparation and
        # its native DataGenerator, including shuffle=False and __getitem__.
        trrows, trcols = rows[traininds], cols[traininds]
        varows, vacols = rows[valinds], cols[valinds]
        train_drugs, train_prots, train_Y = native.prepare_interaction_pairs(
            XD, XT, Y, trrows, trcols
        )
        val_drugs, val_prots, val_Y = native.prepare_interaction_pairs(
            XD, XT, Y, varows, vacols
        )
        with Path(flags.dataset_path, "protein_feature_vecblsm.json").open() as stream:
            pro2vec = json.load(stream)
        index2char = {index: char for char, index in native.CHARPROTSET.items()}
        train_generator = native.DataGenerator(
            train_drugs, train_prots, train_Y, pro2vec,
            flags.batch_size, flags.max_seq_len, index2char, shuffle=False
        )
        validation_generator = native.DataGenerator(
            val_drugs, val_prots, val_Y, pro2vec,
            flags.batch_size, flags.max_seq_len, index2char, shuffle=False
        )
        first_batch_inputs, first_batch_labels = train_generator[0]
        first_batch_size = int(len(first_batch_labels))
        if first_batch_size != 256 or len(first_batch_inputs[0]) != 256:
            raise RuntimeError("Native generator did not yield a 256-pair first batch")
        del first_batch_inputs, first_batch_labels
        result["dta"]["first_training_batch_size"] = first_batch_size
        result["fold_sizes"]["train_batches_from_native_generator"] = len(train_generator)
        result["fold_sizes"]["validation_batches_from_native_generator"] = len(validation_generator)
        result["timings_seconds"]["dataset_loading"] = time.monotonic() - load_start
        emit("data_ready", dataset=result["dataset"],
             fold_identity=result["fold_identity"], fold_sizes=result["fold_sizes"])
        sample_boundary("before_gan", result)

        phase = "gan_and_construction"
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
                sample_boundary("after_gan_before_dta_construction", result)

        sys.setprofile(gan_profile)
        try:
            model = native.build_GAN_C(
                flags, XD_t, flags.num_windows[0],
                flags.smi_window_lengths[0], flags.seq_window_lengths[0]
            )
        finally:
            sys.setprofile(None)
        if result["gan"]["calls"] != 1 or not result["gan"]["completed"]:
            raise RuntimeError("Exactly one native drug GAN did not complete")
        result["dta"]["constructed"] = True
        result["dta"]["parameters"] = int(model.count_params())
        result["dta"]["optimizer_updates"] = int(model.optimizer.iterations.numpy())
        result["timings_seconds"]["variant_c_construction"] = (
            time.monotonic() - construct_start - result["gan"]["runtime_seconds"]
        )
        emit("variant_c_constructed", parameters=result["dta"]["parameters"])
        if result["dta"]["parameters"] <= 0:
            raise RuntimeError("Variant C has no trainable parameters")
        sample_boundary("after_variant_c_construction", result)

        phase = "dta_training"
        fit_start = time.monotonic()
        validation_fit_start = None
        validation_fit_end = None

        class Observe(tf_keras.callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                metrics = finite_metrics(logs)
                result["dta"]["optimizer_updates"] = int(self.model.optimizer.iterations.numpy())
                row = {"batch_zero_based": int(batch),
                       "optimizer_updates": result["dta"]["optimizer_updates"],
                       "metrics": metrics}
                result["dta"]["batch_metrics"].append(row)
                emit("dta_batch_completed", **row)
                if int(batch) == len(train_generator) // 2:
                    sample_boundary("during_dta_epoch", result)

            def on_test_begin(self, logs=None):
                nonlocal validation_fit_start, phase
                phase = "native_validation"
                validation_fit_start = time.monotonic()
                result["validation"]["fit_reached"] = True
                emit("fit_validation_started")

            def on_test_batch_end(self, batch, logs=None):
                result["validation"]["fit_batch_count"] += 1
                emit("fit_validation_batch", batch_zero_based=int(batch),
                     metrics=finite_metrics(logs))

            def on_test_end(self, logs=None):
                nonlocal validation_fit_end
                validation_fit_end = time.monotonic()
                emit("fit_validation_completed", metrics=finite_metrics(logs))

        es = native.EarlyStopping(
            monitor="val_cindex_score", mode="max", verbose=1,
            patience=75, restore_best_weights=True
        )
        emit("dta_fit_started", planned_epochs=1, planned_batches=len(train_generator))
        # Same native API/arguments, with one read-only observation callback.
        history = model.fit_generator(
            generator=train_generator, validation_data=validation_generator,
            epochs=flags.num_epoch, callbacks=[es, Observe()]
        )
        phase = "dta_training"
        fit_end = time.monotonic()
        result["dta"]["epoch_completed"] = True
        result["dta"]["optimizer_updates"] = int(model.optimizer.iterations.numpy())
        result["dta"]["history"] = {
            key: [finite_metrics({key: item})[key] for item in series]
            for key, series in history.history.items()
        }
        result["dta"]["training_metrics"] = {
            key: series[-1] for key, series in result["dta"]["history"].items()
            if not key.startswith("val_") and series
        }
        result["dta"]["train_loss"] = result["dta"]["training_metrics"].get("loss")
        result["dta"]["train_CI"] = result["dta"]["training_metrics"].get("cindex_score")
        result["timings_seconds"]["fit_including_validation"] = fit_end - fit_start
        result["timings_seconds"]["training_before_fit_validation"] = (
            validation_fit_start - fit_start if validation_fit_start else None
        )
        result["timings_seconds"]["fit_validation"] = (
            validation_fit_end - validation_fit_start
            if validation_fit_end and validation_fit_start else None
        )
        emit("dta_epoch_completed", optimizer_updates=result["dta"]["optimizer_updates"],
             history=result["dta"]["history"])
        if len(result["dta"]["history"].get("loss", [])) != 1:
            raise RuntimeError("Expected exactly one native DTA training epoch")
        if result["dta"]["optimizer_updates"] < 2:
            raise RuntimeError("Fewer than two DTA optimizer updates completed")
        if result["dta"]["optimizer_updates"] != len(train_generator):
            raise RuntimeError("One epoch did not complete all native DTA optimizer updates")
        if result["validation"]["fit_batch_count"] != len(validation_generator):
            raise RuntimeError("Native fit did not evaluate all validation batches")
        sample_boundary("after_dta_epoch_and_fit_validation", result)

        phase = "native_validation"
        postfit_start = time.monotonic()
        predicted_labels = model.predict_generator(validation_generator, verbose=0)
        # Same metric selection and helpers as general_nfold_cv for one epoch.
        validation_ci = result["dta"]["history"]["val_cindex_score"]
        validation_loss = result["dta"]["history"]["val_loss"]
        rperf = max(validation_ci)
        selected_epoch_index = validation_ci.index(rperf)
        loss = validation_loss[selected_epoch_index]
        rm2 = native.get_rm2(val_Y, predicted_labels)
        labels = [row[0] for row in predicted_labels.tolist()]
        if len(labels) != len(val_Y):
            raise RuntimeError("Native prediction count does not match the validation fold")
        temp = [1 if value > 7 else 0 for value in val_Y]
        aupr = native.average_precision_score(temp, labels)
        metrics = measured_metrics({
            "CI": rperf, "MSE_val_loss": loss, "AUPR": aupr, "RM2": rm2
        })
        validation_finite = all(isinstance(value, float) and math.isfinite(value)
                                for value in metrics.values())
        result["validation"] = {
            "fit_reached": True,
            "fit_batch_count": result["validation"]["fit_batch_count"],
            "reached": True,
            "metrics": metrics,
            "selected_epoch_index_zero_based": selected_epoch_index,
            "prediction_count": len(labels),
            "finite": validation_finite,
            "test_set_used": False,
        }
        if result["test_set_used"] is not False:
            raise RuntimeError("Held-out test set was used")
        sample_boundary("after_native_validation", result)
        result["timings_seconds"]["postfit_prediction_and_metrics"] = (
            time.monotonic() - postfit_start
        )
        result["finite"] = validation_finite
        result["status"] = "completed" if validation_finite else "failed_nonfinite_validation"
        result["failure_classification"] = None if validation_finite else "validation failure"
        result["stopping_reason"] = (
            "one DTA epoch and native validation completed" if validation_finite
            else "native validation returned non-finite metric"
        )
        emit("native_validation_completed", validation=result["validation"])
    except BaseException as error:
        result["status"] = "failed"
        result["stopping_reason"] = phase + " exception"
        result["failure_classification"] = classify_failure(phase, error)
        result["exception"] = {
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "phase": phase,
        }
        result["finite"] = not isinstance(error, FloatingPointError)
        emit("worker_exception", exception=result["exception"])
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
        "status": "preparing", "stopping_reason": None, "exception": None,
        "failure_classification": None, "test_set_used": False,
        "environment": {}, "GPU": {}, "dataset": {}, "fold_identity": {},
        "fold_sizes": {},
        "configuration": {}, "gan": {}, "dta": {}, "validation": {},
        "finite": None, "timings_seconds": {}, "host_memory": {},
        "gpu_memory": {}, "artifacts": {},
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
            "input_git_identity_verified": False,
            "heldout_test_fold_file_read_or_hashed": False,
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
                                    "dta_epoch_completed", "native_validation_completed"} and kind not in announced:
                            print(f"{kind} at {elapsed:.1f}s", flush=True)
                            announced.add(kind)
                    if proc.poll() is None:
                        if elapsed > WALL_LIMIT_SECONDS:
                            summary["stopping_reason"] = "total wall time exceeded 1800 seconds"
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
            summary["failure_classification"] = (
                "harness failure" if "monitor" in parent_stop_reason.lower()
                else "resource failure"
            )
        elif proc.returncode != 0:
            summary["status"] = "failed"
            summary["stopping_reason"] = summary["stopping_reason"] or f"worker exit {proc.returncode}"
            summary["failure_classification"] = summary.get("failure_classification") or "harness failure"
        elif summary.get("status") != "completed":
            summary["status"] = "failed"
            summary["stopping_reason"] = "worker exited without completed validation evidence"
            summary["failure_classification"] = "harness failure"
        summary["worker_returncode"] = proc.returncode
        summary["worker_pid"] = proc.pid
    except BaseException as error:
        summary["status"] = "failed"
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
                summary["source"]["unchanged_after_run"] = all(
                    sha256(source / name) == digest for name, digest in fingerprints.items()
                )
            except OSError as error:
                summary["source"]["integrity_check_error"] = repr(error)
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
                "native_validation_completed",
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
        }
        config = summary.get("configuration", {})
        gan = summary.get("gan", {})
        dta = summary.get("dta", {})
        validation = summary.get("validation", {})
        folds = summary.get("fold_sizes", {})
        observed_gpu = summary.get("gpu_memory", {}).get("max_observed_used_mib")
        observed_rss = summary.get("host_memory", {}).get("max_observed_rss_bytes")
        gpu_limit = summary.get("GPU", {}).get("memory_limit_mib")
        criteria = {
            "exact_128_4_8_configuration": (
                config.get("model") == "C" and config.get("num_filters") == 128
                and config.get("drug_kernel") == 4 and config.get("protein_kernel") == 8
                and config.get("problem_type") == 1 and config.get("is_log") == 0
                and config.get("epochs") == 1
            ),
            "one_native_drug_gan_completed": gan.get("calls") == 1 and gan.get("completed") is True,
            "variant_c_constructed": dta.get("constructed") is True and (dta.get("parameters") or 0) > 0,
            "batch_256_used": config.get("batch_size") == 256 and dta.get("first_training_batch_size") == 256,
            "multiple_native_updates_and_one_epoch": (
                dta.get("epoch_completed") is True
                and (dta.get("optimizer_updates") or 0) >= 2
                and dta.get("optimizer_updates") == folds.get("train_batches_from_native_generator")
                and len(dta.get("history", {}).get("loss", [])) == 1
            ),
            "finite_training": summary.get("finite") is True
                and bool(dta.get("training_metrics")),
            "native_validation_reached_and_finite": (
                validation.get("reached") is True and validation.get("finite") is True
                and validation.get("prediction_count") == folds.get("validation")
                and validation.get("fit_batch_count") == folds.get("validation_batches_from_native_generator")
            ),
            "heldout_test_unused": summary.get("test_set_used") is False
                and summary.get("fold_identity", {}).get("heldout_test_fold_file_read") is False,
            "sampled_resource_thresholds_not_crossed": (
                observed_rss is not None and observed_rss <= HOST_RSS_LIMIT_BYTES
                and observed_gpu is not None and gpu_limit is not None
                and observed_gpu < gpu_limit
                and summary["timings_seconds"]["total_parent"] <= WALL_LIMIT_SECONDS
            ),
            "source_copy_unchanged": summary.get("source", {}).get("unchanged_after_run") is True,
            "worker_exit_zero": proc is not None and proc.returncode == 0,
        }
        summary["pass_criteria"] = criteria
        summary["preflight_pass"] = summary["status"] == "completed" and all(criteria.values())
        if summary["status"] == "completed" and not summary["preflight_pass"]:
            summary["status"] = "failed"
            summary["stopping_reason"] = "preflight PASS criteria not all satisfied"
            if not criteria["sampled_resource_thresholds_not_crossed"]:
                summary["failure_classification"] = "resource failure"
            elif not criteria["native_validation_reached_and_finite"]:
                summary["failure_classification"] = "validation failure"
            elif not criteria["multiple_native_updates_and_one_epoch"] or not criteria["finite_training"]:
                summary["failure_classification"] = "training failure"
            else:
                summary["failure_classification"] = "harness failure"
        write_json(OUTPUT / "summary.json", summary)
        print(f"Final status: {summary['status']}; reason: {summary['stopping_reason']}", flush=True)
        print(f"Summary: {OUTPUT / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        sys.exit(worker())
    if sys.argv[1:]:
        raise SystemExit("Usage: python stage09a_dcgan_preflight_kaggle.py")
    sys.exit(parent())
