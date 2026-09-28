#!/usr/bin/env python3
"""One bounded Kaggle T4 validation of native DCGAN-DTA Variant C.

Attach this file and the DCGAN-DTA repository as Kaggle inputs, then run this
file as a script. No baseline source or input data is edited. A parent process
monitors one worker and writes evidence under /kaggle/working/stage08_5a/.
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
OUTPUT = Path("/kaggle/working/stage08_5a")
WORK_SOURCE = OUTPUT / "source"
WALL_LIMIT_SECONDS = 30 * 60
HOST_RSS_LIMIT_BYTES = 12 * 1024**3
GPU_HEADROOM_MIB = 512
POLL_SECONDS = 1.0

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
            summary["fold_sizes"] = event["fold_sizes"]
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


def worker():
    started = time.monotonic()
    result = {
        "status": "running",
        "stopping_reason": None,
        "exception": None,
        "environment": {},
        "dataset": {},
        "fold_sizes": {},
        "configuration": {},
        "gan": {"completed": False, "calls": 0, "runtime_seconds": None},
        "dta": {
            "constructed": False,
            "parameters": None,
            "optimizer_updates": 0,
            "epoch_completed": False,
            "training_metrics": {},
            "batch_metrics": [],
        },
        "validation": {
            "fit_reached": False, "fit_batch_count": 0,
            "reached": False, "metrics": {},
        },
        "finite": True,
        "timings_seconds": {},
        "deviations": [
            "one first grid configuration",
            "one first shipped validation fold",
            "one DTA epoch",
            "no final test pass or plotting/session reset",
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
        # The native parser supplies all other defaults. This is one grid point.
        sys.argv = [
            "run_experiments.py",
            "--dataset_path", "data/pdb/",
            "--problem_type", "1", "--is_log", "0", "--model", "C",
            "--num_windows", "128", "--smi_window_lengths", "4",
            "--seq_window_lengths", "4", "--batch_size", "256",
            "--num_epoch", "1", "--max_smi_len", "200",
            "--max_seq_len", "2000", "--log_dir", str(OUTPUT),
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
        assert flags.model == "C" and flags.num_windows == [128]
        assert flags.smi_window_lengths == [4] and flags.seq_window_lengths == [4]
        assert flags.batch_size == 256 and flags.num_epoch == 1
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
        test_set, outer_train_sets = dataset.read_sets(flags)
        assert len(outer_train_sets) == 5
        valinds = outer_train_sets[0]
        traininds = [index for fold in outer_train_sets[1:] for index in fold]
        result["dataset"] = {
            "name": "PDBbind", "XD_shape": list(XD.shape),
            "XT_shape": list(XT.shape), "Y_shape": list(Y.shape),
            "XD_t_shape": list(XD_t.shape),
            "observed_pairs": len(rows),
            "split_source": "shipped data/pdb/folds/*_fold_setting1.txt",
        }
        result["fold_sizes"] = {
            "fold_index_zero_based": 0, "train": len(traininds),
            "validation": len(valinds), "test_not_run": len(test_set),
            "train_batches": (len(traininds) + 255) // 256,
            "validation_batches": (len(valinds) + 255) // 256,
        }
        if (len(rows), len(traininds), len(valinds), len(test_set)) != (5014, 3344, 836, 834):
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
        result["timings_seconds"]["dataset_loading"] = time.monotonic() - load_start
        emit("data_ready", dataset=result["dataset"], fold_sizes=result["fold_sizes"])

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
        if result["dta"]["parameters"] != 3_070_593:
            raise RuntimeError("Variant C parameter count differs from Stage 08C")

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

            def on_test_begin(self, logs=None):
                nonlocal validation_fit_start
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
            epochs=1, callbacks=[es, Observe()]
        )
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
        if result["dta"]["optimizer_updates"] != len(train_generator):
            raise RuntimeError("One epoch did not complete all native DTA optimizer updates")
        if result["validation"]["fit_batch_count"] != len(validation_generator):
            raise RuntimeError("Native fit did not evaluate all validation batches")

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
            "reached": True,
            "metrics": metrics,
            "selected_epoch_index_zero_based": selected_epoch_index,
            "prediction_count": len(labels),
            "finite": validation_finite,
            "test_set_used": False,
        }
        result["timings_seconds"]["postfit_prediction_and_metrics"] = (
            time.monotonic() - postfit_start
        )
        result["finite"] = validation_finite
        result["status"] = "completed" if validation_finite else "failed_nonfinite_validation"
        result["stopping_reason"] = (
            "one DTA epoch and native validation completed" if validation_finite
            else "native validation returned non-finite metric"
        )
        emit("native_validation_completed", validation=result["validation"])
    except BaseException as error:
        result["status"] = "failed"
        result["stopping_reason"] = phase + " exception"
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
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

    start = time.monotonic()
    summary = {
        "status": "preparing", "stopping_reason": None, "exception": None,
        "environment": {}, "GPU": {}, "dataset": {}, "fold_sizes": {},
        "configuration": {}, "gan": {}, "dta": {}, "validation": {},
        "finite": None, "timings_seconds": {}, "host_memory": {},
        "gpu_memory": {}, "artifacts": {},
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
        summary["source"] = {
            "kaggle_input": str(source), "working_copy": str(WORK_SOURCE),
            "source_sha256": fingerprints,
            "local_nested_HEAD_at_preparation": "453fe16a3c6279dff3ddd4bd28ee63f50465b799",
            "input_git_identity_verified": False,
        }
        data_files = (
            "Y", "ligands.txt", "proteins.txt", "ligands_train.txt",
            "proteins_train.txt", "protein_feature_vecblsm.json",
            "folds/train_fold_setting1.txt", "folds/test_fold_setting1.txt",
        )
        summary["source"]["pdb_data_sha256"] = {
            name: sha256(source / "data/pdb" / name) for name in data_files
        }
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
        elif proc.returncode != 0:
            summary["status"] = "failed"
            summary["stopping_reason"] = summary["stopping_reason"] or f"worker exit {proc.returncode}"
        elif summary.get("status") != "completed":
            summary["status"] = "failed"
            summary["stopping_reason"] = "worker exited without completed validation evidence"
        summary["worker_returncode"] = proc.returncode
        summary["worker_pid"] = proc.pid
    except BaseException as error:
        summary["status"] = "failed"
        summary["stopping_reason"] = summary["stopping_reason"] or "parent exception"
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
            summary["memory_at_boundaries"] = boundaries
        summary.setdefault("timings_seconds", {})["total_parent"] = time.monotonic() - start
        summary["artifacts"] = {
            "summary": str(OUTPUT / "summary.json"),
            "training_log": str(OUTPUT / "training.log"),
            "memory_log": str(OUTPUT / "memory.jsonl"),
            "events": str(OUTPUT / "events.jsonl"),
        }
        write_json(OUTPUT / "summary.json", summary)
        print(f"Final status: {summary['status']}; reason: {summary['stopping_reason']}", flush=True)
        print(f"Summary: {OUTPUT / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        sys.exit(worker())
    if sys.argv[1:]:
        raise SystemExit("Usage: python stage08_5a_kaggle.py")
    sys.exit(parent())
