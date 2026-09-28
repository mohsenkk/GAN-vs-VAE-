#!/usr/bin/env python3
"""One bounded native Co-VAE batch-256 validation on a single Kaggle T4.

Attach this file and the unmodified Co-VAE source/data as Kaggle inputs. Run the
script once; a parent monitors one worker and writes /kaggle/working/stage08_5b/.
No original source, dataset, or fold file is written.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback


INPUT_ROOT = Path("/kaggle/input")
OUTPUT = Path("/kaggle/working/stage08_5b")
WALL_LIMIT_SECONDS = 30 * 60
HOST_RSS_LIMIT_BYTES = 12 * 1024**3
GPU_HEADROOM_MIB = 512
POLL_SECONDS = 1.0
BATCH_SIZE = 256
EPOCHS = 1
SEED = 1000
EXPECTED_TRAIN_INDEX_SHA256 = "74cd831aa2f08dc322e4d2183eb9dc691adf45306ad903a29d9f4f6931e40e5f"
EXPECTED_TEST_INDEX_SHA256 = "57e590130330d13049c835073aa566496138b51105a5a9ebe1b932b6ee2579a8"

SOURCE_HASHES = {
    "run_experiments.py": "39546fa554e628bf4f30dc492d152ed1235fbeedfc83c713218dd0d40d108043",
    "model.py": "a6dc416b3c4257d67e226fb1888d6236c6203f483d943415c5c1a8879e32e8fc",
    "arguments.py": "1b9416dfb212a9c92bda76ec5eea18561530055b9b442d6c0c07e4a483fc388a",
    "datahelper.py": "eb7f1245531be9d5cf9e8099e5063125eac0c5dc3b6dbfe5fda9947f797b917c",
    "emetrics.py": "031696e05fb3c40ea3607ec76c76dcc5079d888d8a7ac2c60658e0cd041fc7f9",
}
DATA_HASHES = {
    "ligands_iso.txt": "9789900f1e723226187215235b379998e5735a3ffec0e9487bad27c1c6a64161",
    "proteins.txt": "187cedeee10cd3b58af915a2861a5ed0c4ff42a2d728f4dea5e0b47e3d75b291",
    "drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt":
        "31b3f3efea9afe53239fa9fe735aedd6195586b99261319917b55b1d1c9e8392",
}

# Both settings must precede the worker's first torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(kind, **fields):
    event = {"kind": kind, "epoch_seconds": time.time(), **fields}
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
            pass  # A safety stop can interrupt the final write.
    return events


def recover_progress(summary, events):
    """Retain measured milestones if the worker cannot write its final summary."""
    for event in events:
        kind = event["kind"]
        if kind == "environment_ready":
            summary["environment"] = event["environment"]
        elif kind == "split_ready":
            summary["dataset"] = event["dataset"]
            summary["split"] = event["split"]
            summary["configuration"] = event["configuration"]
        elif kind == "model_constructed":
            summary["model"] = event["model"]
        elif kind == "optimizer_update":
            summary.setdefault("training", {})["optimizer_updates"] = event["count"]
            if event["count"] == 1:
                summary.setdefault("gpu_allocator", {})["after_first_optimizer_update"] = event["gpu_allocator"]
        elif kind == "train_batch_started" and event["batch_one_based"] == 1:
            summary.setdefault("training", {})["first_batch_size_observed"] = event["batch_size"]
        elif kind == "train_batch_completed":
            training = summary.setdefault("training", {})
            training["completed_batches"] = event["batch_one_based"]
            training["last_batch"] = event["metrics"]
            training.setdefault("first_batch", event["metrics"])
            training["finite"] = training.get("finite", True) and event["finite"]
        elif kind == "epoch_completed":
            summary["training"] = event["training"]
        elif kind == "evaluation_started":
            summary.setdefault("evaluation", {})["started"] = True
        elif kind == "native_evaluation_completed":
            summary["evaluation"] = event["evaluation"]
        elif kind == "worker_exception":
            summary["exception"] = event["exception"]
    return summary


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def digest_indices(indices):
    return hashlib.sha256(
        json.dumps([int(value) for value in indices], separators=(",", ":")).encode("ascii")
    ).hexdigest()


def discover_source():
    matches = []
    for script in INPUT_ROOT.rglob("run_experiments.py"):
        source = script.parent
        if all((source / name).is_file() for name in SOURCE_HASHES):
            if all((source / "data/davis" / name).is_file() for name in DATA_HASHES):
                matches.append(source)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one attached Co-VAE source under {INPUT_ROOT}; found {matches}")
    source = matches[0]
    source_hashes = {name: sha256(source / name) for name in SOURCE_HASHES}
    data_hashes = {name: sha256(source / "data/davis" / name) for name in DATA_HASHES}
    if source_hashes != SOURCE_HASHES or data_hashes != DATA_HASHES:
        raise RuntimeError("Attached Co-VAE source or Davis data differs from the inspected local baseline; stop before training")
    return source, source_hashes, data_hashes


def measured_number(value):
    """Encode numerical failure explicitly; JSON itself cannot contain NaN/Inf."""
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Inf" if number > 0 else "-Inf"
    return number


def scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    import numpy as np
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected one native metric value, got shape {array.shape}")
    return measured_number(array.reshape(-1)[0])


def finite(values):
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in values.values())


def process_rss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def gpu_memory():
    completed = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=index,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=5,
    )
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"Expected one physical GPU 0 record: {lines}")
    index, name, total, used = [part.strip() for part in lines[0].split(",")]
    if index != "0":
        raise RuntimeError(f"Unexpected physical GPU index {index}")
    return {"index": 0, "name": name, "total_mib": int(total), "used_mib": int(used)}


def allocator_snapshot(torch):
    torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def native_flags(RE, source):
    """Use the repository's parser with an explicit singleton configuration."""
    args = [
        "run_experiments.py",
        "--dataset_path", str(source / "data/davis") + "/",
        "--problem_type", "1", "--is_log", "0",
        "--max_smi_len", "85", "--max_seq_len", "1200",
        "--num_windows", "32", "--smi_window_lengths", "4",
        "--seq_window_lengths", "8", "--lamda", "-5",
        "--batch_size", str(BATCH_SIZE), "--num_epoch", str(EPOCHS),
        "--log_dir", str(OUTPUT),
    ]
    original_argv = sys.argv
    try:
        sys.argv = args
        flags = RE.argparser()
    finally:
        sys.argv = original_argv
    if (flags.problem_type, flags.batch_size, flags.num_epoch) != (1, BATCH_SIZE, EPOCHS):
        raise RuntimeError("Native parser changed the requested split, batch, or epoch")
    if (flags.num_windows, flags.smi_window_lengths, flags.seq_window_lengths, flags.lamda) != ([32], [4], [8], [-5]):
        raise RuntimeError("Native parser changed the singleton configuration")
    return flags


def worker():
    started = time.monotonic()
    result = {
        "status": "running", "stopping_reason": None, "exception": None,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "dataset": {}, "split": {}, "seed": SEED,
        "configuration": {}, "model": {"constructed": False},
        "training": {"optimizer_updates": 0, "completed_batches": 0,
                     "epoch_completed": False, "finite": True},
        "evaluation": {"started": False, "reached": False},
        "finite": None, "gpu_allocator": {}, "timings_seconds": {},
        "deviations": [
            "one native runtime-generated split (five folds train, sixth fold test)",
            "one configuration and one training epoch; no grid or repetition",
            "external orchestration and read-only per-batch instrumentation",
            "set population converted to tuple for random.sample on Python 3.11+ if needed",
            "no five-epoch checkpoint or later checkpoint reload",
        ],
    }
    phase = "environment"
    try:
        import importlib.metadata
        import random
        import numpy as np
        import torch

        result["environment"] = {
            "python": sys.version, "platform": platform.platform(),
            "pytorch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_gpu_count": int(torch.cuda.device_count()),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        }
        for package in ("pandas", "scikit-learn", "matplotlib", "tqdm"):
            try:
                result["environment"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                result["environment"][package] = None
        print("Kaggle PyTorch environment:", result["environment"], flush=True)
        if not result["environment"]["cuda_available"] or result["environment"]["visible_gpu_count"] != 1:
            raise RuntimeError("Expected exactly one CUDA-visible GPU")
        device_name = torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        result["environment"]["gpu_name"] = device_name
        result["environment"]["gpu_total_bytes"] = int(properties.total_memory)
        if "T4" not in device_name:
            raise RuntimeError(f"Expected a Tesla T4, got {device_name}")
        emit("environment_ready", environment=result["environment"])
        result["timings_seconds"]["environment_initialization"] = time.monotonic() - started

        phase = "data_and_split"
        source = Path(os.environ["COVAE_SOURCE"])
        os.chdir(source)
        sys.path.insert(0, str(source))
        import run_experiments as RE

        flags = native_flags(RE, source)
        result["configuration"] = {
            "dataset": "Davis", "dataset_path": flags.dataset_path,
            "problem_type": flags.problem_type, "is_log": flags.is_log,
            "seed_for_python_split": SEED, "torch_seed_controlled": False,
            "num_windows": flags.num_windows, "smi_window_lengths": flags.smi_window_lengths,
            "seq_window_lengths": flags.seq_window_lengths, "lamda": flags.lamda,
            "max_smi_len": flags.max_smi_len, "max_seq_len": flags.max_seq_len,
            "batch_size": flags.batch_size, "num_epoch": flags.num_epoch,
        }
        data_started = time.monotonic()
        dataset = RE.DataSet(
            fpath=flags.dataset_path, setting_no=flags.problem_type,
            seqlen=flags.max_seq_len, smilen=flags.max_smi_len, need_shuffle=False,
        )
        flags.charseqset_size = dataset.charseqset_size
        flags.charsmiset_size = dataset.charsmiset_size
        XD, XT, Y = dataset.parse_data(flags)
        XD, XT, Y = np.asarray(XD), np.asarray(XT), np.asarray(Y)
        flags.drug_count, flags.target_count = XD.shape[0], XT.shape[0]
        rows, cols = np.where(np.isnan(Y) == False)
        if XD.shape != (68, 85) or XT.shape != (442, 1200) or Y.shape != (68, 442):
            raise RuntimeError(f"Davis shapes differ from Stage 08: {XD.shape}, {XT.shape}, {Y.shape}")
        if len(rows) != 30056:
            raise RuntimeError(f"Davis observed-pair count changed: {len(rows)}")
        result["dataset"] = {
            "name": "Davis", "XD_shape": list(XD.shape), "XT_shape": list(XT.shape),
            "Y_shape": list(Y.shape), "observed_pairs": len(rows),
        }
        result["timings_seconds"]["dataset_loading"] = time.monotonic() - data_started

        split_started = time.monotonic()
        random.seed(SEED)
        native_sample = random.sample
        shim_used = sys.version_info >= (3, 11)
        if shim_used:
            def sample_compat(population, k):
                return native_sample(tuple(population) if isinstance(population, set) else population, k)

            RE.random.sample = sample_compat
        try:
            folds = RE.get_random_folds(len(rows), 6)
        finally:
            if shim_used:
                RE.random.sample = native_sample
        train_indices = [index for fold in folds[:5] for index in fold]
        test_indices = folds[5]
        if [len(fold) for fold in folds] != [5010, 5010, 5009, 5009, 5009, 5009]:
            raise RuntimeError("Native Davis split sizes differ from Stage 08")
        if len(train_indices) != 25047 or len(test_indices) != 5009:
            raise RuntimeError("Unexpected first native split size")
        if len(set(train_indices) | set(test_indices)) != len(rows) or set(train_indices) & set(test_indices):
            raise RuntimeError("Native split does not partition the observed pairs")
        train_digest, test_digest = digest_indices(train_indices), digest_indices(test_indices)
        if (train_digest, test_digest) != (EXPECTED_TRAIN_INDEX_SHA256, EXPECTED_TEST_INDEX_SHA256):
            raise RuntimeError("Native split membership differs from the local Python 3.10 seed-1000 reference")
        result["split"] = {
            "source": "RE.get_random_folds on observed-pair indices",
            "fold_count": 6, "fold_sizes": [len(fold) for fold in folds],
            "train_fold_indices_zero_based": [0, 1, 2, 3, 4], "test_fold_index_zero_based": 5,
            "train_pairs": len(train_indices), "test_pairs": len(test_indices),
            "train_batches": math.ceil(len(train_indices) / BATCH_SIZE),
            "test_batches": math.ceil(len(test_indices) / BATCH_SIZE),
            "train_indices_sha256": train_digest,
            "test_indices_sha256": test_digest,
            "python_set_sample_shim_used": shim_used,
            "matched_local_python_3_10_split_reference": True,
        }
        trrows, trcols = rows[train_indices], cols[train_indices]
        terows, tecols = rows[test_indices], cols[test_indices]
        train_dataset = RE.prepare_interaction_pairs(XD, XT, Y, trrows, trcols)
        test_dataset = RE.prepare_interaction_pairs(XD, XT, Y, terows, tecols)
        train_loader = RE.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = RE.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
        if train_loader.batch_size != BATCH_SIZE or test_loader.batch_size != BATCH_SIZE:
            raise RuntimeError("Native DataLoader batch size changed")
        result["timings_seconds"]["split_and_pair_preparation"] = time.monotonic() - split_started
        emit("split_ready", dataset=result["dataset"], split=result["split"],
             configuration=result["configuration"])
        print("Native Davis split:", result["split"], flush=True)

        phase = "model_construction"
        result["gpu_allocator"]["before_model_construction"] = allocator_snapshot(torch)
        construct_started = time.monotonic()
        model = RE.net(flags, 32, 4, 8).cuda()
        model.apply(RE.weights_init)
        result["model"] = {
            "constructed": True,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "device": str(next(model.parameters()).device),
        }
        result["timings_seconds"]["model_construction"] = time.monotonic() - construct_started
        result["gpu_allocator"]["after_model_construction"] = allocator_snapshot(torch)
        emit("model_constructed", model=result["model"])
        print("Native model constructed:", result["model"], flush=True)

        phase = "training"
        state = {"phase": "training", "optimizer_updates": 0, "current_batch_size": None,
                 "first_batch_size": None, "components": [], "batch_metrics": []}
        native_adam = RE.optim.Adam
        native_tqdm = RE.tqdm
        native_loss_f = RE.loss_f

        class ObservedAdam(native_adam):
            def step(self, closure=None):
                value = super().step(closure=closure)
                torch.cuda.synchronize()
                state["optimizer_updates"] += 1
                count = state["optimizer_updates"]
                result["training"]["optimizer_updates"] = count
                if count == 1:
                    snapshot = allocator_snapshot(torch)
                    result["gpu_allocator"]["after_first_optimizer_update"] = snapshot
                    emit("optimizer_update", count=count, gpu_allocator=snapshot)
                else:
                    emit("optimizer_update", count=count)
                return value

        def observed_loss_f(*args, **kwargs):
            value = native_loss_f(*args, **kwargs)
            if state["phase"] == "training":
                state["components"].append(scalar(value))
            return value

        class ObservedTqdm(native_tqdm):
            def __iter__(self):
                for batch in super().__iter__():
                    size = int(batch[0].shape[0])
                    state["current_batch_size"] = size
                    if state["first_batch_size"] is None:
                        state["first_batch_size"] = size
                        result["training"]["first_batch_size_observed"] = size
                        emit("train_batch_started", batch_one_based=1, batch_size=size)
                        if size != BATCH_SIZE:
                            raise RuntimeError(f"First native training batch was {size}, not 256")
                    yield batch

            def set_postfix(self, *args, **kwargs):
                if state["phase"] == "training" and "train_loss" in kwargs:
                    if len(state["components"]) != 2:
                        raise RuntimeError("Expected native drug and target VAE loss components")
                    metrics = {
                        "combined_objective": measured_number(kwargs["train_loss"]),
                        "regression_mse": measured_number(kwargs["mse"]),
                        "drug_reconstruction_plus_kl": state["components"][0],
                        "target_reconstruction_plus_kl": state["components"][1],
                        "train_cindex": measured_number(kwargs["train_cindex"]),
                    }
                    state["components"].clear()
                    batch_number = len(state["batch_metrics"]) + 1
                    if state["optimizer_updates"] != batch_number:
                        raise RuntimeError("Native batch and optimizer update counts diverged")
                    is_finite = finite(metrics)
                    record = {"batch_one_based": batch_number,
                              "batch_size": state["current_batch_size"],
                              "optimizer_updates": state["optimizer_updates"],
                              "metrics": metrics, "finite": is_finite}
                    state["batch_metrics"].append(record)
                    result["training"]["completed_batches"] = batch_number
                    result["training"].setdefault("first_batch", record)
                    result["training"]["last_batch"] = record
                    result["training"]["finite"] = result["training"]["finite"] and is_finite
                    emit("train_batch_completed", **record)
                    if not is_finite:
                        raise FloatingPointError("Native training returned NaN or Inf")
                return super().set_postfix(*args, **kwargs)

        # Wrappers observe native return values; they do not replace the loss or optimizer math.
        RE.optim.Adam = ObservedAdam
        RE.tqdm = ObservedTqdm
        RE.loss_f = observed_loss_f
        torch.cuda.reset_peak_memory_stats(0)
        result["gpu_allocator"]["training_start"] = allocator_snapshot(torch)
        train_started = time.monotonic()
        try:
            model = RE.train(train_loader, model, flags, 32, 4, 8, -5)
        finally:
            RE.optim.Adam = native_adam
            RE.tqdm = native_tqdm
            RE.loss_f = native_loss_f
        result["timings_seconds"]["native_training_epoch"] = time.monotonic() - train_started
        if (state["optimizer_updates"] != len(train_loader)
                or len(state["batch_metrics"]) != len(train_loader)
                or state["first_batch_size"] != BATCH_SIZE):
            raise RuntimeError("Native epoch did not complete all planned batch-256 updates")
        result["training"].update({
            "optimizer_updates": state["optimizer_updates"],
            "completed_batches": len(state["batch_metrics"]),
            "epoch_completed": True,
            "first_batch": state["batch_metrics"][0],
            "last_batch": state["batch_metrics"][-1],
            "all_batch_metrics_finite": all(item["finite"] for item in state["batch_metrics"]),
            "batch_metrics_artifact": str(OUTPUT / "events.jsonl"),
            "loss_component_scope": "native loss_f returns reconstruction plus KL per branch; separate terms are not exposed",
            "native_function": "run_experiments.train",
        })
        result["gpu_allocator"]["after_epoch"] = allocator_snapshot(torch)
        emit("epoch_completed", training=result["training"])
        print(f"One native epoch completed: {state['optimizer_updates']} optimizer updates", flush=True)

        phase = "evaluation"
        state["phase"] = "evaluation"
        result["evaluation"]["started"] = True
        emit("evaluation_started")
        eval_started = time.monotonic()
        ci, mse, rm2, auc = RE.test(model, test_loader, flags, 32, 4, 8, -5)
        result["timings_seconds"]["native_evaluation"] = time.monotonic() - eval_started
        metrics = {"CI": scalar(ci), "MSE": scalar(mse),
                   "RM2": scalar(rm2), "AUC": scalar(auc)}
        result["evaluation"] = {
            "started": True, "reached": True, "native_function": "run_experiments.test",
            "test_pairs": len(test_indices), "metrics": metrics,
            "finite": finite(metrics),
            "AUPR": "not returned by this native test function",
        }
        result["gpu_allocator"]["after_evaluation"] = allocator_snapshot(torch)
        emit("native_evaluation_completed", evaluation=result["evaluation"])
        result["finite"] = result["training"]["all_batch_metrics_finite"] and result["evaluation"]["finite"]
        result["status"] = "completed" if result["finite"] else "failed_nonfinite_evaluation"
        result["stopping_reason"] = (
            "one native epoch and evaluation completed" if result["finite"]
            else "native evaluation returned NaN or Inf"
        )
        print("Native evaluation:", result["evaluation"], flush=True)
    except BaseException as error:
        result["status"] = "failed"
        result["stopping_reason"] = phase + " exception"
        if "torch" in locals():
            if isinstance(error, getattr(torch.cuda, "OutOfMemoryError", ())):
                result["failure_class_candidate"] = "resource_failure"
            try:
                if torch.cuda.is_available():
                    result["gpu_allocator"]["at_exception"] = allocator_snapshot(torch)
            except BaseException as snapshot_error:
                result["gpu_allocator"]["exception_snapshot_error"] = repr(snapshot_error)
        result["exception"] = {"phase": phase, "type": type(error).__name__,
                               "message": str(error), "traceback": traceback.format_exc()}
        emit("worker_exception", exception=result["exception"])
        traceback.print_exc()
    finally:
        result["timings_seconds"]["worker_total"] = time.monotonic() - started
        write_json(OUTPUT / "worker_summary.json", result)
    return 0 if result["status"] == "completed" else 1


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
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def parent():
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
        "environment": {}, "GPU": {}, "dataset": {}, "split": {}, "seed": SEED,
        "configuration": {}, "model": {}, "training": {}, "evaluation": {},
        "finite": None, "host_memory": {}, "gpu_allocator": {},
        "system_gpu_memory": {}, "timings_seconds": {}, "source": {},
        "worker_returncode": None, "worker_pid": None,
    }
    write_json(OUTPUT / "summary.json", summary)
    proc = None
    samples = []
    source = None
    try:
        source, source_hashes, data_hashes = discover_source()
        summary["source"] = {
            "kaggle_input": str(source), "source_sha256": source_hashes,
            "davis_data_sha256": data_hashes,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "local_nested_HEAD_at_preparation": "2c172682e502e3d39f6cc6a6d5bca428326e7d0b",
            "input_git_identity_verified": False,
        }
        first_gpu = gpu_memory()
        if "T4" not in first_gpu["name"]:
            raise RuntimeError(f"Expected Tesla T4 at physical GPU 0: {first_gpu['name']}")
        if first_gpu["total_mib"] <= GPU_HEADROOM_MIB:
            raise RuntimeError("GPU memory capacity is too small for the safety boundary")
        gpu_limit = first_gpu["total_mib"] - GPU_HEADROOM_MIB
        if first_gpu["used_mib"] >= gpu_limit:
            raise RuntimeError("GPU 0 already exceeds the memory safety boundary")
        summary["GPU"] = {**first_gpu, "memory_limit_mib": gpu_limit}
        summary["status"] = "running"
        write_json(OUTPUT / "summary.json", summary)

        environment = dict(os.environ)
        environment.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONDONTWRITEBYTECODE": "1",
                            "COVAE_SOURCE": str(source)})
        with (OUTPUT / "training.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker"],
                cwd=OUTPUT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"Worker PID {proc.pid}; source {source}; output {OUTPUT}", flush=True)
            next_gpu_sample = 0.0
            last_gpu = first_gpu
            announced = set()
            with (OUTPUT / "memory.jsonl").open("w", encoding="utf-8") as memory_log:
                while proc.poll() is None:
                    elapsed = time.monotonic() - start
                    rss = process_rss_bytes(proc.pid)
                    if rss is None and proc.poll() is None:
                        # A finished worker can lose VmRSS before poll observes exit.
                        try:
                            proc.wait(timeout=POLL_SECONDS)
                        except subprocess.TimeoutExpired:
                            summary["stopping_reason"] = "worker RSS monitor unavailable"
                    if elapsed >= next_gpu_sample and proc.poll() is None:
                        try:
                            last_gpu = gpu_memory()
                        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
                            summary["stopping_reason"] = "GPU memory monitor failed: " + repr(error)
                        next_gpu_sample = elapsed + POLL_SECONDS
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
                        if kind in {"environment_ready", "split_ready", "model_constructed",
                                    "epoch_completed", "native_evaluation_completed"} and kind not in announced:
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
                    if proc.poll() is not None:
                        break
                    time.sleep(POLL_SECONDS)
            proc.wait()

        parent_stop_reason = summary["stopping_reason"]
        worker_file = OUTPUT / "worker_summary.json"
        if worker_file.exists():
            summary.update(json.loads(worker_file.read_text(encoding="utf-8")))
        else:
            recover_progress(summary, read_events())
        if parent_stop_reason is not None:
            summary["stopping_reason"] = parent_stop_reason
            summary["status"] = "stopped_by_safety_limit"
        elif proc.returncode != 0:
            summary["status"] = "failed"
            summary["stopping_reason"] = summary.get("stopping_reason") or f"worker exit {proc.returncode}"
        elif summary.get("status") != "completed":
            summary["status"] = "failed"
            summary["stopping_reason"] = "worker exited without completed evaluation evidence"
        summary["worker_returncode"] = proc.returncode
        summary["worker_pid"] = proc.pid
    except BaseException as error:
        summary["status"] = "failed"
        summary["stopping_reason"] = summary.get("stopping_reason") or "parent exception"
        summary["exception"] = {"phase": "parent", "type": type(error).__name__,
                                "message": str(error), "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        if proc is not None and proc.poll() is None:
            try:
                terminate_group(proc)
            except BaseException as error:
                summary["termination_error"] = repr(error)
        if source is not None:
            try:
                summary["source"]["unchanged_after_run"] = (
                    all(sha256(source / name) == digest for name, digest in SOURCE_HASHES.items())
                    and all(sha256(source / "data/davis" / name) == digest
                            for name, digest in DATA_HASHES.items())
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
            summary["system_gpu_memory"] = {
                "first_used_mib": gpu_values[0], "max_observed_used_mib": max(gpu_values),
                "last_used_mib": gpu_values[-1],
                "limit_mib": summary.get("GPU", {}).get("memory_limit_mib"),
                "scope": "physical GPU 0, including any other processes",
            }
            boundaries = {}
            for event in read_events():
                if event["kind"] in {"environment_ready", "split_ready", "model_constructed",
                                     "optimizer_update", "epoch_completed", "native_evaluation_completed"}:
                    if event["kind"] == "optimizer_update" and event["count"] != 1:
                        continue
                    nearest = min(samples, key=lambda item: abs(item["epoch_seconds"] - event["epoch_seconds"]))
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
        raise SystemExit("Usage: python stage08_5b_covae_kaggle.py")
    sys.exit(parent())
