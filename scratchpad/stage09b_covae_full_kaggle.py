#!/usr/bin/env python3
"""Stage 09B: one seeded native Co-VAE fit, then one epoch-100 test pass.

Kaggle only. No baseline/input files are written. Use the companion four-cell
notebook, which pins this runner's hash. --preflight checks inputs/environment
without loading the baseline, constructing a model, or training.
"""

import csv
import hashlib
import importlib.metadata
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


INPUT_ROOT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working/stage09b_full')
EPOCHS = 100
BATCH_SIZE = 256
SEED = 1000
FILTERS, DRUG_KERNEL, TARGET_KERNEL, LAMBDA_EXPONENT = 32, 5, 7, -5
LEARNING_RATE = 0.001  # Checked against native Adam; there is no CLI lr argument.
HOST_RSS_LIMIT = 12 * 1024**3
GPU_HEADROOM_MIB = 512
POLL_SECONDS = 1.0
MILESTONES = {1, 10, 25, 50, 75, 100}
SOURCE_HASHES = {
    'run_experiments.py': '39546fa554e628bf4f30dc492d152ed1235fbeedfc83c713218dd0d40d108043',
    'model.py': 'a6dc416b3c4257d67e226fb1888d6236c6203f483d943415c5c1a8879e32e8fc',
    'arguments.py': '1b9416dfb212a9c92bda76ec5eea18561530055b9b442d6c0c07e4a483fc388a',
    'datahelper.py': 'eb7f1245531be9d5cf9e8099e5063125eac0c5dc3b6dbfe5fda9947f797b917c',
    'emetrics.py': '031696e05fb3c40ea3607ec76c76dcc5079d888d8a7ac2c60658e0cd041fc7f9',
}
DATA_HASHES = {
    'ligands_iso.txt': '9789900f1e723226187215235b379998e5735a3ffec0e9487bad27c1c6a64161',
    'proteins.txt': '187cedeee10cd3b58af915a2861a5ed0c4ff42a2d728f4dea5e0b47e3d75b291',
    'drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt':
        '31b3f3efea9afe53239fa9fe735aedd6195586b99261319917b55b1d1c9e8392',
}
REQUIRED = [
    'environment.json', 'protocol.json', 'dataset_summary.json', 'split_summary.json',
    'split_membership.json', 'history.csv', 'history.jsonl', 'live_status.json',
    'resource_history.csv', 'milestones.jsonl', 'final_summary.json',
    'test_predictions.csv', 'artifact_hashes.json', 'final_model.state_dict.pt',
    'model_summary.txt', 'training.log', 'events.jsonl', 'worker_summary.json',
]

# Set before any torch import. PYTHONHASHSEED must also be set at process launch.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def append_json(name, value):
    with (OUTPUT / name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def index_hash(indices):
    return hashlib.sha256(json.dumps(
        [int(i) for i in indices], separators=(',', ':')).encode('ascii')).hexdigest()


def verify_source(source):
    code = {name: sha256(source / name) for name in SOURCE_HASHES}
    data = {name: sha256(source / 'data/davis' / name) for name in DATA_HASHES}
    if code != SOURCE_HASHES or data != DATA_HASHES:
        raise RuntimeError('Original source/Davis hash mismatch; do not auto-repair')
    return {'source_sha256': code, 'dataset_sha256': data}


def verified_inputs():
    if platform.system() != 'Linux' or not INPUT_ROOT.is_dir():
        raise RuntimeError('This execution entry point is for Kaggle Linux only')
    expected = os.environ.get('STAGE09B_EXPECTED_RUNNER_SHA256')
    actual = sha256(Path(__file__).resolve())
    if not expected or actual != expected:
        raise RuntimeError('Missing/mismatched pinned runner SHA256; use the notebook')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Launch with PYTHONHASHSEED=0; setting it after startup is insufficient')
    candidates = [p.parent for p in INPUT_ROOT.rglob('run_experiments.py')
                  if all((p.parent / f).is_file() for f in SOURCE_HASHES)
                  and all((p.parent / 'data/davis' / f).is_file() for f in DATA_HASHES)]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one Co-VAE input; found {candidates}')
    source = candidates[0].resolve()
    return source, {
        **verify_source(source), 'runner_sha256': actual, 'input_path': str(source),
        'local_nested_HEAD_at_preparation': '2c172682e502e3d39f6cc6a6d5bca428326e7d0b',
        'input_git_identity_verified': False,
    }


def physical_gpu():
    result = subprocess.run(
        ['nvidia-smi', '-i', '0', '--query-gpu=name,memory.total,memory.used',
         '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=5)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError('Expected one physical GPU 0 record')
    name, total, used = [s.strip() for s in lines[0].split(',')]
    if 'T4' not in name:
        raise RuntimeError(f'Expected Tesla T4; got {name}')
    return {'name': name, 'total_mib': int(total), 'used_mib': int(used),
            'safety_limit_mib': int(total) - GPU_HEADROOM_MIB}


def environment_info():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one logical CUDA GPU is required')
    props = torch.cuda.get_device_properties(0)
    if 'T4' not in props.name:
        raise RuntimeError(f'Expected Tesla T4; got {props.name}')
    info = {
        'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
        'pytorch': torch.__version__, 'pytorch_cuda': torch.version.cuda,
        'cuda_available': True, 'logical_gpu_count': torch.cuda.device_count(),
        'gpu_name': props.name, 'gpu_total_bytes': props.total_memory,
        'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
        'PYTHONHASHSEED_at_process_launch': os.environ.get('PYTHONHASHSEED'),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
    }
    for name in ('numpy', 'pandas', 'scikit-learn', 'matplotlib', 'tqdm'):
        info[name] = importlib.metadata.version(name)
    return info


def rss_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def cuda_snapshot(torch):
    return {key: int(function(0)) for key, function in {
        'cuda_allocated_bytes': torch.cuda.memory_allocated,
        'cuda_reserved_bytes': torch.cuda.memory_reserved,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved,
    }.items()}


def finite_scalar(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError('NaN/Inf in observed native value')
    return value


def native_flags(RE, source):
    args = ['run_experiments.py', '--dataset_path', str(source / 'data/davis') + '/',
            '--problem_type', '2', '--is_log', '0', '--num_windows', '32',
            '--smi_window_lengths', '5', '--seq_window_lengths', '7', '--lamda', '-5',
            '--batch_size', '256', '--num_epoch', '100', '--max_smi_len', '85',
            '--max_seq_len', '1200', '--log_dir', str(OUTPUT)]
    saved = sys.argv
    try:
        sys.argv = args
        flags = RE.argparser()
    finally:
        sys.argv = saved
    expected = {'problem_type': 2, 'is_log': 0, 'num_windows': [32],
                'smi_window_lengths': [5], 'seq_window_lengths': [7], 'lamda': [-5],
                'batch_size': BATCH_SIZE, 'num_epoch': EPOCHS,
                'max_smi_len': 85, 'max_seq_len': 1200}
    if any(getattr(flags, key) != value for key, value in expected.items()):
        raise RuntimeError('Native parser did not resolve the fixed protocol')
    return flags, args


def protocol_info(flags, args, source_identity):
    return {
        'claim': 'representative paper-grid full-training reference',
        'dataset': 'Davis', 'setting': 'new-drug', 'flags': vars(flags), 'argv': args,
        'source_identity': source_identity,
        'configuration_rationale': '32 fixed filters; 5/7/-5 belong to the paper grid; winning Davis configuration undisclosed',
        'model': 'RE.net(flags,32,5,7).cuda(); model.apply(RE.weights_init)',
        'loss': 'MSE + 10**(-5) * (drug_reconstruction_plus_KL + 85/1200 * target_reconstruction_plus_KL)',
        'loss_f': 'mean(sequence-summed cross entropy + diagonal Gaussian KL); separate terms not exposed',
        'optimizer': 'native Adam recreated by RE.train every epoch; default lr checked == 0.001',
        'epoch_selection': 'fixed epoch 100; no early stopping or selection by validation/test/paper numbers',
        'seeds': {'python': SEED, 'numpy': SEED, 'torch_cpu': SEED,
                  'torch_cuda': SEED, 'split': SEED, 'python_hash_at_launch': 0},
        'determinism': 'seeded; bitwise GPU determinism not claimed; no backend flags changed',
        'native_test': 'one RE.test call; CI, MSE, RM2, ROC AUC at pKd > 7; no native AUPR return',
        'external_metrics': 'MSE/MAE and chunked full-vector check of native CI estimator from saved CSV',
        'compatibility': 'temporary external random.sample set-to-tuple shim only during native split construction',
        'controls': 'one split/fit, full outer training pool, no native grid/test-informed driver or repeated-test checkpoints',
        'heldout_access': 'membership metadata allowed; no held-out tensors/loader until completion gate; native parser loads full Y',
        'evaluation_stochasticity': 'native VAE reparameterization samples CUDA noise even under eval/no_grad',
        'safety': {'host_rss_limit_bytes': HOST_RSS_LIMIT, 'gpu_headroom_mib': GPU_HEADROOM_MIB,
                   'wall_time_kill': None, 'poll_seconds': POLL_SECONDS},
    }


class HeldoutGate:
    """Defer tensor construction and allow one traversal after complete training."""

    def __init__(self, factory, progress, count):
        self.factory, self.progress, self.count = factory, progress, count
        self.opened = False

    def require_complete(self):
        p = self.progress
        if (p['completed_epochs'] != EPOCHS
                or p['optimizer_updates'] != EPOCHS * p['batches_per_epoch']
                or p['optimizer_constructions'] != EPOCHS
                or not p['final_state_saved']):
            raise RuntimeError('Held-out access before the complete epoch-100 state')

    def open_loader(self):
        self.require_complete()
        if self.opened:
            raise RuntimeError('Held-out loader already opened; no repeated evaluation')
        self.opened = True
        loader = self.factory()
        gate = self

        class OnceLoader:
            def __iter__(self):
                gate.require_complete()
                if gate.progress['test_calls'] != 1 or gate.progress['test_passes'] != 0:
                    raise RuntimeError('Test traversal outside the single final native call')
                gate.progress['test_passes'] += 1
                for batch in loader:
                    gate.progress['test_targets'].extend(
                        batch[2].detach().cpu().reshape(-1).tolist())
                    yield batch

            def __len__(self):
                return len(loader)

        return OnceLoader()


def build_training_inputs(RE, np, source, flags, progress):
    import random
    dataset = RE.DataSet(fpath=flags.dataset_path, setting_no=2,
                         seqlen=1200, smilen=85, need_shuffle=False)
    flags.charseqset_size, flags.charsmiset_size = dataset.charseqset_size, dataset.charsmiset_size
    XD, XT, Y = [np.asarray(x) for x in dataset.parse_data(flags)]
    flags.drug_count, flags.target_count = len(XD), len(XT)
    rows, cols = np.where(~np.isnan(Y))
    drugs = json.loads((source / 'data/davis/ligands_iso.txt').read_text())
    proteins = json.loads((source / 'data/davis/proteins.txt').read_text())
    drug_ids, target_ids = list(drugs), list(proteins)
    if Y.shape != (len(XD), len(XT)) or len(drug_ids) != len(XD) or len(target_ids) != len(XT):
        raise RuntimeError('Dataset/entity ordering dimensions mismatch')
    sequence_groups = {}
    for target_id, sequence in proteins.items():
        sequence_groups.setdefault(sequence, []).append(target_id)
    write_json(OUTPUT / 'dataset_summary.json', {
        'name': 'Davis', 'drug_count': len(XD), 'target_count': len(XT),
        'observed_pairs': len(rows), 'XD_shape': list(XD.shape), 'XT_shape': list(XT.shape),
        'Y_shape': list(Y.shape), 'unique_protein_sequences': len(set(proteins.values())),
        'duplicate_protein_sequence_groups': sum(len(group) > 1 for group in sequence_groups.values()),
        'targets_in_duplicate_sequence_groups': sum(len(group) for group in sequence_groups.values() if len(group) > 1),
        'label_transform': '-log10(Kd/1e9), native Davis loader',
        'dataset_sha256': DATA_HASHES,
    })
    native_sample = random.sample
    def sample_compat(population, k):
        return native_sample(tuple(population) if isinstance(population, set) else population, k)
    random.seed(SEED)
    RE.random.sample = sample_compat
    try:
        folds = RE.get_drugwise_folds(rows, cols, flags.drug_count, 6)
    finally:
        RE.random.sample = native_sample
    if len(folds) != 6:
        raise RuntimeError('Expected the native six-way drug split')
    train_indices = [int(i) for fold in folds[:5] for i in fold]
    test_indices = [int(i) for i in folds[5]]
    flat = train_indices + test_indices
    if (not train_indices or not test_indices or len(flat) != len(rows)
            or len(set(flat)) != len(rows) or set(flat) != set(range(len(rows)))):
        raise RuntimeError('Fold coverage, duplication, or index validity failure')
    train_drugs, test_drugs = set(rows[train_indices].tolist()), set(rows[test_indices].tolist())
    train_targets, test_targets = set(cols[train_indices].tolist()), set(cols[test_indices].tolist())
    drug_overlap = train_drugs & test_drugs
    if drug_overlap:
        raise RuntimeError('Train/test drug IDs overlap in new-drug setting')
    sequences = list(proteins.values())
    train_sequences, test_sequences = ({sequences[i] for i in train_targets},
                                     {sequences[i] for i in test_targets})
    split = {
        'native_function': 'RE.get_drugwise_folds -> RE.get_random_folds',
        'seed': SEED, 'fold_pair_counts': [len(f) for f in folds],
        'train_folds': [0, 1, 2, 3, 4], 'test_fold': 5,
        'train_pairs': len(train_indices), 'test_pairs': len(test_indices),
        'unique_train_drugs': len(train_drugs), 'unique_test_drugs': len(test_drugs),
        'train_drug_ids': [drug_ids[i] for i in sorted(train_drugs)],
        'test_drug_ids': [drug_ids[i] for i in sorted(test_drugs)],
        'drug_id_overlap_count': len(drug_overlap), 'train_test_drug_ids_overlap': bool(drug_overlap),
        'target_id_overlap_count': len(train_targets & test_targets),
        'target_sequence_overlap_count': len(train_sequences & test_sequences),
        'shared_duplicate_sequence_groups': sum(len(sequence_groups[s]) > 1 for s in train_sequences & test_sequences),
        'sequence_overlap_interpretation': 'Shared targets/sequences are expected in new-drug; this is not a cold-target claim',
        'train_indices_sha256': index_hash(train_indices), 'test_indices_sha256': index_hash(test_indices),
        'fold_indices_sha256': [index_hash(f) for f in folds],
        'train_batches': math.ceil(len(train_indices) / BATCH_SIZE),
        'test_batches': math.ceil(len(test_indices) / BATCH_SIZE),
        'set_to_tuple_shim': True, 'cross_Python_reference_hash_verified': False,
        'split_identity_claim': 'native seed-1000 split in the recorded Python runtime; cross-version identity not assumed',
    }
    write_json(OUTPUT / 'split_summary.json', split)
    write_json(OUTPUT / 'split_membership.json', {
        'fold_observed_pair_indices': [[int(i) for i in f] for f in folds],
        'observed_rows_sha256': index_hash(rows), 'observed_cols_sha256': index_hash(cols),
        'drug_ids_in_loader_order': drug_ids, 'target_ids_in_loader_order': target_ids,
    })
    train_data = RE.prepare_interaction_pairs(XD, XT, Y, rows[train_indices], cols[train_indices])
    train_loader = RE.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    progress['batches_per_epoch'] = len(train_loader)
    def heldout_factory():
        data = RE.prepare_interaction_pairs(XD, XT, Y, rows[test_indices], cols[test_indices])
        return RE.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=False)
    gate = HeldoutGate(heldout_factory, progress, len(test_indices))
    identities = [(i, int(rows[i]), int(cols[i]), drug_ids[rows[i]], target_ids[cols[i]])
                  for i in test_indices]
    return train_loader, gate, split, identities


def external_vector_checks(np, prediction_file):
    """Read saved vectors; no model, loader, training state, or RNG access."""
    with prediction_file.open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    y = np.array([float(r['target']) for r in records], dtype=np.float64)
    p = np.array([float(r['prediction']) for r in records], dtype=np.float64)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise FloatingPointError('Non-finite/empty saved prediction vectors')
    numerator, denominator = 0.0, 0
    positions = np.arange(len(y))
    for start in range(0, len(y), 128):
        end = min(start + 128, len(y))
        # Preserve emetrics.py:4-18, including its lower-triangle/order convention.
        eligible = (y[start:end, None] > y[None, :]) & (positions[None, :] <= positions[start:end, None])
        scores = (p[start:end, None] > p[None, :]) + 0.5 * (p[start:end, None] == p[None, :])
        numerator += float(np.sum(scores * eligible))
        denominator += int(np.sum(eligible))
    def stats(vector):
        return {'count': len(vector), 'finite_count': int(np.isfinite(vector).sum()),
                'min': float(vector.min()), 'max': float(vector.max()),
                'mean': float(vector.mean()), 'std_population': float(vector.std())}
    return {'external_mse': float(np.mean((p - y)**2)),
            'external_mae': float(np.mean(np.abs(p - y))),
            'external_global_ci_native_estimator': numerator / denominator if denominator else 0.0,
            'ci_scope': 'full saved vector; chunked audited native lower-triangle estimator, not a newly standardized benchmark',
            'prediction_stats': stats(p), 'target_stats': stats(y)}


def worker():
    start = time.monotonic()
    result = {'status': 'running', 'claim': 'representative paper-grid full-training reference',
              'phase': 'environment', 'completed_epochs': 0, 'optimizer_updates': 0,
              'test_calls': 0, 'test_passes': 0, 'exception': None}
    progress = {'completed_epochs': 0, 'optimizer_updates': 0, 'optimizer_constructions': 0,
                'batches_per_epoch': 0, 'final_state_saved': False, 'test_calls': 0,
                'test_passes': 0, 'test_targets': [], 'test_predictions': []}
    live = {'status': 'running', 'phase': 'environment', 'epoch': 0,
            'planned_epochs': EPOCHS, 'warnings': []}
    def publish(**fields):
        live.update(fields, elapsed_seconds=time.monotonic() - start, timestamp=time.time())
        write_json(OUTPUT / 'live_status.json', live)
    try:
        source, identity = verified_inputs()
        result['source'] = identity
        info = environment_info()
        write_json(OUTPUT / 'environment.json', info)
        import random
        import numpy as np
        import torch
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        os.chdir(OUTPUT)
        sys.path.insert(0, str(source))
        import run_experiments as RE
        flags, args = native_flags(RE, source)
        result['phase'] = 'dataset_and_split'
        publish(phase=result['phase'])
        train_loader, gate, split, identities = build_training_inputs(RE, np, source, flags, progress)
        result['split'] = split
        write_json(OUTPUT / 'protocol.json', protocol_info(flags, args, identity))
        result['phase'] = 'construction'
        model = RE.net(flags, FILTERS, DRUG_KERNEL, TARGET_KERNEL).cuda()
        model.apply(RE.weights_init)
        result['model'] = {'parameters': sum(p.numel() for p in model.parameters()),
                           'device': str(next(model.parameters()).device)}
        (OUTPUT / 'model_summary.txt').write_text(str(model) + '\n' + json.dumps(result['model']), encoding='utf-8')
        torch.cuda.reset_peak_memory_stats(0)
        native_adam, native_tqdm, native_loss, native_test = RE.optim.Adam, RE.tqdm, RE.loss_f, RE.test
        state = {'components': [], 'batch_size': 0, 'epoch_batches': 0, 'samples': 0, 'sums': {}}

        class ObservedAdam(native_adam):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                progress['optimizer_constructions'] += 1
                if any(group['lr'] != LEARNING_RATE for group in self.param_groups):
                    raise RuntimeError('Native Adam default learning rate differs from 0.001')
                result.setdefault('native_adam', {key: value for key, value in self.defaults.items()
                                                if isinstance(value, (int, float, bool, tuple, type(None)))})

            def step(self, closure=None):
                value = super().step(closure=closure)
                progress['optimizer_updates'] += 1
                return value

        def observed_loss(*args, **kwargs):
            value = native_loss(*args, **kwargs)
            state['components'].append(finite_scalar(value))
            return value

        class ObservedTqdm(native_tqdm):
            def __iter__(self):
                for batch in super().__iter__():
                    state['batch_size'] = int(batch[0].shape[0])
                    yield batch

            def set_postfix(self, *args, **kwargs):
                if 'train_loss' in kwargs:
                    if len(state['components']) != 2:
                        raise RuntimeError('Expected exactly two native branch loss_f calls')
                    metrics = {'train_total_loss': finite_scalar(kwargs['train_loss']),
                               'train_regression_mse': finite_scalar(kwargs['mse']),
                               'drug_reconstruction_plus_kl': state['components'][0],
                               'target_reconstruction_plus_kl': state['components'][1],
                               'native_train_ci_batch_weighted': finite_scalar(kwargs['train_cindex'])}
                    state['components'].clear()
                    state['epoch_batches'] += 1
                    state['samples'] += state['batch_size']
                    for key, value in metrics.items():
                        state['sums'][key] = state['sums'].get(key, 0.0) + value * state['batch_size']
                    if state['epoch_batches'] % 10 == 0:
                        publish(batch=state['epoch_batches'], batches_per_epoch=len(train_loader),
                                optimizer_updates=progress['optimizer_updates'],
                                latest_batch_metrics=metrics, **cuda_snapshot(torch),
                                host_rss_bytes=rss_bytes(os.getpid()))
                return super().set_postfix(*args, **kwargs)

        def gated_test(*args, **kwargs):
            gate.require_complete()
            if progress['test_calls'] != 0:
                raise RuntimeError('Repeated held-out native evaluation forbidden')
            progress['test_calls'] += 1
            return native_test(*args, **kwargs)

        RE.test = gated_test  # Guard even an accidental call during training.
        RE.optim.Adam, RE.tqdm, RE.loss_f = ObservedAdam, ObservedTqdm, observed_loss
        result['phase'] = 'training'
        try:
            for epoch in range(1, EPOCHS + 1):
                publish(phase='training', epoch=epoch, batch=0)
                epoch_start = time.monotonic()
                state.update(components=[], epoch_batches=0, samples=0, sums={})
                updates_before = progress['optimizer_updates']
                model = RE.train(train_loader, model, flags, FILTERS, DRUG_KERNEL, TARGET_KERNEL, LAMBDA_EXPONENT)
                torch.cuda.synchronize()
                if (state['epoch_batches'] != len(train_loader) or state['samples'] != split['train_pairs']
                        or state['components'] or progress['optimizer_updates'] - updates_before != len(train_loader)
                        or progress['optimizer_constructions'] != epoch):
                    raise RuntimeError('Incomplete native epoch or unexpected optimizer recreation')
                progress['completed_epochs'] = epoch
                row = {'epoch': epoch, **{k: v / state['samples'] for k, v in state['sums'].items()},
                       'training_samples': state['samples'], 'optimizer_updates': progress['optimizer_updates'],
                       'optimizer_constructions': progress['optimizer_constructions'],
                       'epoch_seconds': time.monotonic() - epoch_start,
                       'elapsed_seconds': time.monotonic() - start, **cuda_snapshot(torch),
                       'host_rss_bytes': rss_bytes(os.getpid())}
                with (OUTPUT / 'history.csv').open('a', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if epoch == 1:
                        writer.writeheader()
                    writer.writerow(row)
                append_json('history.jsonl', row)
                append_json('events.jsonl', {'kind': 'epoch_completed', **row})
                if epoch in MILESTONES:
                    append_json('milestones.jsonl', row)
                result.update(completed_epochs=epoch, optimizer_updates=progress['optimizer_updates'],
                              last_epoch=row, optimizer_constructions=progress['optimizer_constructions'])
                publish(completed_epochs=epoch, latest_epoch_metrics=row,
                        optimizer_updates=progress['optimizer_updates'],
                        optimizer_constructions=progress['optimizer_constructions'],
                        estimated_training_remaining_seconds=(EPOCHS - epoch) * row['epoch_seconds'])
        finally:
            RE.optim.Adam, RE.tqdm, RE.loss_f = native_adam, native_tqdm, native_loss
        checkpoint = OUTPUT / 'final_model.state_dict.pt'
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(model.state_dict(), temporary)
        temporary.replace(checkpoint)
        progress['final_state_saved'] = True
        result['reference_epoch'] = EPOCHS
        result['phase'] = 'heldout_evaluation'
        publish(phase=result['phase'], reference_epoch=EPOCHS)
        test_loader = gate.open_loader()

        def capture_predictions(module, inputs, outputs):
            if module.training or torch.is_grad_enabled() or result['phase'] != 'heldout_evaluation':
                raise RuntimeError('Prediction hook outside native final evaluation')
            progress['test_predictions'].extend(outputs[0].detach().cpu().reshape(-1).tolist())
            # Return None: never substitute outputs or retain their autograd graph.

        hook = model.register_forward_hook(capture_predictions)
        evaluation_start = time.monotonic()
        append_json('events.jsonl', {'kind': 'heldout_evaluation_started', 'epoch': EPOCHS, 'count': 1})
        try:
            metrics = RE.test(model, test_loader, flags, FILTERS, DRUG_KERNEL, TARGET_KERNEL, LAMBDA_EXPONENT)
        finally:
            hook.remove()
            RE.test = native_test
            result.update(test_calls=progress['test_calls'], test_passes=progress['test_passes'])
            y, p = progress['test_targets'], progress['test_predictions']
            if len(y) == len(p) == gate.count and np.isfinite(y).all() and np.isfinite(p).all():
                with (OUTPUT / 'test_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['observed_pair_index', 'drug_index', 'target_index', 'drug_id', 'target_id', 'target', 'prediction'])
                    for identity_row, target, prediction in zip(identities, y, p):
                        writer.writerow([*identity_row, format(target, '.17g'), format(prediction, '.17g')])
            else:
                raise RuntimeError('Held-out count mismatch or non-finite prediction/target')
        if progress['test_calls'] != 1 or progress['test_passes'] != 1:
            raise RuntimeError('Exactly one native call and prediction traversal required')
        if len(metrics) != 4:
            raise RuntimeError('Native test did not return the four inspected metrics')
        native_metrics = {name: finite_scalar(value) for name, value in zip(
            ('CI', 'MSE', 'RM2', 'ROC_AUC_pKd_gt_7'), metrics)}
        external = external_vector_checks(np, OUTPUT / 'test_predictions.csv')
        if (not math.isclose(native_metrics['MSE'], external['external_mse'], rel_tol=1e-5, abs_tol=1e-7)
                or not math.isclose(native_metrics['CI'], external['external_global_ci_native_estimator'], rel_tol=1e-6, abs_tol=1e-7)):
            raise RuntimeError('Native versus saved-vector metric consistency failure')
        result.update(status='completed', stopping_reason='100 full epochs and one native held-out evaluation completed',
                      native_test_metrics=native_metrics, external_checks=external,
                      test_count=gate.count, prediction_count=len(p), finite_prediction_count=int(np.isfinite(p).sum()),
                      evaluation_seconds=time.monotonic() - evaluation_start, cuda_memory=cuda_snapshot(torch),
                      numerical_paper_reproduction='unverified; selected configuration and aggregation protocol unrecoverable')
        append_json('events.jsonl', {'kind': 'heldout_evaluation_completed', 'count': len(p), 'native_metrics': native_metrics})
    except BaseException as error:
        result.update(status='failed', stopping_reason=f"{result['phase']} failure",
                      exception={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        result['worker_seconds'] = time.monotonic() - start
        result.update(completed_epochs=progress['completed_epochs'], optimizer_updates=progress['optimizer_updates'],
                      optimizer_constructions=progress['optimizer_constructions'],
                      test_calls=progress['test_calls'], test_passes=progress['test_passes'])
        write_json(OUTPUT / 'worker_summary.json', result)
        publish(status=result['status'], phase='worker_finished', stopping_reason=result.get('stopping_reason'))
    return 0 if result['status'] == 'completed' else 1


def terminate_worker(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except ProcessLookupError:
            proc.wait(timeout=5)


def parent():
    if platform.system() != 'Linux' or not INPUT_ROOT.is_dir():
        print('Kaggle Linux execution only; no output created.', flush=True)
        return 2
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        print('Output directory is nonempty; preserve it. No retry or overwrite performed.', flush=True)
        return 2
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        lock = os.open(OUTPUT / 'experiment_started.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        print('Output already used; inspect it. No retry or overwrite performed.', flush=True)
        return 2
    with os.fdopen(lock, 'w') as stream:
        stream.write(f'started_epoch_seconds={time.time()}\n')
    start, proc, source = time.monotonic(), None, None
    summary = {'status': 'preparing', 'stopping_reason': None, 'exception': None,
               'completed_epochs': 0, 'optimizer_updates': 0, 'test_calls': 0, 'test_passes': 0}
    max_rss, max_gpu, samples = 0, 0, 0
    try:
        source, identity = verified_inputs()
        summary['source'] = identity
        gpu = physical_gpu()
        if gpu['used_mib'] >= gpu['safety_limit_mib']:
            raise RuntimeError('Physical GPU 0 already at the safety threshold')
        summary['gpu'] = gpu
        summary['status'] = 'running'
        write_json(OUTPUT / 'live_status.json', {'status': 'running', 'phase': 'worker_start', 'epoch': 0})
        for name in ('events.jsonl', 'milestones.jsonl'):
            (OUTPUT / name).touch(exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONHASHSEED='0', STAGE09B_PARENT_PID=str(os.getpid()))
        with (OUTPUT / 'training.log').open('w', encoding='utf-8') as log, (OUTPUT / 'resource_history.csv').open('w', newline='', encoding='utf-8') as resource:
            fields = ['timestamp', 'elapsed_seconds', 'worker_rss_bytes', 'gpu_used_mib', 'gpu_total_mib']
            writer = csv.DictWriter(resource, fieldnames=fields)
            writer.writeheader()
            resource.flush()
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker'],
                                    env=env, cwd=OUTPUT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            print(f'Worker PID {proc.pid}; source {source}; output {OUTPUT}', flush=True)
            missing_rss, failed_gpu, announced_epoch = 0, 0, 0
            while proc.poll() is None:
                rss = rss_bytes(proc.pid)
                if rss is None:
                    try:
                        proc.wait(timeout=POLL_SECONDS)
                        break  # Exit race: a finished worker may have no VmRSS.
                    except subprocess.TimeoutExpired:
                        missing_rss += 1
                else:
                    missing_rss = 0
                    max_rss = max(max_rss, rss)
                try:
                    gpu = physical_gpu()
                    failed_gpu = 0
                    max_gpu = max(max_gpu, gpu['used_mib'])
                except (OSError, subprocess.SubprocessError, RuntimeError, ValueError):
                    failed_gpu += 1
                row = {'timestamp': time.time(), 'elapsed_seconds': time.monotonic() - start,
                       'worker_rss_bytes': rss, 'gpu_used_mib': gpu['used_mib'], 'gpu_total_mib': gpu['total_mib']}
                writer.writerow(row)
                resource.flush()
                samples += 1
                try:
                    live = json.loads((OUTPUT / 'live_status.json').read_text())
                    completed = live.get('completed_epochs', 0)
                    if completed > announced_epoch:
                        print(f"Epoch {completed}/100 at {row['elapsed_seconds']:.1f}s; loss={live['latest_epoch_metrics']['train_total_loss']:.6f}", flush=True)
                        announced_epoch = completed
                except (OSError, KeyError, json.JSONDecodeError):
                    pass
                reason = ('Host RSS exceeded 12 GiB' if rss is not None and rss > HOST_RSS_LIMIT
                          else 'Physical GPU 0 reached capacity minus 512 MiB' if gpu['used_mib'] >= gpu['safety_limit_mib']
                          else 'Host RSS monitor unavailable for three live polls' if missing_rss >= 3
                          else 'GPU monitor unavailable for three live polls' if failed_gpu >= 3 else None)
                if reason and proc.poll() is None:
                    summary.update(status='stopped_by_safety_limit', stopping_reason=reason)
                    terminate_worker(proc)
                    break
                time.sleep(POLL_SECONDS)
            proc.wait()
        parent_stop = summary['stopping_reason']
        if (OUTPUT / 'worker_summary.json').is_file():
            summary.update(json.loads((OUTPUT / 'worker_summary.json').read_text()))
        else:
            # Retain observations from the last complete epoch, not assumed progress.
            live = json.loads((OUTPUT / 'live_status.json').read_text())
            summary['completed_epochs'] = live.get('completed_epochs', 0)
            summary['optimizer_updates'] = live.get('optimizer_updates', 0)
            summary['optimizer_constructions'] = live.get('optimizer_constructions', 0)
            summary['last_epoch'] = live.get('latest_epoch_metrics')
            summary['progress_recovered_from_live_status'] = True
        summary['worker_returncode'] = proc.returncode
        if parent_stop:
            summary.update(status='stopped_by_safety_limit', stopping_reason=parent_stop)
        elif proc.returncode != 0 or summary['status'] != 'completed':
            summary.update(status='failed', stopping_reason=summary.get('stopping_reason') or 'Worker did not complete')
    except BaseException as error:
        summary.update(status='failed', stopping_reason='Supervisor failure',
                       exception={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        traceback.print_exc()
    finally:
        if proc is not None:
            terminate_worker(proc)
        if source is not None:
            try:
                verify_source(source)
                summary['immutable_inputs_unchanged'] = True
            except (OSError, RuntimeError) as error:
                summary.update(status='failed', stopping_reason='Post-run source/data integrity failure',
                               immutable_inputs_unchanged=False, integrity_error=str(error))
        summary.update(total_parent_seconds=time.monotonic() - start,
                       resources={'sample_count': samples, 'peak_host_rss_bytes': max_rss,
                                  'sampled_gpu_peak_mib': max_gpu, 'host_limit_bytes': HOST_RSS_LIMIT,
                                  'gpu_limit_mib': summary.get('gpu', {}).get('safety_limit_mib'),
                                  'gpu_scope': 'physical GPU 0, including other processes', 'wall_time_kill': None})
        missing = [name for name in REQUIRED if name not in {'final_summary.json', 'artifact_hashes.json'}
                   and not (OUTPUT / name).is_file()]
        if summary['status'] == 'completed' and (missing or summary.get('completed_epochs') != EPOCHS
                or summary.get('test_calls') != 1 or summary.get('test_passes') != 1):
            summary.update(status='invalid', stopping_reason='Required evidence/completion invariant failure')
        summary['missing_artifacts'] = missing
        summary['artifacts_directory'] = str(OUTPUT)
        write_json(OUTPUT / 'live_status.json', {**summary, 'phase': 'finished'})
        write_json(OUTPUT / 'final_summary.json', summary)
        # Hash final_summary only after its final write. A manifest cannot hash itself.
        hashes = {path.name: sha256(path) for path in sorted(OUTPUT.iterdir())
                  if path.is_file() and path.name != 'artifact_hashes.json'}
        write_json(OUTPUT / 'artifact_hashes.json', {
            'algorithm': 'SHA256', 'files': hashes, 'self_excluded': 'artifact_hashes.json',
            'immutable_inputs': summary.get('source'), 'missing_required': missing})
        print(f"Final status: {summary['status']}; reason: {summary.get('stopping_reason')}", flush=True)
        print(f"Summary: {OUTPUT / 'final_summary.json'}", flush=True)
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        source, identity = verified_inputs()
        print(json.dumps({'runner': str(Path(__file__).resolve()), 'source': str(source),
                          'identity': identity, 'environment': environment_info(),
                          'physical_gpu': physical_gpu(), 'training_started': False}, indent=2))
    elif sys.argv[1:] == ['--worker']:
        if os.environ.get('STAGE09B_PARENT_PID') != str(os.getppid()):
            raise SystemExit('Worker must be launched by this runner supervisor')
        sys.exit(worker())
    elif sys.argv[1:]:
        raise SystemExit('Usage: python -B stage09b_covae_full_kaggle.py [--preflight]')
    else:
        sys.exit(parent())
