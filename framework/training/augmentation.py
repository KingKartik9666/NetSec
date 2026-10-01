"""Experimental training-only conditional GAN augmentation, separate from IDS.

This is a small adversarial generator/discriminator pair, not CTGAN and not
SMOTE. Generated numeric values are synthetic; a conditioning label does not
establish that the row represents a real or successful attack. Protocol bits,
window intervals and missing masks come from a same-class real training donor.
No validation or test input is accepted by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from framework.training.tracking import write_json


def array_sha256(values: np.ndarray) -> str:
    """Hash contiguous row blocks without copying a full training array."""
    digest = hashlib.sha256()
    for start in range(0, len(values), 8192):
        digest.update(np.asarray(values[start:start + 8192]).tobytes(order="C"))
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_generated(values: np.ndarray, donors: np.ndarray, names: list[str]) -> dict:
    """Audit explicit schema invariants; does not prove physical realism."""
    values, donors = np.asarray(values), np.asarray(donors)
    if values.ndim != 2 or values.shape != donors.shape or values.shape[1] != len(names):
        raise ValueError("Generated values and donor schema do not match")
    violations = []
    if not np.array_equal(np.isnan(values), np.isnan(donors)):
        violations.append("missing masks changed")
    if np.isinf(values).any():
        violations.append("infinite values")
    fixed = [i for i, n in enumerate(names) if "__present_" in n or n.startswith("network_protocols_")
             or n in {"network_interval-packets", "log_interval-messages"}]
    if fixed and not np.array_equal(values[:, fixed], donors[:, fixed], equal_nan=True):
        violations.append("protocol/window metadata changed")
    for i, name in enumerate(names):
        if name.endswith("_count") or name in {"network_fragmented-packets"}:
            finite = values[np.isfinite(values[:, i]), i]
            if np.any(finite < 0) or not np.allclose(finite, np.rint(finite), atol=1e-5, rtol=0):
                violations.append(f"invalid integer count: {name}")
    by_name = {name: i for i, name in enumerate(names)}
    for name in names:
        if not name.endswith("_avg"):
            continue
        prefix = name[:-4]
        if prefix + "_min" in by_name and prefix + "_max" in by_name:
            columns = [by_name[prefix + suffix] for suffix in ("_min", "_avg", "_max")]
            rows = values[:, columns]
            complete = np.all(np.isfinite(rows), axis=1)
            if np.any(rows[complete, 0] > rows[complete, 1]) or np.any(rows[complete, 1] > rows[complete, 2]):
                violations.append(f"invalid min/avg/max: {prefix}")
    if "network_packets_all_count" in by_name:
        total = values[:, by_name["network_packets_all_count"]]
        for i, name in enumerate(names):
            if (name.startswith("network_tcp-flags-") and name.endswith("_count")) or name in {
                    "network_packets_src_count", "network_packets_dst_count", "network_fragmented-packets"}:
                finite = np.isfinite(total) & np.isfinite(values[:, i])
                if np.any(values[finite, i] > total[finite]):
                    violations.append(f"count exceeds total packets: {name}")
    return {"passed": not violations, "violations": violations, "rows": len(values),
            "checks": ["same-class donor missing mask", "fixed protocol bits and window intervals",
                       "finite integer nonnegative counts", "ordered min/avg/max", "TCP flags and packet subsets <= total"],
            "limitation": "Schema constraints do not validate attack semantics or cross-feature physical realism"}


def project_generated(generated: np.ndarray, donors: np.ndarray, names: list[str], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Project finite numeric outputs using bounds fitted only to training rows."""
    result = np.asarray(generated, dtype=np.float32).copy()
    donors = np.asarray(donors, dtype=np.float32)
    if result.shape != donors.shape or result.shape[1] != len(names):
        raise ValueError("Generated/donor/schema shapes differ")
    result = np.clip(result, lower, upper)
    by_name = {name: i for i, name in enumerate(names)}
    for i, name in enumerate(names):
        if "__present_" in name or name.startswith("network_protocols_") or name in {"network_interval-packets", "log_interval-messages"}:
            result[:, i] = donors[:, i]
        elif name.endswith("_count") or name == "network_fragmented-packets":
            result[:, i] = np.maximum(0, np.rint(result[:, i]))
        elif name.endswith("_std_deviation"):
            result[:, i] = np.maximum(0, result[:, i])
    for name in names:
        if not name.endswith("_avg"):
            continue
        prefix = name[:-4]
        if prefix + "_min" not in by_name or prefix + "_max" not in by_name:
            continue
        a, lo, hi = by_name[name], by_name[prefix + "_min"], by_name[prefix + "_max"]
        low = np.minimum(result[:, lo], result[:, hi])
        high = np.maximum(result[:, lo], result[:, hi])
        result[:, lo], result[:, hi] = low, high
        result[:, a] = np.clip(result[:, a], low, high)
        std = by_name.get(prefix + "_std_deviation")
        if std is not None:
            result[:, std] = np.minimum(result[:, std], high - low)
    if "network_packets_all_count" in by_name:
        total = result[:, by_name["network_packets_all_count"]]
        for i, name in enumerate(names):
            if (name.startswith("network_tcp-flags-") and name.endswith("_count")) or name in {
                    "network_packets_src_count", "network_packets_dst_count", "network_fragmented-packets"}:
                result[:, i] = np.minimum(result[:, i], total)
    result[np.isnan(donors)] = np.nan
    audit = validate_generated(result, donors, names)
    if not audit["passed"]:
        raise ValueError(f"Synthetic rows failed schema checks: {audit['violations']}")
    return result


def gan_request(X_train, y_train, feature_names: list[str], seed: int, steps: int, batch_size: int) -> dict:
    """Public artifact identity shared by generation and resume validation."""
    return {"seed": seed, "steps": steps, "batch_size": batch_size, "feature_names": feature_names,
               "training_rows": len(y_train), "generator": "32-noise + 16-label embedding + fixed donor context -> 128 -> 128 -> numeric tanh",
               "discriminator": "numeric + fixed donor context + 16-label embedding -> 128 -> 128 -> real/fake logit",
               "minority_selection": "Training class frequency >=100 and <1000; <=min(training_count,256) synthetic rows per class",
               "test_evaluated": False, "validation_used": False,
               "training_features_sha256": array_sha256(X_train), "training_labels_sha256": array_sha256(y_train)}


def train_conditional_gan(X_train, y_train, feature_names: list[str], output: Path,
                          seed: int = 42, steps: int = 1200, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit genuine CUDA cGAN on training minorities and save reproducible rows.

    Training minority counts in [100, 1000) determine the classes; at most 256
    synthetic rows per class and no more than the class's actual training count.
    The GAN sees all 7,945 eligible minority real training rows on this dataset;
    every downstream XGBoost fit additionally retains all 480,405 real rows.
    """
    output = Path(output)
    request = gan_request(X_train, y_train, feature_names, seed, steps, batch_size)
    if (output / "metadata.json").is_file():
        metadata = json.loads((output / "metadata.json").read_text())
        if metadata["request"] != request:
            raise ValueError("Existing GAN configuration differs; use a new output directory")
        values = np.load(output / "synthetic_features.npy")
        labels = np.load(output / "synthetic_labels.npy")
        donors = np.load(output / "training_donor_indices.npy")
        if values.shape != (metadata["synthetic_rows"], len(feature_names)) or labels.shape != (len(values),):
            raise ValueError("Stored GAN synthetic shapes differ from metadata")
        if donors.shape != labels.shape or not np.issubdtype(donors.dtype, np.integer) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Stored GAN donor/label schema is invalid")
        if np.any(donors < 0) or np.any(donors >= len(y_train)) or not np.array_equal(np.asarray(y_train)[donors], labels):
            raise ValueError("Stored GAN donors are outside training or have mismatched classes")
        if hashlib.sha256(values.tobytes() + labels.tobytes()).hexdigest() != metadata["synthetic_sha256"]:
            raise ValueError("Stored GAN synthetic data checksum differs from metadata")
        if array_sha256(donors) != metadata["donor_indices_sha256"] or file_sha256(output / "gan.pt") != metadata["checkpoint_sha256"]:
            raise ValueError("Stored GAN donor/checkpoint checksum differs from metadata")
        if not validate_generated(values, np.asarray(X_train)[donors], feature_names)["passed"]:
            raise ValueError("Stored GAN synthetic data failed schema invariants")
        return values, labels, metadata
    if (output / "gan.pt").exists():
        raise FileExistsError("Refusing to overwrite an incomplete GAN checkpoint")
    import torch
    from torch import nn
    if not torch.cuda.is_available():
        raise RuntimeError("Conditional GAN requires CUDA; no CPU training fallback")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train, dtype=np.int32)
    counts = np.bincount(y)
    selected = np.flatnonzero((counts >= 100) & (counts < 1000))
    if len(selected) < 2:
        raise ValueError("Insufficient supported minority classes for conditional GAN")
    ids = np.flatnonzero(np.isin(y, selected))
    data = np.asarray(X_train[ids], dtype=np.float32)
    if len(feature_names) != data.shape[1] or np.isinf(data).any():
        raise ValueError("Training schema is inconsistent or contains infinities")
    # Binary protocol presence and duration metadata are conditioned on, never
    # generated. Remaining numeric channels use training-only signed-log scaling.
    mutable = np.array([i for i, n in enumerate(feature_names) if "__present_" not in n
                        and not n.startswith("network_protocols_")
                        and n not in {"network_interval-packets", "log_interval-messages"}], dtype=int)
    fixed = np.array([i for i in range(len(feature_names)) if i not in mutable], dtype=int)
    raw = data[:, mutable]
    signed_log = np.sign(raw) * np.log1p(np.abs(raw))
    low, high = np.nanmin(signed_log, axis=0), np.nanmax(signed_log, axis=0)
    low = np.nan_to_num(low, nan=0.)
    high = np.nan_to_num(high, nan=0.)
    span = np.maximum(high - low, 1e-6)
    scaled = 2 * (np.nan_to_num(signed_log, nan=0.) - low) / span - 1
    scaled = np.clip(scaled, -1, 1).astype(np.float32)
    masks = np.isfinite(raw).astype(np.float32)
    context = np.concatenate((np.nan_to_num(data[:, fixed]), masks), axis=1)
    local_y = np.searchsorted(selected, y[ids])
    values = torch.tensor(scaled, device="cuda:0")
    contexts = torch.tensor(context, device="cuda:0")
    labels = torch.tensor(local_y, device="cuda:0", dtype=torch.long)
    n_classes = len(selected)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label = nn.Embedding(n_classes, 16)
            self.net = nn.Sequential(nn.Linear(32 + 16 + context.shape[1], 128), nn.LeakyReLU(.2),
                                     nn.Linear(128, 128), nn.LeakyReLU(.2), nn.Linear(128, len(mutable)), nn.Tanh())

        def forward(self, z, label, donor_context):
            return self.net(torch.cat((z, self.label(label), donor_context), dim=1))

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label = nn.Embedding(n_classes, 16)
            self.net = nn.Sequential(nn.Linear(len(mutable) + context.shape[1] + 16, 128), nn.LeakyReLU(.2),
                                     nn.Linear(128, 128), nn.LeakyReLU(.2), nn.Linear(128, 1))

        def forward(self, row, label, donor_context):
            return self.net(torch.cat((row, self.label(label), donor_context), dim=1))

    generator, discriminator = Generator().cuda(), Discriminator().cuda()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=.0002, betas=(.5, .9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5, .9))
    criterion = nn.BCEWithLogitsLoss()
    pools = [np.flatnonzero(local_y == i) for i in range(n_classes)]
    started = time.perf_counter()
    losses = []
    for step in range(steps):
        chosen_labels = rng.integers(0, n_classes, size=batch_size)
        chosen = np.empty(batch_size, dtype=np.int64)
        for i, pool in enumerate(pools):
            positions = np.flatnonzero(chosen_labels == i)
            chosen[positions] = rng.choice(pool, size=len(positions))
        index = torch.tensor(chosen, device="cuda:0")
        real, label, ctx = values[index], labels[index], contexts[index]
        noise = torch.randn(batch_size, 32, device="cuda:0")
        fake = generator(noise, label, ctx)
        # Missing channels must look identical to the fixed donor mask on both
        # sides of the adversarial task; otherwise D can win by NaN imputation
        # artifacts instead of learning the observed numeric distribution.
        fake = torch.where(ctx[:, -len(mutable):] > .5, fake, real)
        d_optimizer.zero_grad(set_to_none=True)
        d_real, d_fake = discriminator(real, label, ctx), discriminator(fake.detach(), label, ctx)
        d_loss = criterion(d_real, torch.full_like(d_real, .9)) + criterion(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        d_optimizer.step()
        g_optimizer.zero_grad(set_to_none=True)
        generated_logits = discriminator(fake, label, ctx)
        g_loss = criterion(generated_logits, torch.ones_like(generated_logits)) + .1 * (fake.mean(0) - real.mean(0)).square().mean()
        g_loss.backward()
        g_optimizer.step()
        if not torch.isfinite(g_loss) or not torch.isfinite(d_loss):
            raise RuntimeError("Nonfinite GAN loss; generated rows rejected")
        if step % 200 == 0 or step == steps - 1:
            row = {"step": step + 1, "generator_loss": float(g_loss), "discriminator_loss": float(d_loss)}
            losses.append(row)
            print(f"GAN {step + 1}/{steps} G={row['generator_loss']:.4f} D={row['discriminator_loss']:.4f}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    synth, synth_labels, donor_ids, audits = [], [], [], []
    generator.eval()
    with torch.no_grad():
        for i, class_id in enumerate(selected):
            number = min(int(counts[class_id]), 256)
            donor_local = rng.choice(pools[i], size=number, replace=True)
            index = torch.tensor(donor_local, device="cuda:0")
            fake = generator(torch.randn(number, 32, device="cuda:0"), labels[index], contexts[index]).cpu().numpy()
            logged = (fake + 1) / 2 * span + low
            numeric = np.sign(logged) * np.expm1(np.abs(logged))
            donors = data[donor_local]
            candidate = donors.copy()
            candidate[:, mutable] = numeric
            class_real = data[pools[i]]
            with np.errstate(invalid="ignore"):
                lower, upper = np.nanmin(class_real, axis=0), np.nanmax(class_real, axis=0)
            lower = np.nan_to_num(lower, nan=0.)
            upper = np.nan_to_num(upper, nan=0.)
            projected = project_generated(candidate, donors, feature_names, lower, upper)
            audit = validate_generated(projected, donors, feature_names)
            audit["class_index"] = int(class_id)
            audits.append(audit)
            synth.append(projected)
            synth_labels.extend([class_id] * number)
            donor_ids.extend(ids[donor_local])
    synthetic = np.concatenate(synth).astype(np.float32)
    target = np.asarray(synth_labels, dtype=np.int32)
    donors = np.asarray(donor_ids, dtype=np.int64)
    # Extremely collapsed projected output is rejected instead of passed off as
    # useful synthetic diversity. This remains a simple development safeguard.
    hashes = [hashlib.sha256(row.tobytes()).digest() for row in synthetic]
    unique_fraction = len(set(hashes)) / len(hashes)
    if unique_fraction < .5:
        raise RuntimeError(f"GAN projected output collapsed: only {unique_fraction:.1%} unique rows")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {"generator_state": generator.state_dict(), "discriminator_state": discriminator.state_dict(),
                  "mutable_indices": mutable, "fixed_indices": fixed, "selected_class_indices": selected,
                  "signed_log_lower": low, "signed_log_span": span, "request": request}
    torch.save(checkpoint, output / "gan.pt")
    np.save(output / "synthetic_features.npy", synthetic)
    np.save(output / "synthetic_labels.npy", target)
    np.save(output / "training_donor_indices.npy", donors)
    metadata = {"request": request, "backend": "PyTorch CUDA conditional adversarial generator and discriminator",
                "gpu_training_verified": True, "device": str(next(generator.parameters()).device),
                "gpu_name": torch.cuda.get_device_name(0), "training_seconds": seconds,
                "real_gan_training_rows": len(ids), "selected_class_indices": selected.tolist(),
                "synthetic_rows": len(target), "synthetic_class_counts": np.bincount(target, minlength=len(counts)).tolist(),
                "unique_projected_row_fraction": unique_fraction, "loss_history": losses, "domain_audits": audits,
                "synthetic_sha256": hashlib.sha256(synthetic.tobytes() + target.tobytes()).hexdigest(),
                "donor_indices_sha256": array_sha256(donors), "checkpoint_sha256": file_sha256(output / "gan.pt"),
                "limitation": "Synthetic condition labels and schema validity do not establish realistic attack semantics; benefit is tested only by a same-split detector A/B comparison",
                "torch_version": torch.__version__, "determinism": "Seeded NumPy/Torch/CUDA, deterministic Torch algorithms; XGBoost GPU reductions may vary"}
    write_json(output / "metadata.json", metadata)
    del values, contexts, labels, generator, discriminator
    torch.cuda.empty_cache()
    return synthetic, target, metadata
