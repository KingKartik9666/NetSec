"""Upgrade round 2: shared rules and guards (plan: ml_pipeline/notes_20261101/upgrades2_plan.md; import-safe).

No environment variable, study folder or data file is touched at import. The drivers
(``experiment.train_upgrades2_live``, ``experiment.train_fix77_ext``) and the queue (``experiment.upgrades2_queue``)
import the pure rules below so the tests can check them without a GPU or a study.

* U2/U3  ``u3_admitted`` / ``paired_summary``: the fix77 admission rule plus "mean role-0 accuracy gain >= 0" (the
  plan's round-wide "accuracy not lower" condition; U2 families use it for admission and confirmation).
* U4  ``round_curve`` + ``choose_rounds``: probe curves averaged at shared 10-round truncations, argmax of the
  3-point moving mean (edge-padded, as ``train_live77.final``), ties -> fewer rounds; a count above the frozen one
  is adopted only with >= +0.1 pp smoothed role-0 accuracy and a role-0 macro-F1 drop <= 0.005.
* U5  ``adjusted_logits`` / ``select_tau`` / ``u5_adopted``: z = log p_raw - tau log prior on RAW probabilities;
  decision argmax z; display softmax(z / T) (``display_proba``), which never changes the decision.
  ``tau_selection_jobs``: the plan amendment's three fold-1 jobs (final spec, final rounds, seeds 151-153) whose
  OOF selects tau.
* No-harm (plan amendment 2026-11-02)  ``per_class_no_harm`` / ``no_harm_adopted``: classes with >= 30 evaluated
  rows may lose at most 2 pp recall and 2 pp precision against the control (mean of paired seeds when several);
  U3, U4 and U5 (U1 with ping-sweep and recon_scan exempt) are adopted only when every check passes.
* Guards: ``other_controllers`` (fix77 markers, which since 2026-11-02 include this round's drivers, so a fix77
  driver started by hand also waits for them; the queue itself is deliberately not a marker, so the drivers never see
  the waiting watchdog as a GPU job), ``fix77_gate``, ``gpu_free_mib``, ``read_frozen``, ``quarantine_failed``,
  ``guard_run_job`` (re-checks other controllers before every training job: train_live77, train_live77_upgrades and
  train_cascade_v3 have no cross-driver guard and train_fs77_tabm sees only its own study, so one of those started by
  hand would not wait for this round's drivers; the queue's operating rule forbids that), ``hybrid_job_name``
  (rebuilt hybrids are keyed by their member jobs), ``write_json_atomic``, ``clear_stop_for_run`` and
  ``cpu_only_reason`` (the test suite runs CPU only while a controller or runner job is alive).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from framework.training.csv_v3 import arms, fix77

GAIN, F1_TOL = arms.GAIN, arms.F1_TOL
EPS = 1e-12                                   # float tolerance for ">=" thresholds on means of accuracies
DRIVER_MARKERS = ("experiment.train_upgrades2_live", "experiment.train_fix77_ext")
CONTROLLER_MARKERS = tuple(dict.fromkeys(tuple(fix77.CONTROLLER_MARKERS) + DRIVER_MARKERS))   # ordered, no repeats
TAU_GRID = (0.0, 0.125, 0.25, 0.375, 0.5)
LOG_FLOOR = 1e-12                             # the floor of the report-only posthoc that suggested U5
ROUND_STEP, ROUND_LO, ROUND_HI = 10, 10, 150
QUEUED_ENV = "UPGRADES2_QUEUED"               # set by the queue for the drivers it launches
CPU_TESTS_ENV = "IMDAI_TESTS_CPU_ONLY"        # "1" forces the CPU-only test mode


# ------------------------------------------------------------------------------------------------ U3
def u3_admitted(s: dict, n_seeds: int = 3) -> bool:
    """Plan U3: mean J gain >= 0.1 pp and > 2 x SD, unseen gain >= 0, OOF macro-F1 change >= -0.005 (all seeds run;
    ``arms.admitted``) AND mean role-0 accuracy gain >= 0."""
    return bool(arms.admitted(s, n_seeds) and s["tune_acc_gain"] >= 0)


def paired_summary(control: list[dict], runs: list[dict], n_seeds: int) -> dict:
    """``arms.paired`` plus the mean paired role-0 accuracy gain and the U3 decision."""
    by = {r["seed"]: r for r in control}
    s = arms.paired(control, runs)
    s["tune_acc_gain"] = float(np.mean([r["tune_acc"] - by[r["seed"]]["tune_acc"] for r in runs]))
    s["seeds_run"] = len(runs)
    s["admitted"] = u3_admitted(s, n_seeds)
    return s


# ------------------------------------------------------------------------------------------------ U4
def round_curve(probe_curves: list[list[dict]], lo: int = ROUND_LO, hi: int = ROUND_HI,
                step: int = ROUND_STEP) -> list[dict]:
    """Role-0 accuracy / macro-F1 per truncation, averaged over the probes that reach it. Only multiples of
    ``step`` in [lo, hi] are kept (a probe's own last round, e.g. 75, is not a 10-round truncation)."""
    by: dict[int, list[dict]] = {}
    for curve in probe_curves:
        for p in curve:
            r = int(p["rounds"])
            if r % step == 0 and lo <= r <= hi:
                by.setdefault(r, []).append(p)
    return [{"rounds": r, "tune_acc": float(np.mean([p["tune_acc"] for p in by[r]])),
             "tune_f1": float(np.mean([p["tune_f1"] for p in by[r]])), "n_probes": len(by[r])} for r in sorted(by)]


def smooth3(values) -> np.ndarray:
    """3-point moving mean with edge padding (``train_live77.final``)."""
    v = np.asarray(values, float)
    return np.convolve(np.pad(v, 1, mode="edge"), np.ones(3) / 3, mode="valid")


def choose_rounds(curve: list[dict], base: int, gain: float = GAIN, f1_tol: float = F1_TOL) -> dict:
    """Plan U4. ``best`` = argmax of the smoothed accuracy (first maximum = fewer rounds). The frozen count ``base``
    is kept unless ``best`` is larger, beats it by >= ``gain`` smoothed, and loses <= ``f1_tol`` role-0 macro-F1."""
    rounds = [int(p["rounds"]) for p in curve]
    if not curve or base not in rounds:
        return {"chosen": int(base), "adopted": False, "best": None, "reason": f"{base} rounds not on the curve"}
    s = smooth3([p["tune_acc"] for p in curve])
    # First index within EPS of the maximum: windows holding the same values in another order can differ by one ulp,
    # and an exact argmax would then pick a later (larger) count, breaking "ties -> fewer rounds".
    k, b = int(np.flatnonzero(s >= s.max() - EPS)[0]), rounds.index(base)
    out = {"best": rounds[k], "smoothed": [float(x) for x in s], "rounds": rounds,
           "smoothed_gain": float(s[k] - s[b]), "f1_change": float(curve[k]["tune_f1"] - curve[b]["tune_f1"])}
    if rounds[k] <= base:
        return {**out, "chosen": int(base), "adopted": False, "reason": "smoothed argmax is not above the frozen count"}
    if s[k] - s[b] < gain - EPS:
        return {**out, "chosen": int(base), "adopted": False, "reason": "smoothed gain below +0.1 pp"}
    if curve[k]["tune_f1"] - curve[b]["tune_f1"] < -f1_tol - EPS:
        return {**out, "chosen": int(base), "adopted": False, "reason": "role-0 macro-F1 falls by more than 0.005"}
    return {**out, "chosen": rounds[k], "adopted": True, "reason": "rule passed"}


U4_RULE = ("argmax of the 3-point moving mean of role-0 accuracy over 10-round truncations 10-150 (probe mean), "
           "ties -> fewer rounds; adopt a larger count only with >= +0.1 pp smoothed and macro-F1 drop <= 0.005")


def u4_record(spec: str, base_rounds: int, curve: list[dict], choice: dict, probes: list[dict], smoke: bool,
              no_harm: dict | None = None) -> dict:
    """The frozen U4 decision: deterministic fields only. Wall-clock latency is NOT part of it (a re-measured latency
    differs on every run and would make every restart fail the freeze); the driver writes it to u4_latency.json.
    A larger count that passed ``choose_rounds`` is adopted only if its role-0 per-class no-harm check (chosen count
    vs the frozen-count truncation; amendment 2026-11-02) passed; otherwise the frozen count is kept."""
    if choice["adopted"] and no_harm is None:
        raise ValueError("a larger round count needs its per-class no-harm check")
    adopted = no_harm_adopted(choice["adopted"], no_harm)
    return {"spec": spec, "base_rounds": int(base_rounds), "curve": curve, "choice": choice, "no_harm_role0": no_harm,
            "adopted": adopted, "rounds": int(choice["chosen"]) if adopted else int(base_rounds), "probes": probes,
            "rule": U4_RULE + "; and the per-class no-harm check on role 0 (chosen count vs the frozen count)",
            "smoke": bool(smoke)}


# ------------------------------------------------------------------------------------------------ U5
def class_prior(y: np.ndarray, rows: np.ndarray, n_classes: int) -> np.ndarray:
    """Class frequencies of the rows a model was fitted on."""
    c = np.bincount(np.asarray(y)[rows], minlength=n_classes).astype(np.float64)
    return c / c.sum()


def adjusted_logits(p_raw: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """z = log p_raw - tau x log prior (floors 1e-12). The decision is argmax z."""
    return np.log(np.clip(np.asarray(p_raw, np.float64), LOG_FLOOR, 1.0)) - \
        float(tau) * np.log(np.clip(np.asarray(prior, np.float64), LOG_FLOOR, 1.0))


def display_proba(z: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(z / T): the displayed probabilities. T > 0 never changes argmax."""
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    s = np.asarray(z, np.float64) / float(temperature)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def select_tau(oof: list[tuple[np.ndarray, np.ndarray, np.ndarray]], grid=TAU_GRID) -> dict:
    """Mean OOF accuracy over models (each: raw OOF probabilities, labels, its fit prior) for every tau; the best
    tau is the first maximum in ascending order (ties -> smaller tau). Gain is measured against tau = 0."""
    grid = sorted(float(t) for t in grid)
    if grid[0] != 0.0:
        raise ValueError("the grid must contain tau = 0 (the reference)")
    acc = {t: float(np.mean([np.mean(adjusted_logits(p, prior, t).argmax(1) == y) for p, y, prior in oof]))
           for t in grid}
    best = grid[0]
    for t in grid[1:]:
        if acc[t] > acc[best] + EPS:
            best = t
    return {"tau": best, "oof_acc": {str(t): a for t, a in acc.items()}, "gain": acc[best] - acc[0.0]}


def u5_adopted(sel: dict, role0: dict) -> bool:
    """Plan U5 at the selected tau: OOF accuracy gain >= +0.1 pp; role-0 accuracy gain >= 0; role-0 unseen gain >= 0;
    role-0 macro-F1 change >= -0.005 (role-0 numbers from the final model's raw probabilities)."""
    return bool(sel["tau"] > 0 and sel["gain"] >= GAIN - EPS and role0["acc_gain"] >= 0
                and role0["unseen_gain"] >= 0 and role0["f1_change"] >= -F1_TOL - EPS)


def tau_selection_jobs(configs: dict, seeds, spec: str, rounds: int, weight: dict | None) -> dict[int, tuple[str, dict]]:
    """Plan U5 amendment (2026-11-01, before any U5 result): tau is selected on fold-1 OOF of fold-1 models trained
    with the FINAL spec and the FINAL round count (U4's decision) on seeds 151-153, not on the 18-round U3 fold
    models (their logit scale differs from the >= 50-round final model). Returns seed -> (job id, config): the seed's
    config, plus ``weight`` as row_weight when the spec is WC, with exactly ``rounds`` boosting rounds (early stopping
    set past the last round). The job id carries spec and rounds, so a cached job of another spec/round count is
    never reused (``runner.Controller.run_job`` also refuses a cached id whose config differs)."""
    import copy
    rounds = int(rounds)
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if (spec == "WC") != (weight is not None):
        raise ValueError(f"spec {spec!r} and row weight {weight!r} disagree")
    out = {}
    for s in seeds:
        c = copy.deepcopy(configs[s])
        c.pop("row_weight", None)
        if weight is not None:
            c["row_weight"] = dict(weight)
        c.update(n_estimators=rounds, early_stopping_rounds=rounds + 1)
        out[int(s)] = (f"77_u2_tau_{spec}_r{rounds}_s{int(s)}", c)
    return out


def final_display(p_raw: np.ndarray, prior: np.ndarray, u5: dict, calibration: dict) -> np.ndarray:
    """Displayed probabilities the final live pipeline gives raw probabilities ``p_raw`` (before the ping merge):
    softmax(z / T) when U5 is adopted, else the final model's calibrator (identity: unchanged). U1 sums two of these
    entries, which is not invariant to T or smoothing, so its OOF check must use the same transform as role 0."""
    if u5["adopted"]:
        return display_proba(adjusted_logits(p_raw, prior, u5["tau"]), u5["temperature"])
    if (calibration or {}).get("kind", "identity") == "identity":
        return np.asarray(p_raw, np.float64)
    from framework.training.csv_v3 import calibrate
    return np.asarray(calibrate.apply(calibration, np.asarray(p_raw)), np.float64)


def proba_scale(p: np.ndarray) -> dict:
    """Report-only scale of raw probabilities: mean max probability and mean top-1 minus top-2 log-probability
    margin (log floor 1e-12). Shows whether the fold models tau was selected on (final spec and rounds, fold-1 rows)
    have the final model's logit scale, and how far the 18-round U3 fold models are from it; nothing is chosen
    from it."""
    lp = np.log(np.clip(np.asarray(p, np.float64), LOG_FLOOR, 1.0))
    top = np.sort(lp, axis=1)[:, -2:]
    return {"mean_max_proba": float(np.exp(top[:, 1]).mean()),
            "mean_log_margin": float((top[:, 1] - top[:, 0]).mean()), "rows": int(len(lp))}


# ----------------------------------------------------------------------- per-class no-harm (amendment 2026-11-02)
NO_HARM_MIN_ROWS, NO_HARM_MAX_DROP = 30, 0.02
NO_HARM_RULE = ("per-class no-harm (plan amendment 2026-11-02): every class with >= 30 evaluated rows (exempt classes "
                "excepted) may lose at most 2 pp recall and at most 2 pp precision against the control; precision is "
                "checked when the control predicts the class at least once (a candidate that never predicts it has "
                "precision 0); with several paired seeds/probes the MEAN paired drop per class must be <= 2 pp; any "
                "breach -> not adopted")


def _pairs(y, control, candidate) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(labels, control, candidate) per pair. One pair when ``control`` is a 1-D prediction array; otherwise a
    sequence of paired prediction arrays, with ``y`` either shared or one label array per pair."""
    def flat(a) -> bool:                       # one prediction/label array (not a sequence of them)
        return a.ndim == 1 if isinstance(a, np.ndarray) else (len(a) == 0 or np.ndim(a[0]) == 0)

    controls, candidates = ([control], [candidate]) if flat(control) else (list(control), list(candidate))
    if len(controls) != len(candidates) or not controls:
        raise ValueError("control and candidate must hold the same number (>= 1) of paired predictions")
    ys = [y] * len(controls) if flat(y) else list(y)
    if len(ys) != len(controls):
        raise ValueError("one label array per pair (or one shared array) is required")
    out = []
    for yy, c, x in zip(ys, controls, candidates):
        yy, c, x = (np.asarray(a, np.int64).ravel() for a in (yy, c, x))
        if not len(yy) == len(c) == len(x):
            raise ValueError("labels, control and candidate predictions differ in length")
        out.append((yy, c, x))
    return out


def per_class_no_harm(y, control, candidate, n_classes: int, min_rows: int = NO_HARM_MIN_ROWS,
                      max_drop: float = NO_HARM_MAX_DROP, exempt=(), names=None) -> dict:
    """Plan amendment 2026-11-02, the per-class no-harm filter of U3/U4/U5 (and U1 with ping-sweep and recon_scan
    exempt). It can only refuse an adoption, never raise a score.

    ``y``/``control``/``candidate``: labels and argmax predictions of the evaluated rows (role 0 or fold-1 OOF).
    Checked classes: at least ``min_rows`` rows in the evaluated rows (with several pairs whose labels differ: in
    every pair), not in ``exempt`` (class indices). Recall drop = control recall - candidate recall; a breach is a
    drop > ``max_drop`` (2 pp; exactly 2 pp passes, EPS tolerance). Precision is checked only where the control
    predicts the class at least once; a candidate that never predicts it has precision 0.

    Multi-seed (U3: C0 vs WC on seeds 151-153; U4: the probes that reach the chosen count; U5 OOF: the three
    tau-selection jobs): pass one prediction array per pair. Per class, the drop is computed per pair and the MEAN
    over pairs must be <= ``max_drop`` (the plan's "mean of paired seeds"); a single bad seed inside a passing mean
    does not fail the check. Precision means use only the pairs whose control predicts the class (checked when at
    least one does). Before/after values are means over the same pairs.

    Returns ``passed``, ``checked`` (class names, or indices without ``names``), per-class recall/precision before,
    after and drop, the largest recall and precision drops (ties -> lowest class index) and the ``breaches``. With no
    class reaching ``min_rows`` nothing is checked and the result passes (stated by ``checked`` = [])."""
    pairs = _pairs(y, control, candidate)
    n, ex = int(n_classes), {int(c) for c in exempt}
    name = (lambda c: names[c]) if names is not None else (lambda c: int(c))
    rec_drop, rec_b, rec_a, prec = [], [], [], []
    for yy, c, x in pairs:
        support = np.bincount(yy, minlength=n).astype(float)
        tp_c = np.bincount(yy[c == yy], minlength=n).astype(float)
        tp_x = np.bincount(yy[x == yy], minlength=n).astype(float)
        pred_c, pred_x = np.bincount(c, minlength=n).astype(float), np.bincount(x, minlength=n).astype(float)
        r_c = np.divide(tp_c, support, out=np.zeros(n), where=support > 0)
        r_x = np.divide(tp_x, support, out=np.zeros(n), where=support > 0)
        p_c = np.divide(tp_c, pred_c, out=np.zeros(n), where=pred_c > 0)
        p_x = np.divide(tp_x, pred_x, out=np.zeros(n), where=pred_x > 0)     # never predicted -> precision 0
        rec_b.append(r_c), rec_a.append(r_x), rec_drop.append(r_c - r_x)
        prec.append((pred_c > 0, p_c, p_x, support))
    rows = np.min([s for *_, s in prec], axis=0)
    checked = [k for k in range(n) if rows[k] >= min_rows and k not in ex]
    per_class, breaches = [], []
    for k in checked:
        item = {"class": name(k), "rows": int(rows[k]), "recall_before": float(np.mean([r[k] for r in rec_b])),
                "recall_after": float(np.mean([r[k] for r in rec_a])),
                "recall_drop": float(np.mean([d[k] for d in rec_drop]))}
        use = [(pc[k], px[k]) for has, pc, px, _ in prec if has[k]]
        item["precision_checked"] = bool(use)
        item["precision_before"] = float(np.mean([u[0] for u in use])) if use else None
        item["precision_after"] = float(np.mean([u[1] for u in use])) if use else None
        item["precision_drop"] = float(np.mean([u[0] - u[1] for u in use])) if use else None
        item["precision_pairs"] = len(use)
        for metric in ("recall", "precision"):
            d = item[f"{metric}_drop"]
            if d is not None and d > max_drop + EPS:
                breaches.append({"class": name(k), "metric": metric, "drop": d})
        per_class.append(item)

    def largest(metric):
        items = [p for p in per_class if p[f"{metric}_drop"] is not None]
        if not items:
            return None
        top = max(items, key=lambda p: p[f"{metric}_drop"])        # max() keeps the first (lowest index) on ties
        return {"class": top["class"], "drop": top[f"{metric}_drop"]}

    return {"passed": not breaches, "checked": [p["class"] for p in per_class],
            "exempt": [name(k) for k in sorted(ex)], "min_rows": int(min_rows), "max_drop": float(max_drop),
            "n_pairs": len(pairs), "per_class": per_class, "largest_recall_drop": largest("recall"),
            "largest_precision_drop": largest("precision"), "breaches": breaches, "rule": NO_HARM_RULE}


def no_harm_adopted(rule_passed: bool, *checks) -> bool:
    """Adoption = the step's own rule AND every per-class no-harm check passed. A missing (None) check refuses:
    a step that skipped the filter is never adopted."""
    return bool(rule_passed and checks and all(c is not None and c["passed"] for c in checks))


# --------------------------------------------------------------------------------------- guards + I/O
def other_controllers(markers=CONTROLLER_MARKERS) -> list[int]:
    """PIDs of other training controllers or runner jobs (same logic as ``fix77.other_controllers``, wider list)."""
    import psutil
    mine = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    busy = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.info["pid"] in mine:
            continue
        try:
            cmd = p.info["cmdline"] or []
        except Exception:                      # noqa: BLE001 - a vanished process is not a controller
            continue
        if not [c for c in cmd if c in markers]:
            continue
        if any(c in ("status", "stop", "start") for c in cmd):
            continue
        busy.append(p.info["pid"])
    return busy


def read_phase(study: Path) -> str | None:
    path = Path(study) / "state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("phase")
    except (OSError, ValueError):
        return None


def fix77_gate(chain_alive: bool, busy: list, phases: dict) -> tuple[str, str]:
    """'wait' while the fix77 chain (or any controller/job) runs; afterwards 'go' only if every fix77 study reports
    phase 'done', else 'blocked' (error, stopped, running-but-dead or missing state never proceeds)."""
    if chain_alive:
        return "wait", "fix77 chain process is alive"
    if busy:
        return "wait", f"other controllers or jobs running: {busy}"
    bad = {k: v for k, v in phases.items() if v != "done"}
    if bad:
        return "blocked", f"fix77 chain has ended but these studies are not 'done': {bad}"
    return "go", "fix77 stage 1 finished (all studies done)"


def gpu_free_mib() -> int | None:
    """Free memory of GPU 0 in MiB (nvidia-smi), or None when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def read_frozen(path: Path):
    """Value of an ``arms.freeze`` file, verified against its sha256 (read-only: never writes)."""
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if arms._sha(stored["value"]) != stored["sha256"]:
        raise RuntimeError(f"{path}: stored value does not match its sha256")
    return stored["value"]


def quarantine_failed(study: Path, log=print) -> None:
    """Move failed job folders to ``failed_jobs/`` so a restart retries them (fix77 drivers' rule). A result.json
    that cannot be parsed (e.g. cut short by a reboot: ``runner._write_result`` is not atomic) counts as failed."""
    for res in sorted((Path(study) / "jobs").glob("*/result.json")):
        try:
            status = json.loads(res.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError, AttributeError):
            status = "unreadable"
        if status in ("failed", "unreadable"):
            dest = Path(study) / "failed_jobs" / f"{res.parent.name}_{time.strftime('%Y%m%d_%H%M%S')}"
            dest.parent.mkdir(exist_ok=True)
            os.replace(res.parent, dest)
            log(f"moved {status} job {res.parent.name} to {dest.relative_to(study)}; it will be retried")


def write_json_atomic(path: Path, value, retries: int = 20) -> None:
    """Write JSON through a temp file and ``os.replace`` (a reboot mid-write never leaves a half file)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")
    for attempt in range(retries):       # Windows: a reader (antivirus, indexer) can hold the target briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.25)


def clear_stop_for_run(study: Path, env=None) -> bool:
    """A manual ``run`` clears its study STOP; a driver launched by the queue does not (a queue ``stop`` issued
    while the driver starts must not be erased; the queue's ``start`` clears old STOP files). True if removed."""
    env = os.environ if env is None else env
    stop = Path(study) / "STOP"
    if env.get(QUEUED_ENV) == "1" or not stop.exists():
        return False
    stop.unlink(missing_ok=True)
    return True


def wait_alone(ctl, log=print, busy=None, poll: float = 60.0, note_every: float = 1800.0, sleep=time.sleep) -> None:
    """Block while another training controller or runner job is alive; STOP raises KeyboardInterrupt."""
    busy = other_controllers if busy is None else busy
    last = None
    while True:
        pids = busy()
        if not pids:
            return
        if ctl.stopped():
            raise KeyboardInterrupt("STOP present")
        now = time.time()
        if last is None or now - last >= note_every:
            log(f"waiting before the next training job: other controllers or jobs running {pids}")
            last = now
        sleep(poll)


def guard_run_job(ctl, log=print, busy=None, poll: float = 60.0, sleep=time.sleep):
    """Wrap ``ctl.run_job``: before a job that must train (no cached complete result), wait until no other
    controller or runner job is alive. This only protects this side: an upgrades2 driver never starts a fit while
    another controller is alive. The fix77 drivers list this round's drivers (fix77.CONTROLLER_MARKERS) and refuse to
    start; train_live77, train_live77_upgrades, train_cascade_v3 (as a controller) and train_fs77_tabm do not, so one
    of them started by hand while a driver is between jobs would not wait (operating rule: do not start them)."""
    original = ctl.run_job

    def run_job(job_id, *args, **kwargs):
        done = ctl.result(job_id)
        if done is None or done.get("status") != "complete":
            wait_alone(ctl, log, busy=busy, poll=poll, sleep=sleep)
        return original(job_id, *args, **kwargs)

    ctl.run_job = run_job
    return ctl


def hybrid_job_name(name: str, branches: dict, fold_jobs: dict) -> str:
    """Hybrid job name keyed by its member jobs (branch job ids + fold work dirs): a different member set never
    reuses a cached hybrid built from another set."""
    key = {m: {"job": branches[m]["job_id"],
               "folds": [Path(f["work_dir"]).as_posix() for f in fold_jobs.get(m, [])]} for m in sorted(branches)}
    return f"{name}_{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]}"


def check_cached_hybrid(result: dict | None, branches: dict) -> None:
    """Raise when a cached hybrid result was built from other member jobs than the requested ones."""
    if result is None or result.get("status") != "complete":
        return
    want = {m: branches[m]["job_id"] for m in branches}
    if result.get("branch_jobs") != want:
        raise RuntimeError(f"cached hybrid {result.get('job_id')} was built from {result.get('branch_jobs')}, "
                           f"requested {want}")


def shipped_jobs(export: dict, names) -> dict[str, str]:
    """Model -> shipped job id from the export report (only this mapping matters; its other keys do not)."""
    models = export["models"]
    return {n: models[n]["job"] for n in names if n in models}


def cpu_only_reason(env=None, busy=None) -> str | None:
    """Why the test suite must stay on the CPU (None: CUDA tests may run). A training controller or runner job
    alive, ``IMDAI_TESTS_CPU_ONLY=1``, or a failed check (fail safe) each forces CPU-only mode."""
    env = os.environ if env is None else env
    if env.get(CPU_TESTS_ENV) == "1":
        return f"{CPU_TESTS_ENV}=1"
    try:
        pids = (other_controllers if busy is None else busy)()
    except Exception as exc:              # noqa: BLE001 - cannot tell: stay off the GPU
        return f"controller check failed ({exc!r})"
    return f"training controllers or runner jobs running (pids {pids})" if pids else None


def ext_skip_reason(decision: dict | None, phase: str | None) -> str | None:
    """None when U2 must run: fix77 TabM-NE finished ('done') with an admitted arm that passed confirmation
    (never a smoke-forced one). Otherwise the reason U2 is skipped."""
    if phase != "done":
        return f"fix77 TabM-NE study phase is {phase!r}, not 'done'"
    if decision is None:
        return "fix77 TabM-NE decision.json is missing"
    if decision.get("smoke_forced"):
        return "fix77 decision was smoke-forced (wiring only)"
    chosen = decision.get("chosen")
    if not decision.get("admitted") or chosen is None:
        return "no fix77 arm was admitted for TabM-NE"
    if chosen not in decision["admitted"]:
        return f"chosen arm {chosen} is not among the admitted arms"
    if not decision.get("confirmed"):
        return f"{chosen} failed confirmation on seeds 151-153"
    return None
