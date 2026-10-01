"""Regenerate synthetic software-demo fixtures; never research benchmark data."""

import csv

import numpy as np

from framework.data.loader import DEFAULT_DATASET, FEATURES


def generate() -> None:
    """Write 180 seeded train rows and four hand-designed held-out examples."""
    rng = np.random.default_rng(42)
    # packet rate, packet size, SYN fraction, connections/s, destinations/s
    profiles = {
        "BENIGN": ([100, 600, 0.10, 5, 4], [18, 50, 0.015, 1, 1]),
        "DDoS": ([8500, 120, 0.93, 2500, 3], [600, 15, 0.02, 250, 1]),
        "PORT_SCAN": ([250, 70, 0.80, 220, 170], [35, 8, 0.025, 25, 20]),
    }
    rows = []
    for label, (center, spread) in profiles.items():
        for i in range(60):
            packet, size, syn, connection, destinations = rng.normal(center, spread)
            packet, size, connection = [round(max(1, x), 3) for x in (packet, size, connection)]
            values = [packet, round(packet * size, 3), round(float(np.clip(syn, 0, 1)), 4),
                      connection, max(1, round(destinations)), size]
            rows.append([f"train_{label.lower()}_{i:03d}", "train", label, *values])
    rows.extend([
        ["benign", "demo", "BENIGN", 105, 65100, 0.11, 5, 4, 620],
        ["ddos", "demo", "DDoS", 8800, 1056000, 0.94, 2650, 3, 120],
        ["scan", "demo", "PORT_SCAN", 260, 18200, 0.82, 230, 180, 70],
        ["unknown", "demo", "UNKNOWN", 65, 552500, 0.015, 7, 5, 8500],
    ])
    DEFAULT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_DATASET.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "split", "label", *FEATURES])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} SYNTHETIC DEMONSTRATION rows to {DEFAULT_DATASET}")


if __name__ == "__main__":
    generate()
