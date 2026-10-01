"""Transparent live behavior rules; separate from the DataSense ML checkpoint."""

LIMITATIONS = (
    "Experimental behavior rules, not the trained XGBoost model or SHAP. "
    "No rule alert means benign-looking within this observation only, not proof of safety. "
    "Alerts may be legitimate activity and do not confirm an attack. Encrypted payloads are not inspected."
)


def assessment(status, evidence=None):
    labels = {"not_assessed": "Not checked yet", "idle": "No traffic observed",
              "benign_looking": "Benign-looking activity", "suspicious": "Suspicious activity — review needed"}
    return {"status": status, "label": labels[status], "method": "Live behavior rules",
            "evidence": evidence or [], "limitations": LIMITATIONS}


def assess_connections(connections: list[dict], packet_rate: float) -> dict:
    """Only current OS connection state is available; no packet contents inferred."""
    pending = sum(c["status"] == "SYN_SENT" for c in connections)
    evidence = []
    if pending >= 40:
        evidence.append({"rule": "connection_attempt_burst", "observed": pending, "threshold": 40,
                         "message": f"{pending} connections are awaiting a TCP handshake. This may indicate repeated connection attempts or connection failures."})
    return assessment("suspicious" if evidence else "benign_looking" if connections or packet_rate > 0 else "idle", evidence)


def assess_packets(stats: dict) -> dict:
    """Evaluate header counts using their observed duration, without an accuracy claim."""
    packets = stats["packets"]
    seconds = max(stats.get("seconds", 5), .1)
    evidence = []
    syn = stats.get("syn", 0)
    # A high bulk-download rate by itself is not an attack signature.
    if packets >= 100 and syn / packets >= .8 and syn / seconds >= 50:
        evidence.append({"rule": "syn_burst", "observed": syn / seconds, "threshold": 50,
                         "message": f"{syn} SYN-only packets in {seconds:.1f} seconds, with few other packets. Possible connection-attempt flooding; verify the endpoints."})
    max_ports = max(stats.get("unique_ports_per_source", {}).values(), default=0)
    if max_ports >= 25:
        evidence.append({"rule": "port_sweep", "observed": max_ports, "threshold": 25,
                         "message": f"One source sent connection attempts to {max_ports} distinct destination ports. Possible port scanning; verify intent."})
    return assessment("suspicious" if evidence else "benign_looking" if packets else "idle", evidence)
