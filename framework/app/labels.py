"""Readable UI names; original encoded research labels remain unchanged."""
from collections import Counter
import re


def display_label(label: str) -> str:
    if label == "benign":
        return "Normal network activity"
    family, _, detail = label.partition("_")
    match = re.search(r"-port-(\d+)$", detail)
    port = f", port {match.group(1)}" if match else ""
    detail = re.sub(r"-port-\d+$", "", detail)
    descriptions = {
        "syn-flood": "SYN flooding", "ack-frag-flood": "Fragmented ACK flooding",
        "connect-flood": "Connection flooding", "http-flood": "HTTP request flooding",
        "icmp-flood": "Ping flooding", "icmp-frag-flood": "Fragmented ping flooding",
        "mqtt-publish-flood": "MQTT message flooding", "push-ack-flood": "PSH/ACK flooding",
        "rst-fin-flood": "RST/FIN flooding", "slowloris": "Slow connection attack",
        "synonymousip-flood": "Spoofed-address SYN flooding", "tcp-flood": "TCP flooding",
        "udp-flood": "UDP flooding", "udp-frag-flood": "Fragmented UDP flooding",
        "dictionary-ssh": "SSH password guessing", "dictionary-telnet": "Telnet password guessing",
        "mirai-syn-flood": "Mirai botnet SYN flooding", "mirai-udp-flood": "Mirai botnet UDP flooding",
        "arp-spoofing": "ARP address spoofing", "impersonation": "Device impersonation",
        "ip-spoofing": "IP address spoofing", "host-disc-arp-ping": "ARP host discovery",
        "host-disc-tcp-ack-ping": "ACK host discovery", "host-disc-tcp-syn-ping": "SYN host discovery",
        "host-disc-tcp-syn-stealth": "Stealth host discovery", "host-disc-udp-ping": "UDP host discovery",
        "os-scan": "Operating system scanning", "ping-sweep": "Network ping sweep",
        "port-scan": "Port scanning", "vuln-scan": "Vulnerability scanning",
        "backdoor-upload": "Backdoor upload", "command-injection": "Command injection",
        "sql-injection": "SQL injection", "sql-injection-blind": "Blind SQL injection", "xss": "Cross-site scripting",
    }
    name = descriptions.get(detail, detail.replace("-", " ").replace("_", " ").capitalize())
    category = {"ddos": "DDoS", "dos": "DoS", "bruteforce": "Password attack",
                "malware": "Malware", "mitm": "Interception", "recon": "Reconnaissance", "web": "Web attack"}.get(family)
    return f"{name} ({category}{port})" if category else name


def sample_names(samples: list[dict]) -> list[dict]:
    counts = Counter()
    result = []
    groups = {"benign": "Normal traffic", "ddos": "Distributed flooding", "dos": "Traffic flooding",
              "bruteforce": "Password attacks", "malware": "Malware", "mitm": "Interception",
              "recon": "Scanning and discovery", "web": "Web attacks"}
    for sample in samples:
        label = sample["label"]
        counts[label] += 1
        result.append({**sample, "display_name": f"{display_label(label)} — Example {counts[label]}",
                       "group": groups.get(label.split("_")[0], "Other traffic")})
    return result
