"""Opt-in local network observation. No payload storage or automatic network action."""
from __future__ import annotations

from collections import defaultdict
import copy
from datetime import datetime, timezone
import importlib.util
import ipaddress
import logging
import os
from pathlib import Path
import platform
import socket
import threading
import time
import uuid

from framework.detection.live_rules import assessment, assess_connections, assess_packets


class LiveMonitor:
    """One bounded local monitor, started by an explicit user action only."""

    def __init__(self, detector=None):
        self._lock = threading.RLock()
        self._control = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._started = None
        self._last_observation = None
        self._sniffer = None
        self._packet_model = None
        self._detector = detector
        self._capture_socket = None
        self._window = None
        self._ports = defaultdict(set)
        self._local_addresses = set()
        self._local_macs = set()
        self._data = {"running": False, "mode": None, "interface": None, "state": "stopped", "error": None,
                      "elapsed_seconds": 0, "metrics": {"packets_per_second": 0, "bytes_per_second": 0,
                                                        "packets_observed": 0, "active_connections": 0},
                      "assessment": assessment("not_assessed"), "model_prediction": None, "multiclass_prediction": None,
                      "connections": [], "recent_packets": [], "history": []}

    def capabilities(self) -> dict:
        available = importlib.util.find_spec("psutil") is not None
        interfaces = []
        if available:
            import psutil
            stats, counters = psutil.net_if_stats(), psutil.net_io_counters(pernic=True)
            for name, addresses in psutil.net_if_addrs().items():
                ips = [a.address.split("%")[0] for a in addresses if a.family in (socket.AF_INET, socket.AF_INET6)]
                if name in counters and ips and not all(ipaddress.ip_address(v).is_loopback for v in ips):
                    interfaces.append({"id": name, "name": name, "addresses": ips,
                                       "is_up": bool(stats.get(name) and stats[name].isup)})
        scapy_available = importlib.util.find_spec("scapy") is not None
        windows = platform.system() == "Windows"
        npcap = (Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/Npcap/wpcap.dll").is_file() if windows else True
        capture_available = scapy_available and npcap
        message = ("Ready for packet inspection; capture permission is checked when you start."
                   if capture_available else "Install Npcap and restart the app for packet inspection. Windows may require running the app as Administrator."
                   if windows and not npcap else "Install requirements-live.txt for packet inspection.")
        preferred = next((v["id"] for v in interfaces if v["is_up"] and v["name"].lower() in {"wi-fi", "wifi", "ethernet"}),
                         next((v["id"] for v in interfaces if v["is_up"]), ""))
        model_error = None
        try:
            model_compatible = bool(self._detector and self._detector.partial_input_supported)
        except Exception as error:
            model_compatible, model_error = False, str(error)
        return {"interfaces": interfaces, "recommended_interface": preferred, "connections_available": available,
                "packet_capture_available": capture_available, "packet_capture_message": message,
                "setup_url": "https://npcap.com/#download",
                "model_compatible": model_compatible, "model_error": model_error,
                "packet_model_available": (Path(__file__).resolve().parents[1] / "models/packet_live/bundle.json").is_file(),
                "packet_model_message": "Separate experimental 7-count XGBoost model; DataSense binary validation does not measure live PC accuracy."}

    def snapshot(self) -> dict:
        with self._lock:
            result = copy.deepcopy(self._data)
            if self._started:
                result["elapsed_seconds"] = round(time.monotonic() - self._started, 1)
            reference = self._last_observation or self._started
            if reference is not None:
                age = time.monotonic() - reference
                result["observation_age_seconds"] = round(age, 1)
                bound = 25 if result["mode"] == "packets" else 8
                if result["running"] and age > bound:
                    result.update(state="error", error="No fresh observations are available. Stop monitoring and retry.",
                                  assessment=assessment("not_assessed"), model_prediction=None, multiclass_prediction=None)
            if result.get("error") or not result["running"]:
                result.update(model_prediction=None, multiclass_prediction=None)
            return result

    def start(self, interface: str, mode: str) -> dict:
        with self._control:
            return self._start(interface, mode)

    def _start(self, interface: str, mode: str) -> dict:
        if mode not in {"connections", "packets"}:
            raise ValueError("Choose connections or packets monitoring")
        capabilities = self.capabilities()
        selected = next((v for v in capabilities["interfaces"] if v["id"] == interface), None)
        if selected is None:
            raise ValueError("Choose an available network adapter")
        if not capabilities["connections_available"]:
            raise ValueError("Install requirements-live.txt to monitor this PC")
        if mode == "packets" and not capabilities["packet_capture_available"]:
            raise ValueError(capabilities["packet_capture_message"])
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("Monitoring is already running. Stop it before changing the adapter or mode.")
            self._stop.clear()
            self._started = time.monotonic()
            self._last_observation = None
            self._local_addresses = set(selected["addresses"])
            import psutil
            self._local_macs = {a.address.replace("-", ":").lower()
                                for a in psutil.net_if_addrs().get(interface, []) if a.family == psutil.AF_LINK}
            self._data.update(running=True, mode=mode, interface=interface, state="starting", error=None,
                              elapsed_seconds=0, assessment=assessment("not_assessed"), connections=[], recent_packets=[], history=[])
            self._data["model_prediction"] = None
            self._data["multiclass_prediction"] = None
            self._data["metrics"] = {"packets_per_second": 0, "bytes_per_second": 0,
                                     "packets_observed": 0, "active_connections": 0}
            self._ports.clear()
            self._window = self._empty_window()
            self._thread = threading.Thread(target=self._run, args=(interface, mode), daemon=True, name="IMDAI-live-monitor")
            self._thread.start()
        return self.snapshot()

    def stop(self) -> dict:
        with self._control:
            return self._stop_monitor()

    def _stop_monitor(self) -> dict:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=6)
        with self._lock:
            if thread is not None and thread.is_alive():
                self._data.update(state="error", error="Monitor did not stop yet; wait before starting another capture.",
                                  assessment=assessment("not_assessed"), model_prediction=None, multiclass_prediction=None)
            else:
                self._data.update(running=False, state="stopped", assessment=assessment("not_assessed"),
                                  model_prediction=None, multiclass_prediction=None)
        return self.snapshot()

    def _connections(self) -> list[dict]:
        import psutil
        result, names = [], {}
        for connection in psutil.net_connections(kind="inet"):
            if not connection.raddr or not connection.laddr:
                continue
            local_ip = connection.laddr.ip.split("%")[0]
            remote_ip = connection.raddr.ip.split("%")[0]
            if local_ip not in self._local_addresses or ipaddress.ip_address(remote_ip).is_loopback:
                continue
            pid = connection.pid
            if pid not in names:
                try:
                    names[pid] = psutil.Process(pid).name() if pid else "Unavailable"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    names[pid] = "Unavailable"
            result.append({"protocol": "TCP" if connection.type == socket.SOCK_STREAM else "UDP",
                           "local": f"{local_ip}:{connection.laddr.port}", "remote": f"{remote_ip}:{connection.raddr.port}",
                           "status": connection.status, "process": names[pid]})
        return result

    def _packet(self, packet):
        """Inspect headers only. Whole Packet objects are not retained."""
        from scapy.layers.inet import IP, TCP, UDP, ICMP
        from scapy.layers.inet6 import IPv6
        from scapy.layers.l2 import Ether, ARP
        ip = packet.getlayer(IP) or packet.getlayer(IPv6)
        ethernet, arp = packet.getlayer(Ether), packet.getlayer(ARP)
        own_ip = ip is not None and bool({ip.src.split("%")[0], ip.dst.split("%")[0]} & self._local_addresses)
        own_mac = ethernet is not None and bool({(ethernet.src or "").lower(), (ethernet.dst or "").lower()} & self._local_macs)
        own_arp = arp is not None and bool({arp.psrc, arp.pdst} & self._local_addresses)
        if not (own_ip or own_mac or own_arp):
            return
        length = len(packet)
        tcp, udp = packet.getlayer(TCP), packet.getlayer(UDP)
        protocol = "TCP" if tcp is not None else "UDP" if udp is not None else "ARP" if arp is not None else "ICMP" if packet.haslayer(ICMP) else "IPv6" if packet.haslayer(IPv6) else "IP" if ip is not None else "Ethernet"
        source = ip.src if ip is not None else arp.psrc if arp is not None else ethernet.src
        destination = ip.dst if ip is not None else arp.pdst if arp is not None else ethernet.dst
        flags = int(tcp.flags) if tcp is not None else 0
        syn_only = bool(flags & 2) and not bool(flags & 16)
        with self._lock:
            self._window["packets"] += 1
            self._window["bytes"] += length
            self._window["syn"] += int(syn_only)
            self._window["syn_ack"] += int(bool(flags & 2) and bool(flags & 16))
            self._window["ack"] += int(bool(flags & 16))
            self._window["syn_all"] += int(bool(flags & 2))
            self._window["fin"] += int(bool(flags & 1))
            self._window["rst"] += int(bool(flags & 4))
            self._window["psh"] += int(bool(flags & 8))
            self._window["urg"] += int(bool(flags & 32))
            if syn_only and tcp is not None:
                # Only track enough distinct ports to evaluate the explicit threshold.
                key = f"{source}>{destination}"
                if len(self._ports) < 1000 or key in self._ports:
                    if len(self._ports[key]) < 100:
                        self._ports[key].add(int(tcp.dport))
            recent = self._data["recent_packets"]
            recent.append({"time": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                           "source": source, "destination": destination, "length": length})
            del recent[:-40]

    def _begin_capture(self, interface: str):
        from scapy.all import AsyncSniffer, conf
        if platform.system() == "Windows":
            conf.use_pcap = True
            if not conf.use_pcap:
                raise RuntimeError("Npcap could not be loaded. Install Npcap, restart the app and check capture permissions.")
            selected = next((v for v in conf.ifaces.values() if v.name == interface or v.description == interface), None)
            if selected is None:
                # Scapy friendly names can differ; resolve by the adapter's own IP.
                selected = next((v for v in conf.ifaces.values() if v.ip in self._local_addresses), None)
            if selected is None:
                raise RuntimeError("Cannot match the selected adapter to a packet capture device")
        else:
            selected = interface
        self._capture_socket = conf.L2listen(iface=selected, promisc=False, filter="ip or ip6 or arp")
        ready = threading.Event()
        self._sniffer = AsyncSniffer(opened_socket=self._capture_socket, store=False,
                                     prn=self._packet, started_callback=ready.set)
        self._sniffer.start()
        if not ready.wait(timeout=3):
            raise RuntimeError("Packet capture did not start. Check Npcap and capture permissions.")

    def _run(self, interface: str, mode: str):
        import psutil
        try:
            if mode == "packets":
                self._begin_capture(interface)
            previous = psutil.net_io_counters(pernic=True)[interface]
            last = time.monotonic()
            period = 10 if mode == "packets" else 2
            with self._lock:
                self._data["state"] = "monitoring"
                if mode == "packets":
                    self._window = self._empty_window()
                    self._ports.clear()
                    last = time.monotonic()
            while not self._stop.wait(max(0, last + period - time.monotonic())):
                now = time.monotonic()
                if now - self._started > 3600:
                    break  # bounded one-hour session
                if mode == "packets" and not self._sniffer.running:
                    raise RuntimeError("Packet capture stopped unexpectedly; no health verdict is available")
                current = psutil.net_io_counters(pernic=True).get(interface)
                if current is None:
                    raise RuntimeError("The selected adapter disappeared")
                elapsed = max(now - last, .1)
                observed_at = datetime.now(timezone.utc).isoformat()
                observation_id = uuid.uuid4().hex
                binary = multiclass = None
                # Rotate promptly. Neither model inference nor OS process enumeration
                # may block the capture callback or extend a completed window.
                with self._lock:
                    if mode == "packets":
                        stats = {**self._window, "seconds": elapsed,
                                 "unique_ports_per_source": {k: len(v) for k, v in self._ports.items()}}
                        count, size = stats["packets"], stats["bytes"]
                        decision = assess_packets(stats)
                        self._ports.clear()
                        self._window = self._empty_window()
                connections = self._connections()
                if mode == "packets":
                    try:
                        binary = self._predict_window(stats) if count else None
                    except Exception as error:
                        binary = self._model_failure("Binary packet model", error)
                    if binary is not None:
                        binary.update(observation_id=observation_id, observed_at=observed_at)
                    try:
                        multiclass = self._predict_multiclass_window(stats, observation_id, observed_at)
                    except Exception as error:
                        multiclass = {**self._model_failure("Shared multiclass model", error),
                                      "observation_id": observation_id, "observed_at": observed_at}
                else:
                    count = max(0, current.packets_sent + current.packets_recv - previous.packets_sent - previous.packets_recv)
                    size = max(0, current.bytes_sent + current.bytes_recv - previous.bytes_sent - previous.bytes_recv)
                    decision = assess_connections(connections, count / elapsed)
                with self._lock:
                    if self._stop.is_set():
                        break
                    self._data.update(model_prediction=binary, multiclass_prediction=multiclass)
                    self._data["metrics"] = {"packets_per_second": count / elapsed, "bytes_per_second": size / elapsed,
                                             "packets_observed": self._data["metrics"]["packets_observed"] + count,
                                             "active_connections": len(connections)}
                    self._data["assessment"] = decision
                    self._last_observation = now
                    self._data["connections"] = connections[:100]
                    self._data["history"].append({"time": datetime.now(timezone.utc).isoformat(),
                                                   "packets_per_second": count / elapsed, "bytes_per_second": size / elapsed,
                                                   "status": decision["status"]})
                    del self._data["history"][:-120]
                previous, last = current, now
        except Exception as exc:
            with self._lock:
                self._data.update(state="error", error=f"Monitoring failed: {exc}",
                                  assessment=assessment("not_assessed"), model_prediction=None, multiclass_prediction=None)
        finally:
            try:
                if self._sniffer is not None and self._sniffer.running:
                    self._sniffer.stop(join=True)
            finally:
                if self._capture_socket is not None:
                    self._capture_socket.close()
                self._sniffer = self._capture_socket = None
                with self._lock:
                    if self._started is not None:
                        self._data["elapsed_seconds"] = round(time.monotonic() - self._started, 1)
                        self._started = None
                    self._data["running"] = False
                    self._data.update(assessment=assessment("not_assessed"), model_prediction=None, multiclass_prediction=None)
                    if self._data["state"] != "error":
                        self._data["state"] = "stopped"

    @staticmethod
    def _empty_window():
        return {"packets": 0, "bytes": 0, "syn": 0, "syn_ack": 0, "syn_all": 0,
                "ack": 0, "fin": 0, "rst": 0, "psh": 0, "urg": 0}

    def _predict_window(self, stats):
        if abs(stats.get("seconds", 10) - 10) > 1:
            return {"label": "Model skipped this delayed window", "available": False,
                    "limitations": "Packet capture and rule checks continue. This interval differs from the model's ten-second training window."}
        path = Path(__file__).resolve().parents[1] / "models/packet_live/bundle.json"
        if not path.is_file():
            return {"label": "Packet model unavailable", "available": False,
                    "limitations": "Only live behavior rules were evaluated; no model prediction was made."}
        try:
            if self._packet_model is None:
                from framework.detection.packet_xgboost import PacketXGBoost
                self._packet_model = PacketXGBoost(path)
            return {**self._packet_model.predict(stats), "observation_id": uuid.uuid4().hex,
                    "observed_at": datetime.now(timezone.utc).isoformat()}
        except Exception as error:
            return self._model_failure("Binary packet model", error)

    @staticmethod
    def _model_failure(name, error):
        logging.getLogger(__name__).warning("%s unavailable: %s", name, error)
        return {"available": False, "label": name + " unavailable", "error": str(error),
                "limitations": "This model capability failed. Packet observation and independent behavior rules continue; no substitute prediction was returned."}

    def _predict_multiclass_window(self, stats, observation_id, observed_at):
        if not self._detector or not stats.get("packets") or abs(stats.get("seconds", 0) - 10) > 1:
            return None
        try:
            if not self._detector.partial_input_supported:
                return None
            from framework.data.feature_schema import packet_vector
            from framework.detection.trained_xgboost import json_features
            values = packet_vector(stats, self._detector.feature_names)
            report = self._detector.analyze(json_features(values), {
                "kind": "Actual local packet observation", "mode": "packets", "synthetic": False,
                "observation_id": observation_id, "observed_at": observed_at,
                "interface": self._data["interface"], "observed_window_seconds": stats["seconds"]})
            return {**report, "observation_id": observation_id, "observed_at": observed_at,
                    "observed_window_seconds": stats["seconds"]}
        except Exception as error:
            return {**self._model_failure("Shared multiclass model", error),
                    "observation_id": observation_id, "observed_at": observed_at,
                    "observed_window_seconds": stats["seconds"]}
