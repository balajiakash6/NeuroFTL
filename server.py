"""
Adaptive Workload-Aware FTL Telemetry Server
Provides Server-Sent Events (SSE) stream and REST API for real-time dashboard.
AI-SSD Firmware Digital Twin Hackathon Project
Authors: Balaji Akash S, Karthick P, Padmanabhan S, Benit D Binu
"""

import json
import os
import sys
import time
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse
from models import HardwareSpecs
from simulator import DualWorkloadSimulator


class FTLSimulationServer:
    def __init__(self, port: int = 8000):
        self.port = port
        self.specs = HardwareSpecs()
        self.simulator = DualWorkloadSimulator(self.specs)
        self.sim_lock = threading.Lock()
        self.static_dir = os.path.dirname(os.path.abspath(__file__))

    def get_handler_class(self):
        sim = self.simulator
        sim_lock = self.sim_lock
        static_dir = self.static_dir
        specs = self.specs

        class TelemetryRequestHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=static_dir, **kwargs)

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                if parsed.path in ("/telemetry", "/api/telemetry"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache, no-transform")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()

                    try:
                        while True:
                            with sim_lock:
                                tick = sim.tick()
                                data = json.dumps({
                                    "tick_id": tick.tick_id,
                                    "timestamp": tick.timestamp,
                                    "workload_phase": tick.workload_phase,
                                    "standard_ftl": tick.standard_ftl.__dict__,
                                    "heuristic_v1_ftl": tick.heuristic_v1_ftl.__dict__ if tick.heuristic_v1_ftl else None,
                                    "adaptive_ftl": tick.adaptive_ftl.__dict__,
                                    "gpu_stall_reduction_pct": tick.gpu_stall_reduction_pct,
                                    "accumulated_gpu_stall_saved_ms": tick.accumulated_gpu_stall_saved_ms,
                                    "io_blender_active": tick.io_blender_active
                                })
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                            self.wfile.flush()
                            time.sleep(0.12) # ~8 ticks/second smooth streaming
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
                        self.close_connection = True
                        return
                    except Exception:
                        self.close_connection = True
                        return
                elif parsed.path == "/api/tick":
                    with sim_lock:
                        tick = sim.tick()
                        data = {
                            "tick_id": tick.tick_id,
                            "timestamp": tick.timestamp,
                            "workload_phase": tick.workload_phase,
                            "standard_ftl": tick.standard_ftl.__dict__,
                            "heuristic_v1_ftl": tick.heuristic_v1_ftl.__dict__ if tick.heuristic_v1_ftl else None,
                            "adaptive_ftl": tick.adaptive_ftl.__dict__,
                            "gpu_stall_reduction_pct": tick.gpu_stall_reduction_pct,
                            "accumulated_gpu_stall_saved_ms": tick.accumulated_gpu_stall_saved_ms,
                            "io_blender_active": tick.io_blender_active
                        }
                    self._send_json(data)

                elif parsed.path == "/api/status":
                    with sim_lock:
                        status_data = {
                            "status": "online",
                            "tick_counter": sim.tick_counter,
                            "current_phase": sim.current_phase,
                            "specs": specs.__dict__
                        }
                    self._send_json(status_data)
                elif parsed.path == "/" or parsed.path == "":
                    self.path = "/index.html"
                    return super().do_GET()
                else:
                    return super().do_GET()

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                content_len = int(self.headers.get('Content-Length', 0))
                body_bytes = self.rfile.read(content_len) if content_len > 0 else b'{}'
                try:
                    payload = json.loads(body_bytes.decode('utf-8'))
                except Exception:
                    payload = {}

                if parsed.path == "/api/workload":
                    phase = payload.get("phase", "BASELINE")
                    lock = payload.get("lock", True)
                    with sim_lock:
                        sim.set_workload_phase(phase, lock=lock)
                    self._send_json({"success": True, "phase": phase, "lock": lock})
                elif parsed.path == "/api/config":
                    with sim_lock:
                        if "seq_threshold" in payload:
                            specs.seq_score_threshold = float(payload["seq_threshold"])
                        if "read_disturb_threshold" in payload:
                            specs.read_disturb_threshold = int(payload["read_disturb_threshold"])
                        if "window_size" in payload:
                            specs.seq_window_size = int(payload["window_size"])
                    self._send_json({"success": True, "specs": specs.__dict__})
                elif parsed.path == "/api/reset":
                    with sim_lock:
                        sim.__init__(specs)
                    self._send_json({"success": True, "message": "Simulator reset"})
                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_json(self, data_dict):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data_dict).encode("utf-8"))

            def log_message(self, format, *args):
                # Suppress verbose SSE tick & poll logs
                try:
                    msg = format % args if args else format
                    if "/telemetry" not in str(msg) and "/api/tick" not in str(msg):
                        super().log_message(format, *args)
                except Exception:
                    pass

        return TelemetryRequestHandler

    def start(self):
        handler_class = self.get_handler_class()
        server = ThreadingHTTPServer(('0.0.0.0', self.port), handler_class)
        print(f"[Adaptive-FDP Server] Serving on http://localhost:{self.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = FTLSimulationServer(port=port)
    server.start()
