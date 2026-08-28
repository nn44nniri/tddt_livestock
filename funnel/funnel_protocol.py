from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SensorPacket:
    timestamp: str
    indoor_temp_c: float
    indoor_rh_pct: float
    indoor_air_speed_m_s: float = 0.25
    indoor_co2_ppm: float = 900.0
    indoor_nh3_ppm: float = 2.0
    indoor_h2o_g_m3: float = 8.0
    lora_rssi: float = -70.0
    packet_status: str = "SIMULATED"

    def to_dict(self) -> dict:
        return asdict(self)

class FunnelBase:
    def read_sensor_packet(self) -> dict:
        now = datetime.utcnow().replace(microsecond=0).isoformat()
        return SensorPacket(timestamp=now, indoor_temp_c=18.0, indoor_rh_pct=60.0).to_dict()

    def send_command(self, command: dict) -> dict:
        return {"status": "ACK", "command": dict(command)}
