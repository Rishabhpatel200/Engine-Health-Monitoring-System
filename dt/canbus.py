"""
Engine data acquisition over CAN.

Two interchangeable transports behind one interface:

  SocketCanBus   binds a real SocketCAN interface (``can0``/``vcan0``) using
                 raw AF_CAN sockets - no third-party stack required. Available
                 on Linux, which is where the edge computer runs.
  VirtualCanBus  an in-process loopback used on non-Linux hosts and in CI, with
                 identical framing so the parsing path is exercised either way.

Framing follows the J1939 style used by common aero-engine ECU/FADEC gateways:
29-bit extended identifiers carrying a PGN, 8-byte payloads, scaled unsigned
integers with an offset. The PGN map below is the twin's ICD.
"""

from __future__ import annotations

import platform
import struct
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

CAN_EFF_FLAG = 0x80000000
CAN_FRAME_FMT = "=IB3x8s"          # SocketCAN struct can_frame
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)


@dataclass
class Signal:
    name: str
    start_byte: int
    length: int          # bytes: 1 or 2
    scale: float
    offset: float
    unit: str


@dataclass
class PGNDef:
    pgn: int
    name: str
    rate_hz: float
    signals: List[Signal]


# --- Interface Control Document -------------------------------------------
PGN_MAP: Dict[int, PGNDef] = {
    0xF004: PGNDef(0xF004, "EEC1_SpeedLoad", 50, [
        Signal("rpm", 0, 2, 0.25, 0.0, "rpm"),
        Signal("throttle_pct", 2, 1, 0.4, 0.0, "%"),
        Signal("map_kpa", 3, 2, 0.01, 0.0, "kPa"),
        Signal("iat_c", 5, 1, 1.0, -40.0, "degC"),
        Signal("bus_v", 6, 2, 0.001, 0.0, "V"),
    ]),
    0xFEEE: PGNDef(0xFEEE, "EngineTemp1", 10, [
        Signal("coolant_temp_c", 0, 1, 1.0, -40.0, "degC"),
        Signal("oil_temp_c", 1, 2, 0.03125, -273.0, "degC"),
        Signal("cht1_c", 3, 2, 0.03125, -273.0, "degC"),
        Signal("cht2_c", 5, 2, 0.03125, -273.0, "degC"),
        Signal("coolant_valve_pct", 7, 1, 0.4, 0.0, "%"),
    ]),
    0xFEEF: PGNDef(0xFEEF, "EngineTemp2", 10, [
        Signal("cht3_c", 0, 2, 0.03125, -273.0, "degC"),
        Signal("cht4_c", 2, 2, 0.03125, -273.0, "degC"),
        Signal("oil_press_kpa", 4, 2, 0.05, 0.0, "kPa"),
        Signal("amb_press_kpa", 6, 2, 0.01, 0.0, "kPa"),
    ]),
    0xFEF2: PGNDef(0xFEF2, "FuelEconomy", 10, [
        Signal("fuel_flow_lph", 0, 2, 0.002, 0.0, "L/h"),
        Signal("alt_current_a", 2, 2, 0.002, -50.0, "A"),
        Signal("oat_c", 4, 1, 1.0, -80.0, "degC"),
        Signal("vib_rms_g", 5, 2, 0.0005, 0.0, "g"),
        Signal("prop_load_pct", 7, 1, 0.4, 0.0, "%"),
    ]),
    0xFDD1: PGNDef(0xFDD1, "VibrationOrders", 20, [
        Signal("vib_half_g", 0, 2, 0.0005, 0.0, "g"),
        Signal("vib_1x_g", 2, 2, 0.0005, 0.0, "g"),
        Signal("vib_broad_g", 4, 2, 0.0005, 0.0, "g"),
        Signal("knock_level", 6, 1, 0.5, 0.0, "%"),
    ]),
    0xFD0E: PGNDef(0xFD0E, "ExhaustTemp1", 20, [
        Signal("egt1_c", 0, 2, 0.05, 0.0, "degC"),
        Signal("egt2_c", 2, 2, 0.05, 0.0, "degC"),
        Signal("egt3_c", 4, 2, 0.05, 0.0, "degC"),
        Signal("egt4_c", 6, 2, 0.05, 0.0, "degC"),
    ]),
}

SIGNAL_INDEX: Dict[str, Tuple[int, Signal]] = {
    s.name: (p.pgn, s) for p in PGN_MAP.values() for s in p.signals}

SOURCE_ADDRESS = 0x00           # ECU
PRIORITY = 3


def build_can_id(pgn: int, priority: int = PRIORITY, sa: int = SOURCE_ADDRESS) -> int:
    return ((priority & 0x7) << 26) | ((pgn & 0x3FFFF) << 8) | (sa & 0xFF) | 0

def parse_can_id(can_id: int) -> Tuple[int, int, int]:
    can_id &= ~CAN_EFF_FLAG
    return ((can_id >> 26) & 0x7, (can_id >> 8) & 0x3FFFF, can_id & 0xFF)


def encode_pgn(pgn: int, values: Dict[str, Optional[float]]) -> bytes:
    """Pack one PGN payload. Unavailable signals are sent as all-ones, which is
    the J1939 'not available' convention the parser honours."""
    buf = bytearray(b"\xff" * 8)
    for sig in PGN_MAP[pgn].signals:
        v = values.get(sig.name)
        if v is None:
            continue
        raw = int(round((v - sig.offset) / sig.scale))
        maxv = (1 << (8 * sig.length)) - 1
        raw = max(0, min(maxv - 1, raw))       # reserve max for 'not available'
        buf[sig.start_byte:sig.start_byte + sig.length] = raw.to_bytes(sig.length, "little")
    return bytes(buf)


def decode_pgn(pgn: int, data: bytes) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    d = PGN_MAP.get(pgn)
    if d is None:
        return out
    for sig in d.signals:
        chunk = data[sig.start_byte:sig.start_byte + sig.length]
        if len(chunk) < sig.length:
            continue
        raw = int.from_bytes(chunk, "little")
        if raw == (1 << (8 * sig.length)) - 1:
            out[sig.name] = None               # signal not available
        else:
            out[sig.name] = raw * sig.scale + sig.offset
    return out


@dataclass
class CanFrame:
    can_id: int
    data: bytes
    timestamp: float

    @property
    def pgn(self) -> int:
        return parse_can_id(self.can_id)[1]


class BaseCanBus:
    """Common encode/decode logic; transports override send/recv."""

    def __init__(self):
        self._phase: Dict[int, float] = {p: 0.0 for p in PGN_MAP}
        self.tx_frames = 0
        self.rx_frames = 0
        self.rx_errors = 0

    def publish(self, values: Dict[str, Optional[float]], t_s: float) -> List[CanFrame]:
        """Emit every PGN that is due at mission time `t_s`."""
        frames: List[CanFrame] = []
        for pgn, d in PGN_MAP.items():
            period = 1.0 / d.rate_hz
            if t_s + 1e-9 >= self._phase[pgn]:
                self._phase[pgn] = t_s + period
                f = CanFrame(build_can_id(pgn), encode_pgn(pgn, values), t_s)
                self.send(f)
                frames.append(f)
                self.tx_frames += 1
        return frames

    @staticmethod
    def frames_to_signals(frames: List[CanFrame]) -> Dict[str, Optional[float]]:
        merged: Dict[str, Optional[float]] = {}
        for f in frames:
            merged.update(decode_pgn(f.pgn, f.data))
        return merged

    def send(self, frame: CanFrame) -> None:      # pragma: no cover - override
        raise NotImplementedError

    def recv(self, timeout: float = 0.0) -> List[CanFrame]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def stats(self) -> dict:
        return {"tx": self.tx_frames, "rx": self.rx_frames, "errors": self.rx_errors,
                "transport": type(self).__name__}


class VirtualCanBus(BaseCanBus):
    """In-process loopback with the same framing as the real bus."""

    def __init__(self, bus_load_limit: int = 4000):
        super().__init__()
        self._q: List[CanFrame] = []
        self.bus_load_limit = bus_load_limit

    def send(self, frame: CanFrame) -> None:
        self._q.append(frame)
        if len(self._q) > self.bus_load_limit:      # emulate a full RX FIFO
            self._q.pop(0)
            self.rx_errors += 1

    def recv(self, timeout: float = 0.0) -> List[CanFrame]:
        got, self._q = self._q, []
        self.rx_frames += len(got)
        return got


class SocketCanBus(BaseCanBus):
    """Raw AF_CAN/SOCK_RAW binding - Linux only, no external dependencies."""

    def __init__(self, channel: str = "vcan0"):
        super().__init__()
        import socket
        if not hasattr(socket, "AF_CAN"):
            raise RuntimeError(
                f"SocketCAN unavailable on {platform.system()}; "
                "use VirtualCanBus or run the edge node on Linux")
        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.bind((channel,))
        self.sock.setblocking(False)
        self.channel = channel

    def send(self, frame: CanFrame) -> None:
        payload = struct.pack(CAN_FRAME_FMT, frame.can_id | CAN_EFF_FLAG,
                              len(frame.data), frame.data)
        try:
            self.sock.send(payload)
        except OSError:
            self.rx_errors += 1

    def recv(self, timeout: float = 0.0) -> List[CanFrame]:
        import select
        frames: List[CanFrame] = []
        deadline = time.time() + timeout
        while True:
            r, _, _ = select.select([self.sock], [], [], max(0.0, deadline - time.time()))
            if not r:
                break
            try:
                buf = self.sock.recv(CAN_FRAME_SIZE)
            except OSError:
                self.rx_errors += 1
                break
            can_id, dlc, data = struct.unpack(CAN_FRAME_FMT, buf)
            frames.append(CanFrame(can_id & ~CAN_EFF_FLAG, data[:dlc], time.time()))
            self.rx_frames += 1
            if timeout <= 0:
                break
        return frames

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def open_bus(channel: str = "vcan0", prefer_socketcan: bool = True) -> BaseCanBus:
    """Open the best available transport, falling back to the loopback."""
    if prefer_socketcan:
        try:
            return SocketCanBus(channel)
        except Exception:
            pass
    return VirtualCanBus()


def bus_load_percent(bitrate: int = 500_000) -> float:
    """Steady-state bus utilisation of the ICD above: 29-bit ID + 8 data bytes
    is ~128 bits on the wire including stuffing and inter-frame space."""
    bits_per_s = sum(d.rate_hz * 128 for d in PGN_MAP.values())
    return 100.0 * bits_per_s / bitrate
