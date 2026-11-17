"""Feetech STS serial-bus servo driver (the servos in the kit).

Packet layout (Feetech SCS/STS protocol, same family LeRobot uses for the
SO-100 arm):

    0xFF 0xFF  ID  LEN  INSTR  PARAM...  CHECKSUM
    CHECKSUM = ~(ID + LEN + INSTR + PARAMS) & 0xFF

We only need three instructions: PING (1), WRITE (3) and SYNC_WRITE (0x83).
SYNC_WRITE is what makes a 16-servo humanoid possible at 50 Hz: one packet
addresses every servo, so the whole body moves in one bus transaction instead
of 16 round trips.

The register map below is for STS3215 (the common hobby humanoid servo).
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass

try:                                  # pyserial is optional (import on board only)
    import serial
except Exception:                      # pragma: no cover
    serial = None

HEADER = b"\xff\xff"
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_WRITE = 0x83

ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_GOAL_SPEED = 46
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_LOAD = 60
ADDR_PRESENT_TEMP = 63

BROADCAST = 0xFE


def checksum(parts: bytes) -> int:
    return (~sum(parts)) & 0xFF


def build_packet(servo_id: int, instr: int, params: bytes) -> bytes:
    body = bytes([servo_id, len(params) + 2, instr]) + params
    return HEADER + body + bytes([checksum(body)])


def goal_position_packet(servo_id: int, pos: int, speed: int = 0, acc: int = 0) -> bytes:
    params = bytes([ADDR_GOAL_POSITION, 2]) + struct.pack("<H", max(0, min(1000, pos)))
    return build_packet(servo_id, INST_WRITE, params)


def sync_write_positions(ids: list[int], positions: list[int], speed: int = 3000) -> bytes:
    """One packet that moves every joint: [addr][len][id0][p0l][p0h][id1]..."""
    params = bytearray([ADDR_GOAL_POSITION, 2])
    for sid, pos in zip(ids, positions):
        params.append(sid & 0xFF)
        params += struct.pack("<H", max(0, min(1000, int(pos))))
    return build_packet(BROADCAST, INST_SYNC_WRITE, bytes(params))


@dataclass
class ServoState:
    position: int
    load: int
    temperature: int


class ServoBus:
    """Blocking serial wrapper with the handful of calls we need."""

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 1_000_000,
                 timeout: float = 0.02):
        if serial is None:
            raise RuntimeError("pyserial missing: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.ser.reset_input_buffer()

    def close(self) -> None:
        self.ser.close()

    def _read_status(self) -> bytes:
        head = self.ser.read(4)
        if len(head) < 4:
            return b""
        n = head[3]
        return head + self.ser.read(n)

    def ping(self, servo_id: int) -> bool:
        self.ser.write(build_packet(servo_id, INST_PING, b""))
        return len(self._read_status()) > 0

    def write_position(self, servo_id: int, pos: int) -> None:
        self.ser.write(goal_position_packet(servo_id, pos))

    def sync_write(self, ids: list[int], positions: list[int]) -> None:
        self.ser.write(sync_write_positions(ids, positions))

    def read_position(self, servo_id: int) -> int | None:
        params = bytes([ADDR_PRESENT_POSITION, 2])
        self.ser.write(build_packet(servo_id, INST_READ, params))
        resp = self._read_status()
        if len(resp) < 8:
            return None
        return struct.unpack("<H", resp[5:7])[0]

    def torque(self, servo_id: int, on: bool) -> None:
        self.ser.write(build_packet(servo_id, INST_WRITE,
                                    bytes([ADDR_TORQUE_ENABLE, 1, 1 if on else 0])))


class SimBus:
    """Same interface, no hardware: used by tests and by `--dry-run`."""

    def __init__(self, n_joints: int = 16):
        self.state = [500] * n_joints
        self.frames = 0

    def sync_write(self, ids: list[int], positions: list[int]) -> None:
        for sid, pos in zip(ids, positions):
            if 0 <= sid - 1 < len(self.state):
                self.state[sid - 1] = int(pos)
        self.frames += 1

    def close(self) -> None:
        pass


def self_test(port: str = "/dev/ttyUSB0", n: int = 16) -> int:
    bus = ServoBus(port)
    found = [i for i in range(1, n + 1) if bus.ping(i)]
    print(f"found servos: {found}")
    for i in found:
        bus.torque(i, True)
    bus.close()
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(self_test())
