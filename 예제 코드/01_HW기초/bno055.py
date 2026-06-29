import os
import struct
import time
import fcntl

_I2C_SLAVE = 0x0703

class BNO055:
    def __init__(self, bus: int = 7, addr: int = 0x29):
        self._addr = addr
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        fcntl.ioctl(self.fd, _I2C_SLAVE, self._addr)
        time.sleep(1.0)
        self._init_sensor()

    def _write(self, reg: int, value: int) -> None:
        os.write(self.fd, bytes([reg, value]))
        time.sleep(0.01)

    def _read(self, reg: int, length: int) -> bytes:
        os.write(self.fd, bytes([reg]))
        return os.read(self.fd, length)

    def _init_sensor(self) -> None:
        chip_id = self._read(0x00, 1)[0]
        if chip_id != 0xA0:
            raise RuntimeError(f"BNO055 감지 실패 (CHIP_ID: 0x{chip_id:02X})")

        self._write(0x3D, 0x00)  # CONFIG 모드
        time.sleep(0.05)

        self._write(0x3F, 0x20)  # 소프트 리셋
        time.sleep(0.65)

        self._write(0x3D, 0x0C)  # NDOF 모드
        time.sleep(0.02)

    @property
    def status(self) -> int:
        """센서 상태 레지스터 값 반환"""
        return self._read(0x39, 1)[0]

    @property
    def euler(self) -> tuple[float, float, float]:
        """(Yaw, Pitch, Roll) 단위: degree"""
        data = self._read(0x1A, 6)
        yaw   = struct.unpack_from('<h', data, 0)[0] / 16.0
        pitch = struct.unpack_from('<h', data, 2)[0] / 16.0
        roll  = struct.unpack_from('<h', data, 4)[0] / 16.0
        return yaw, pitch, roll

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

