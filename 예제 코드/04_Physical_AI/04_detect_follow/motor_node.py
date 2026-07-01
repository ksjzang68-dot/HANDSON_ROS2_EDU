"""
Motor Node  (2단계 — YOLO 추적)
================================
역할
  - /yolo/detections 구독 → rank=1 객체를 추적
  - /imu/yaw         구독 → 현재 yaw 유지 (조향 보조)
  - /odom            발행 → (x, y, yaw) 위치 (gui_node 맵에 표시)
  - /motor/phase     발행 → 상태 (idle / tracking / arrived)

FSM
  idle      : 감지 없음 or 도착 후 정지
  tracking  : rank=1 객체 추적 중 (조향 PID + 전진 PID)
  arrived   : 박스 높이 ≥ 화면 높이 × STOP_RATIO → 정지

PID 조향
  error_x  = (박스 중심 x - 화면 중심 x) / 화면 너비
             → 양수: 오른쪽 치우침 → 오른쪽 회전
  steer    = KP_steer × error_x   (단순 P 제어로도 충분)

PID 전진
  error_h  = STOP_RATIO - (box_h / frame_h)
             → 양수: 아직 멀다 → 전진
             → 0 이하: 도착 → 정지

오도메트리
  엔코더 델타 → dist_mm 적분  (1단계와 동일 수식)
  dx = dist * sin(yaw_rad)
  dy = dist * cos(yaw_rad)
"""

import math
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Pose2D

try:
    from HandsON_BuildHat_API import Motor
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False

# ── 상수 ──────────────────────────────────────────────────────────

WHEEL_CIRC_MM  = 298.45      # 바퀴 둘레 (mm)
STOP_RATIO     = 0.70        # 박스높이 / 화면높이 ≥ 이 값이면 도착

# ── PID 게인 ──────────────────────────────────────────────────────

# 조향: error = (cx - frame_cx) / frame_w  → [-0.5, +0.5]
STEER_KP = 80.0
STEER_KI = 0.0
STEER_KD = 5.0

# 전진: error = STOP_RATIO - (box_h / frame_h)  → [0, ~0.7]
FWD_KP   = 80.0
FWD_KI   = 0.0
FWD_KD   = 3.0

# 속도 클램프
FWD_MAX   = 60
FWD_MIN   = 15
STEER_MAX = 40

# ── PID ──────────────────────────────────────────────────────────

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev = 0.0

    def compute(self, error: float, dt: float) -> float:
        self._i   += error * dt
        d          = (error - self._prev) / dt if dt > 0 else 0.0
        self._prev = error
        return self.kp * error + self.ki * self._i + self.kd * d

    def reset(self):
        self._i    = 0.0
        self._prev = 0.0


# ── Motor Node ────────────────────────────────────────────────────

class MotorNode(Node):

    def __init__(self):
        super().__init__('motor_node')

        self.declare_parameter('control_hz', 20.0)
        hz       = self.get_parameter('control_hz').value
        self._dt = 1.0 / hz

        # ── 하드웨어 ─────────────────────────────────────────────
        if HW_AVAILABLE:
            self.left_m  = Motor('E')
            self.right_m = Motor('F')
            self.get_logger().info('SpikePI 초기화 완료')
        else:
            self.left_m = self.right_m = None
            self._sim_dist = 0.0
            self.get_logger().warn('SpikePI 없음 → 시뮬레이션 모드')

        # ── 상태 ─────────────────────────────────────────────────
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0          # IMU 에서 갱신

        self.phase      = 'idle'
        self._target    = None  # 최신 rank=1 감지 dict
        self._no_det_cnt = 0    # 감지 없는 연속 tick 수

        # ── 엔코더 베이스 ────────────────────────────────────────
        self._enc_l_base = 0.0
        self._enc_r_base = 0.0

        # ── PID ──────────────────────────────────────────────────
        self.steer_pid = PID(STEER_KP, STEER_KI, STEER_KD)
        self.fwd_pid   = PID(FWD_KP,   FWD_KI,   FWD_KD)

        # ── ROS2 ─────────────────────────────────────────────────
        self.sub_yaw  = self.create_subscription(
            Float32, '/imu/yaw',       self._cb_yaw,  10)
        self.sub_det  = self.create_subscription(
            String,  '/yolo/detections', self._cb_det, 10)

        self.pub_odom  = self.create_publisher(Pose2D, '/odom',        10)
        self.pub_phase = self.create_publisher(String, '/motor/phase', 10)

        self.timer = self.create_timer(self._dt, self._tick)
        self.get_logger().info(f'motor_node 시작 ({hz}Hz)')

    # ── 콜백 ─────────────────────────────────────────────────────

    def _cb_yaw(self, msg: Float32):
        self.yaw = msg.data

    def _cb_det(self, msg: String):
        try:
            dets = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        # rank=1 인 첫 번째 객체만 사용
        rank1 = next((d for d in dets if d.get('rank') == 1), None)

        if rank1 is None:
            self._no_det_cnt += 1
            # 10tick(~0.5s) 이상 감지 없으면 idle
            if self._no_det_cnt > 10:
                self._target = None
        else:
            self._no_det_cnt = 0
            self._target = rank1
            # 새 목표 감지 시 arrived 상태 해제
            if self.phase == 'idle':
                self.phase = 'tracking'
                self.steer_pid.reset()
                self.fwd_pid.reset()
                self._reset_enc()
                self.get_logger().info('추적 시작')

    # ── 제어 tick ─────────────────────────────────────────────────

    def _tick(self):

        # 오도메트리 갱신 (항상)
        step_mm = self._read_enc_delta()
        if step_mm > 0:
            rad     = math.radians(self.yaw)
            self.x += step_mm * math.sin(rad)
            self.y += step_mm * math.cos(rad)

        if self.phase == 'idle' or self.phase == 'arrived':
            self._hw_stop()
            self._pub()
            return

        if self._target is None:
            # 감지 없음 → 정지 대기
            self._hw_stop()
            self.phase = 'idle'
            self._pub()
            return

        d        = self._target
        frame_h  = d.get('frame_h', 480)
        frame_w  = d.get('frame_w', 640)
        box_h    = d['box_h']
        cx       = d['cx']

        # ── 도착 판정 ─────────────────────────────────────────────
        h_ratio = box_h / frame_h
        if h_ratio >= STOP_RATIO:
            self._hw_stop()
            self.phase = 'arrived'
            self.get_logger().info(
                f'목표 도착! box_h/frame_h = {h_ratio:.2f}')
            self._pub()
            return

        # ── 조향 오차 ─────────────────────────────────────────────
        #   error_x: [-0.5, +0.5]  양수=오른쪽, 음수=왼쪽
        frame_cx  = frame_w / 2.0
        error_x   = (cx - frame_cx) / frame_w

        # ── 전진 오차 ─────────────────────────────────────────────
        #   error_h: [0, ~0.7]  클수록 멀다
        error_h   = STOP_RATIO - h_ratio   # 양수여야 전진

        # ── PID 계산 ─────────────────────────────────────────────
        steer  = self.steer_pid.compute(error_x, self._dt)
        steer  = max(-STEER_MAX, min(STEER_MAX, steer))

        forward = self.fwd_pid.compute(error_h, self._dt)
        forward = max(FWD_MIN, min(FWD_MAX, forward))

        # 물체가 화면 중앙에서 많이 벗어나면 직진 속도 감소
        #   |error_x| > 0.3 → forward *= 0.3
        center_ratio = max(0.0, 1.0 - abs(error_x) / 0.3)
        forward     *= center_ratio

        left_spd  = forward + steer
        right_spd = forward - steer
        self._hw_drive(left_spd, right_spd)

        self._pub()

    # ── 발행 ─────────────────────────────────────────────────────

    def _pub(self):
        odom       = Pose2D()
        odom.x     = self.x
        odom.y     = self.y
        odom.theta = self.yaw
        self.pub_odom.publish(odom)

        p      = String()
        p.data = self.phase
        self.pub_phase.publish(p)

    # ── 하드웨어 래퍼 ────────────────────────────────────────────

    def _hw_drive(self, left: float, right: float):
        left  = max(-100, min(100, int(left)))
        right = max(-100, min(100, int(right)))
        if self.left_m:
            self.left_m.start(-left)
            self.right_m.start(right)
        else:
            # 시뮬: 평균 속도로 가상 이동
            avg = (abs(left) + abs(right)) / 2.0
            self._sim_dist += avg * 0.01 * (WHEEL_CIRC_MM / 100.0)

    def _hw_stop(self):
        if self.left_m:
            self.left_m.stop()
            self.right_m.stop()

    def _reset_enc(self):
        if self.left_m:
            self._enc_l_base = self.left_m.get_degrees_counted()
            self._enc_r_base = self.right_m.get_degrees_counted()
        else:
            self._sim_dist = 0.0

    def _read_enc_delta(self) -> float:
        if self.left_m:
            l = self.left_m.get_degrees_counted()  - self._enc_l_base
            r = self.right_m.get_degrees_counted() - self._enc_r_base
            self._enc_l_base = self.left_m.get_degrees_counted()
            self._enc_r_base = self.right_m.get_degrees_counted()
            return ((abs(l) + abs(r)) / 2.0) / 360.0 * WHEEL_CIRC_MM
        else:
            d = self._sim_dist
            self._sim_dist = 0.0
            return d

    def destroy_node(self):
        self._hw_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
