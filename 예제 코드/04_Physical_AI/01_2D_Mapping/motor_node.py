"""
Motor Node
==========
역할
  - /imu/yaw     구독 → 현재 yaw 유지
  - /goal_pose   구독 → (x_mm, y_mm) 수신 → 회전 후 직진
  - /odom        퍼블리시 → (x, y, yaw, phase) 20Hz

- 구독(Subscription) + 퍼블리시(Publisher) 동시 운용
- FSM(상태머신) : idle → turning → driving → idle
- 오도메트리 핵심 수식 :
    dx = dist * sin(yaw_rad)
    dy = dist * cos(yaw_rad)

============================================
turning  : 출발 전 1회만 — 목표 방향 정렬
driving  : phase 전환 없이 PID 조향으로 직선 유지
"""

import math
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

WHEEL_CIRC_MM    = 298.45
FIELD_MM         = 1000.0
ARRIVE_THRESH_MM = 10.0
TURN_THRESH_DEG  = 5.0      # 최초 회전 완료 판정

# ── PID 게인 ──────────────────────────────────────────────────────

TURN_KP  = 1.2
TURN_KI  = 0.0
TURN_KD  = 0.3

DRIVE_KP = 0.8
DRIVE_KI = 0.0
DRIVE_KD = 0.2

# ── 속도 ──────────────────────────────────────────────────────────

DRIVE_BASE = 40
TURN_MAX   = 60
TURN_MIN   = 18
STEER_MAX  = 35

# 오차 이 각도 이상이면 직진 속도 0 (사실상 제자리 회전)
FORWARD_CUTOFF_DEG = 30.0


class PIDController:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_err = 0.0

    def compute(self, error: float, dt: float) -> float:
        self._integral += error * dt
        derivative      = (error - self._prev_err) / dt if dt > 0 else 0.0
        self._prev_err  = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0


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
            self.get_logger().warn('SpikePI 없음 → 시뮬레이션 모드')

        # ── 상태 ─────────────────────────────────────────────────
        self.x   = 100.0
        self.y   = 100.0
        self.yaw = 0.0

        self.target_x = None
        self.target_y = None
        self.phase    = 'idle'

        self._enc_l_base = 0.0
        self._enc_r_base = 0.0
        self._sim_dist   = 0.0

        # ── PID ──────────────────────────────────────────────────
        self.turn_pid  = PIDController(TURN_KP,  TURN_KI,  TURN_KD)
        self.drive_pid = PIDController(DRIVE_KP, DRIVE_KI, DRIVE_KD)

        # ── ROS2 ─────────────────────────────────────────────────
        self.sub_yaw  = self.create_subscription(
            Float32, '/imu/yaw',   self._cb_yaw,  10)
        self.sub_goal = self.create_subscription(
            Pose2D,  '/goal_pose', self._cb_goal, 10)

        self.pub_odom  = self.create_publisher(Pose2D, '/odom',        10)
        self.pub_phase = self.create_publisher(String, '/motor/phase', 10)

        self.timer = self.create_timer(self._dt, self._control_tick)
        self.get_logger().info(f'motor_node 시작 ({hz}Hz)')

    # ── 구독 콜백 ─────────────────────────────────────────────────

    def _cb_yaw(self, msg: Float32):
        self.yaw = msg.data

    def _cb_goal(self, msg: Pose2D):
        self.target_x = max(0.0, min(FIELD_MM, msg.x))
        self.target_y = max(0.0, min(FIELD_MM, msg.y))
        self.phase    = 'turning'   # 새 목표 → 항상 최초 회전부터
        self.turn_pid.reset()
        self.drive_pid.reset()
        self._reset_encoder()
        self.get_logger().info(
            f'목표: ({self.target_x:.0f}, {self.target_y:.0f}) mm')

    # ── 제어 tick ─────────────────────────────────────────────────

    def _control_tick(self):
        if self.phase == 'idle' or self.target_x is None:
            self._hw_stop()
            self._publish_odom()
            return

        dx   = self.target_x - self.x
        dy   = self.target_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < ARRIVE_THRESH_MM:
            self.x, self.y = self.target_x, self.target_y
            self.phase = 'idle'
            self._hw_stop()
            self.get_logger().info(
                f'도착: ({self.x:.0f}, {self.y:.0f}) mm')
            self._publish_odom()
            return

        target_yaw = math.degrees(math.atan2(dx, dy)) % 360
        err        = _angle_diff(self.yaw, target_yaw)

        # ── TURNING : 출발 전 1회만 ──────────────────────────────
        if self.phase == 'turning':
            if abs(err) <= TURN_THRESH_DEG:
                self._hw_stop()
                self.turn_pid.reset()
                self.drive_pid.reset()
                self._reset_encoder()
                self.phase = 'driving'
                self.get_logger().info('방향 정렬 완료 → 주행 시작')
            else:
                output = self.turn_pid.compute(err, self._dt)
                spd    = max(TURN_MIN, min(TURN_MAX, abs(output)))
                if output > 0:
                    self._hw_drive( spd, -spd)
                else:
                    self._hw_drive(-spd,  spd)

        # ── DRIVING : phase 전환 없이 PID 조향만 ─────────────────
        elif self.phase == 'driving':
            # 오차에 따라 직진 속도 자동 감소
            #   err=  0도 → forward = DRIVE_BASE (풀 직진)
            #   err= 15도 → forward = DRIVE_BASE × 0.5
            #   err= 30도 → forward = 0 (사실상 제자리 회전)
            forward_ratio = max(0.0, 1.0 - abs(err) / FORWARD_CUTOFF_DEG)
            forward       = DRIVE_BASE * forward_ratio

            steer = self.drive_pid.compute(err, self._dt)
            steer = max(-STEER_MAX, min(STEER_MAX, steer))

            self._hw_drive(forward + steer, forward - steer)

            step_mm = self._read_encoder_delta()
            if step_mm > 0:
                rad     = math.radians(self.yaw)
                self.x += step_mm * math.sin(rad)
                self.y += step_mm * math.cos(rad)
                self.x  = max(0.0, min(FIELD_MM, self.x))
                self.y  = max(0.0, min(FIELD_MM, self.y))

        self._publish_odom()

    # ── 오도메트리 퍼블리시 ───────────────────────────────────────

    def _publish_odom(self):
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
        avg = (abs(left) + abs(right)) / 2.0
        self._sim_dist += avg * 0.01 * (WHEEL_CIRC_MM / 100.0)

    def _hw_stop(self):
        if self.left_m:
            self.left_m.stop()
            self.right_m.stop()

    def _reset_encoder(self):
        if self.left_m:
            self._enc_l_base = self.left_m.get_degrees_counted()
            self._enc_r_base = self.right_m.get_degrees_counted()
        self._sim_dist = 0.0

    def _read_encoder_delta(self) -> float:
        if self.left_m:
            l       = self.left_m.get_degrees_counted()  - self._enc_l_base
            r       = self.right_m.get_degrees_counted() - self._enc_r_base
            dist_mm = ((abs(l) + abs(r)) / 2.0) / 360.0 * WHEEL_CIRC_MM
            self._enc_l_base = self.left_m.get_degrees_counted()
            self._enc_r_base = self.right_m.get_degrees_counted()
            return dist_mm
        else:
            d = self._sim_dist
            self._sim_dist = 0.0
            return d

    def destroy_node(self):
        self._hw_stop()
        super().destroy_node()


# ── 헬퍼 ──────────────────────────────────────────────────────────

def _angle_diff(current: float, target: float) -> float:
    return (target - current + 180) % 360 - 180


# ── 엔트리포인트 ──────────────────────────────────────────────────

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