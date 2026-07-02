#!/usr/bin/env python3
"""
mission_node.py
===============
역할:
  - /agv/detection 구독 → lock 선택 → 조향/접근/grab/navigate/release 순차 실행
  - AGV 위치·자세·페이즈를 /agv/state 에 publish (Visualizer가 구독)
  - 모터(HandsON_BuildHat_API) / IMU(BNO055) 직접 제어

Subscribe :
  /agv/detection  (std_msgs/String)  ← yolo_node 에서 수신

Publish   :
  /agv/state      (std_msgs/String)  ← JSON 직렬화 상태 (Visualizer용)

JSON 구조 (/agv/state):
  {
    "x": 0.0, "y": 0.0, "yaw": 0.0, "distance": 0.0,
    "speed_l": 0.0, "speed_r": 0.0,
    "phase": "tracking",
    "dest_label": "", "dest_x": 0.0, "dest_y": 0.0,
    "cycle": 0,
    "grabbed_labels": [],
    "detection": {              ← 최신 detection snapshot (Visualizer용)
      "detected": false,
      "cx": 0, "cy": 0,
      "confidence": 0.0,
      "label": "",
      "steer": 0.0,
      "locked": false,
      "lock_label": "",
      "all_boxes": []
    }
  }
"""

import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from agv_tracker.config import (
    WHEEL_DIAMETER_MM, FIELD_MM,
    BASE_SPEED, STEER_GAIN, STEER_D_GAIN, MAX_STEER, DEAD_ZONE,
    LOST_TIMEOUT, YAW_KP,
    CLOSE_Y_THRESH_MAP, CLOSE_Y_THRESH_DEFAULT, STABLE_TRACK_SEC,
    APPROACH_DIST_MM, APPROACH_SPEED,
    LOCK_STABLE_SEC, LOCK_POS_TOLERANCE,
    DESTINATION_MAP,
    NAV_SPEED, NAV_TURN_SPEED, NAV_ARRIVE_THRESH, NAV_YAW_KP,
    RELEASE_DURATION, RELEASE_SPEED,
    TOTAL_CYCLES,
    SPIN_180_SPEED,
    RETURN_SPEED, RETURN_TURN_SPEED, RETURN_ARRIVE_THRESH, RETURN_ORIGIN,
    CAM_W,
)

# ── 하드웨어 임포트 (실제 로봇에서만 사용 가능) ─────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from HandsON_BuildHat_API import Motor
    from bno055 import BNO055
    _HW_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] 하드웨어 임포트 실패: {e}")
    _HW_AVAILABLE = False


class MissionNode(Node):
    """AGV 미션 제어 노드."""

    def __init__(self):
        super().__init__("mission_node")

        # ── 파라미터 ───────────────────────────────────────────────
        self.declare_parameter("hw_enabled", _HW_AVAILABLE)
        self._hw = self.get_parameter("hw_enabled").value

        # ── 하드웨어 초기화 ────────────────────────────────────────
        if self._hw:
            self._lm  = Motor("E")
            self._rm  = Motor("F")
            self._imu = BNO055()
            self.get_logger().info("하드웨어 초기화 완료 (모터 E/F, IMU BNO055)")
        else:
            self.get_logger().warn("하드웨어 없음 — 시뮬레이션 모드로 실행")

        # ── yaw 오프셋 (reset 시 현재 yaw를 0 기준으로 보정) ────
        self._yaw_offset = 0.0

        # ── AGV 상태 ───────────────────────────────────────────────
        self._state = {
            "x": 0.0, "y": 0.0, "yaw": 0.0, "distance": 0.0,
            "speed_l": 0.0, "speed_r": 0.0,
            "phase": "tracking",
            "dest_label": "", "dest_x": 0.0, "dest_y": 0.0,
            "cycle": 0,
            "grabbed_labels": [],
            "_last_grabbed_label": "",
        }
        self._state_lock = threading.Lock()

        # ── 트래킹 상태 ────────────────────────────────────────────
        self._tracking = {
            "detected": False,
            "cx": 0, "cy": 0,
            "confidence": 0.0, "label": "",
            "bbox": None, "steer": 0.0,
            "last_seen": 0.0,
            "all_boxes": [],
            "locked": False,
            "lock_label": "", "lock_cx": 0, "lock_cy": 0,
        }
        self._tracking_lock = threading.Lock()

        self._resume_event = threading.Event()

        # ── ROS 인터페이스 ─────────────────────────────────────────
        self.sub_det  = self.create_subscription(
            String, "/agv/detection", self._detection_cb, 10)
        self.pub_state = self.create_publisher(String, "/agv/state", 10)

        # ── /agv/cmd 구독 (visualizer START/RESET 명령) ──────────
        self._start_event = threading.Event()
        self._reset_event = threading.Event()
        self.create_subscription(String, "/agv/cmd", self._cmd_cb, 10)

        # ── 상태 publish 타이머 (20 Hz) ───────────────────────────
        self.create_timer(0.05, self._publish_state)

        # ── 미션 스레드 시작 ───────────────────────────────────────
        t = threading.Thread(target=self._mission_thread, daemon=True)
        t.start()
        self.get_logger().info("mission_node 준비 완료 — START 버튼을 누르세요")

    # ==============================================================
    # 하드웨어 헬퍼
    # ==============================================================

    def _drive(self, l_speed: float, r_speed: float):
        if self._hw:
            self._lm.start(int(-l_speed))
            self._rm.start(int(r_speed))
        with self._state_lock:
            self._state["speed_l"] = l_speed
            self._state["speed_r"] = r_speed

    def _stop(self):
        if self._hw:
            self._lm.stop()
            self._rm.stop()
        with self._state_lock:
            self._state["speed_l"] = 0.0
            self._state["speed_r"] = 0.0

    def _get_yaw(self) -> float:
        if not self._hw:
            with self._state_lock:
                return self._state["yaw"]
        yaw, _, _ = self._imu.euler
        if yaw > 180:
            yaw -= 360
        # yaw 오프셋 적용 (-180 ~ 180 범위 유지)
        yaw -= self._yaw_offset
        if yaw > 180:  yaw -= 360
        if yaw < -180: yaw += 360
        return yaw

    def _get_encoder(self):
        """(left_deg, right_deg) 누적 각도 반환."""
        if not self._hw:
            return 0.0, 0.0
        return self._lm.get_degrees_counted(), self._rm.get_degrees_counted()

    # ==============================================================
    # Detection 콜백 — yolo_node로부터 수신
    # ==============================================================

    def _detection_cb(self, msg: String):
        """
        /agv/detection 수신 → lock 로직 적용 → _tracking 갱신.
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        all_boxes  = data.get("all_boxes", [])   # [{"conf":, "xyxy":, "label":}, ...]
        last_seen  = data.get("stamp", time.time())
        cx_center  = CAM_W // 2

        with self._tracking_lock:
            locked     = self._tracking["locked"]
            lock_label = self._tracking["lock_label"]
            lock_cx    = self._tracking["lock_cx"]
            lock_cy    = self._tracking["lock_cy"]

        # ── best 박스 선택 ─────────────────────────────────────────
        if locked:
            best = self._find_locked_box(all_boxes, lock_label, lock_cx, lock_cy)
        else:
            best = self._find_closest_box(all_boxes)

        # ── tracking 딕셔너리 갱신 ─────────────────────────────────
        with self._tracking_lock:
            self._tracking["all_boxes"] = all_boxes
            if best is not None:
                x1, y1, x2, y2 = best["xyxy"]
                bcx = (x1 + x2) // 2
                bcy = (y1 + y2) // 2
                error = bcx - cx_center
                steer = 0.0
                if abs(error) >= DEAD_ZONE:
                    steer = max(-MAX_STEER, min(MAX_STEER, STEER_GAIN * error))

                self._tracking.update({
                    "detected": True,
                    "cx": bcx, "cy": bcy,
                    "confidence": best["conf"],
                    "label": best["label"],
                    "bbox": tuple(best["xyxy"]),
                    "steer": steer,
                    "last_seen": last_seen,
                })
                if self._tracking["locked"]:
                    self._tracking["lock_cx"] = bcx
                    self._tracking["lock_cy"] = bcy
            else:
                self._tracking.update({
                    "detected": False,
                    "cx": 0, "cy": 0,
                    "confidence": 0.0, "label": "",
                    "bbox": None, "steer": 0.0,
                })

    # ==============================================================
    # Lock 헬퍼
    # ==============================================================

    @staticmethod
    def _find_closest_box(all_boxes: list):
        """y2 최대 박스 반환 (가장 가까운 물체)."""
        if not all_boxes:
            return None
        return max(all_boxes, key=lambda b: b["xyxy"][3])

    @staticmethod
    def _find_locked_box(all_boxes: list, lock_label: str,
                         lock_cx: int, lock_cy: int):
        """lock_label 일치 + 위치 근접 박스 반환."""
        candidates = [
            b for b in all_boxes
            if b["label"].strip().lower() == lock_label.strip().lower()
        ]
        if not candidates:
            return None

        def dist(b):
            x1, y1, x2, y2 = b["xyxy"]
            return math.sqrt(((x1+x2)//2 - lock_cx)**2 +
                             ((y1+y2)//2 - lock_cy)**2)

        best = min(candidates, key=dist)
        return best if dist(best) <= LOCK_POS_TOLERANCE else None

    def _release_lock(self):
        with self._tracking_lock:
            self._tracking["locked"]     = False
            self._tracking["lock_label"] = ""
            self._tracking["lock_cx"]    = 0
            self._tracking["lock_cy"]    = 0
        self.get_logger().info("[LOCK] lock 해제")

    # ==============================================================
    # State publish
    # ==============================================================

    def _publish_state(self):
        with self._state_lock:
            snap = dict(self._state)
            snap["grabbed_labels"] = list(self._state["grabbed_labels"]) \
                if isinstance(self._state["grabbed_labels"], set) else self._state["grabbed_labels"]

        with self._tracking_lock:
            tsnap = {k: v for k, v in self._tracking.items()
                     if k not in ("bbox", "all_boxes")}
            tsnap["all_boxes"] = self._tracking["all_boxes"]
            tsnap["bbox"] = list(self._tracking["bbox"]) \
                if self._tracking["bbox"] else None

        snap["detection"] = tsnap
        # _last_grabbed_label은 내부 전용 필드 — 외부 노출 제외
        snap.pop("_last_grabbed_label", None)

        msg = String()
        msg.data = json.dumps(snap)
        self.pub_state.publish(msg)

    # ==============================================================
    # CMD 콜백 (visualizer → mission)
    # ==============================================================

    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == "start":
            self.get_logger().info("[CMD] START 수신")
            self._start_event.set()
        elif cmd == "reset":
            self.get_logger().info("[CMD] RESET 수신")
            self._reset_event.set()

    def _wait_for_start(self):
        """START 신호가 올 때까지 블로킹. RESET이 오면 False 반환."""
        self.get_logger().info("[MISSION] START 대기 중...")
        with self._state_lock:
            self._state["phase"] = "waiting"
        self._start_event.clear()
        self._reset_event.clear()
        while True:
            if self._reset_event.is_set():
                return False
            if self._start_event.is_set():
                self._start_event.clear()
                return True
            time.sleep(0.05)

    def _reset_state(self):
        """소프트웨어 상태 완전 초기화. 모터 즉시 정지."""
        self._stop()
        # 현재 IMU yaw를 오프셋으로 저장 → 이후 _get_yaw()가 0 반환
        if self._hw:
            raw_yaw, _, _ = self._imu.euler
            if raw_yaw > 180: raw_yaw -= 360
            self._yaw_offset = raw_yaw
        else:
            self._yaw_offset = 0.0
        self.get_logger().info(f"[RESET] yaw 오프셋 설정: {self._yaw_offset:.1f}°")
        with self._state_lock:
            self._state.update({
                "x": 0.0, "y": 0.0, "yaw": 0.0, "distance": 0.0,
                "speed_l": 0.0, "speed_r": 0.0,
                "phase": "waiting",
                "dest_label": "", "dest_x": 0.0, "dest_y": 0.0,
                "cycle": 0,
                "grabbed_labels": [],
                "_last_grabbed_label": "",
            })
        with self._tracking_lock:
            self._tracking.update({
                "detected": False,
                "cx": 0, "cy": 0,
                "confidence": 0.0, "label": "",
                "bbox": None, "steer": 0.0,
                "last_seen": 0.0,
                "all_boxes": [],
                "locked": False,
                "lock_label": "", "lock_cx": 0, "lock_cy": 0,
            })
        self._resume_event.clear()
        self._start_event.clear()
        self._reset_event.clear()
        self.get_logger().info("[RESET] 상태 초기화 완료")

    # ==============================================================
    # 이동 유틸리티
    # ==============================================================

    @staticmethod
    def _degree_to_mm(deg: float) -> float:
        return (deg / 360.0) * math.pi * WHEEL_DIAMETER_MM

    def _turn_to(self, target_deg: float, speed: float = 55):
        """목표 yaw까지 제자리 회전."""
        while True:
            yaw   = self._get_yaw()
            error = target_deg - yaw
            if error > 180:
                error -= 360
            elif error < -180:
                error += 360
            with self._state_lock:
                self._state["yaw"] = yaw
            if abs(error) < 12:
                break
            if error > 0:
                self._drive(speed, -speed)
            else:
                self._drive(-speed, speed)
            time.sleep(0.01)
        self._stop()
        self.get_logger().info(f"[TURN] 완료 — yaw={self._get_yaw():.1f}°")

    def _straight_by_encoder(self, dist_mm: float, heading_deg: float,
                              speed: float, yaw_kp: float = NAV_YAW_KP,
                              max_steer: float = 30.0):
        """엔코더 기반 직진. heading_deg 방향을 자이로로 유지."""
        target_enc = (dist_mm / (math.pi * WHEEL_DIAMETER_MM)) * 360.0
        start_l, start_r = self._get_encoder()
        prev_deg = 0.0

        while True:
            cl, cr   = self._get_encoder()
            avg_deg  = (abs(cl - start_l) + abs(cr - start_r)) / 2
            if avg_deg >= target_enc:
                break

            yaw     = self._get_yaw()
            yaw_err = heading_deg - yaw
            if yaw_err > 180:
                yaw_err -= 360
            elif yaw_err < -180:
                yaw_err += 360

            steer   = max(-max_steer, min(max_steer, yaw_kp * yaw_err))
            self._drive(speed + steer, speed - steer)

            delta    = self._degree_to_mm(avg_deg - prev_deg)
            prev_deg = avg_deg
            yaw_rad  = math.radians(yaw)
            with self._state_lock:
                self._state["x"] = max(0, min(FIELD_MM,
                    self._state["x"] + delta * math.sin(yaw_rad)))
                self._state["y"] = max(0, min(FIELD_MM,
                    self._state["y"] + delta * math.cos(yaw_rad)))
                self._state["yaw"]      = yaw
                self._state["distance"] += delta
            time.sleep(0.02)

        self._stop()

    # ==============================================================
    # 미션 시퀀스
    # ==============================================================

    def _mission_thread(self):
        """최상위 미션 루프. START 대기 → 미션 실행. RESET 시 재대기."""
        while True:
            # ── START 대기 ────────────────────────────────────────
            if not self._wait_for_start():
                self._reset_state()
                continue

            self.get_logger().info("[MISSION] 미션 시작!")
            with self._state_lock:
                self._state["phase"] = "tracking"
                self._state["yaw"]   = self._get_yaw()
            self._turn_to(45, speed=40)
            time.sleep(0.5)

            # ── 미션 루프 ─────────────────────────────────────────
            aborted = False
            while True:
                if self._reset_event.is_set():
                    self.get_logger().info("[MISSION] RESET — 미션 중단")
                    self._reset_state()
                    aborted = True
                    break

                with self._state_lock:
                    phase = self._state["phase"]

                if phase == "done":
                    self._stop()
                    self.get_logger().info("[MISSION] 모든 임무 완료.")
                    while not self._reset_event.is_set():
                        time.sleep(0.1)
                    self._reset_state()
                    aborted = True
                    break

                if phase in ("approach", "grab", "navigate", "release",
                             "spin180", "returning"):
                    time.sleep(0.1)
                    continue

                if self._resume_event.is_set():
                    self._resume_event.clear()

                self._run_tracking_loop()
                time.sleep(0.05)

            if aborted:
                continue

        self._stop()

    # ──────────────────────────────────────────────────────────────
    def _run_tracking_loop(self):
        """tracking phase 루프. grab 트리거 시 반환."""
        start_l, start_r   = self._get_encoder()
        prev_deg            = 0.0
        track_start_time    = None
        was_tracking_stable = False
        close_triggered     = False
        last_valid_label    = ""
        lock_set            = False
        prev_error          = 0.0
        prev_time           = time.time()
        lost_since          = None          # 미감지 시작 시각

        while True:
            if self._reset_event.is_set():
                self._stop()
                return

            with self._state_lock:
                phase = self._state["phase"]
            if phase != "tracking":
                return

            now = time.time()

            with self._tracking_lock:
                detected  = self._tracking["detected"]
                steer     = self._tracking["steer"]
                last_seen = self._tracking["last_seen"]
                cy        = self._tracking["cy"]
                label     = self._tracking["label"]
                cx        = self._tracking["cx"]

            # 이미 grab한 클래스는 무시
            with self._state_lock:
                grabbed = set(self._state["grabbed_labels"])
            if detected and label.strip().lower() in grabbed:
                detected = False
                steer    = 0.0

            lost = (now - last_seen) > LOST_TIMEOUT

            # ── 연속 추적 + lock 설정 ────────────────────────────
            if detected and not lost:
                lost_since       = None          # 물체 보이면 타이머 리셋
                if label:
                    last_valid_label = label
                if track_start_time is None:
                    track_start_time = now
                continuous_sec = now - track_start_time

                if not lock_set and continuous_sec >= LOCK_STABLE_SEC:
                    with self._tracking_lock:
                        self._tracking["locked"]     = True
                        self._tracking["lock_label"] = label
                        self._tracking["lock_cx"]    = cx
                        self._tracking["lock_cy"]    = cy
                    lock_set = True
                    self.get_logger().info(
                        f"[LOCK] '{label}' lock 설정 — cx={cx}, cy={cy}")

                if continuous_sec >= STABLE_TRACK_SEC:
                    was_tracking_stable = True
            else:
                # 근접 판정 B: 안정 추적 후 소실
                if not close_triggered and was_tracking_stable:
                    self.get_logger().info(
                        f"[MISSION] 근접 판정 B: 안정 추적 후 소실 → grab")
                    close_triggered     = True
                    was_tracking_stable = False
                    track_start_time    = None
                    lost_since          = None
                    self._stop()
                    self._trigger_grab(self._get_yaw(), last_valid_label)
                    return
                track_start_time    = None
                was_tracking_stable = False

                # ── 3초 미감지 → 재탐색 ──────────────────────────
                if not close_triggered:
                    if lost_since is None:
                        lost_since = now
                    elif now - lost_since >= 3.0:
                        self.get_logger().info(
                            "[TRACKING] 3초 미감지 → 재탐색 (center scan)")
                        lost_since = None
                        self._stop()
                        self._release_lock()
                        t = threading.Thread(
                            target=self._move_to_center_and_scan, daemon=True)
                        t.start()
                        return

            # 근접 판정 A: cy > CLOSE_Y_THRESH
            if detected and not lost and not close_triggered:
                close_y = CLOSE_Y_THRESH_MAP.get(label.strip().lower(), CLOSE_Y_THRESH_DEFAULT)
                if cy > close_y:
                    self.get_logger().info(
                        f"[MISSION] 근접 판정 A: cy={cy} > {close_y} → grab")
                    close_triggered = True
                    self._stop()
                    self._trigger_grab(self._get_yaw(), label)
                    return

            # 일반 주행 — PD 제어
            if detected and not lost and not close_triggered:
                now_t = time.time()
                dt    = max(now_t - prev_time, 0.001)

                with self._tracking_lock:
                    error = self._tracking["cx"] - (CAM_W // 2)

                p_term   = STEER_GAIN * error if abs(error) >= DEAD_ZONE else 0.0
                d_term   = STEER_D_GAIN * (error - prev_error) / dt
                pd_steer = max(-MAX_STEER, min(MAX_STEER, p_term + d_term))

                prev_error = error
                prev_time  = now_t

                self._drive(BASE_SPEED + pd_steer, BASE_SPEED - pd_steer)
            else:
                self._drive(0.0, 0.0)

            # 데드레코닝
            yaw = self._get_yaw()
            cl, cr   = self._get_encoder()
            avg_deg  = (abs(cl - start_l) + abs(cr - start_r)) / 2
            delta    = self._degree_to_mm(avg_deg - prev_deg)
            prev_deg = avg_deg
            yaw_rad  = math.radians(yaw)

            with self._state_lock:
                self._state["x"] = max(0, min(FIELD_MM,
                    self._state["x"] + delta * math.sin(yaw_rad)))
                self._state["y"] = max(0, min(FIELD_MM,
                    self._state["y"] + delta * math.cos(yaw_rad)))
                self._state["yaw"]      = yaw
                self._state["distance"] += delta

            time.sleep(0.02)

    # ──────────────────────────────────────────────────────────────
    def _trigger_grab(self, target_yaw: float, grabbed_label: str):
        """grab_sequence를 별도 스레드로 실행."""
        with self._state_lock:
            self._state["phase"]               = "approach"
            self._state["_last_grabbed_label"] = grabbed_label
        t = threading.Thread(
            target=self._grab_sequence,
            args=(target_yaw, grabbed_label),
            daemon=True)
        t.start()

    def _grab_sequence(self, target_yaw: float, grabbed_label: str):
        """10cm 전진 → Motor A grab → navigate → release."""
        self.get_logger().info(f"[GRAB] 10cm 접근 시작 (label={grabbed_label})")

        # 10cm 직진 (자이로 보정)
        start_l, start_r = self._get_encoder()
        prev_deg = 0.0
        while True:
            cl, cr   = self._get_encoder()
            avg_deg  = (abs(cl - start_l) + abs(cr - start_r)) / 2
            traveled = self._degree_to_mm(avg_deg)
            if traveled >= APPROACH_DIST_MM:
                break
            yaw     = self._get_yaw()
            yaw_err = target_yaw - yaw
            if yaw_err > 180:
                yaw_err -= 360
            elif yaw_err < -180:
                yaw_err += 360
            corr = max(-(APPROACH_SPEED - 1),
                       min(APPROACH_SPEED - 1, YAW_KP * yaw_err))
            self._drive(APPROACH_SPEED + corr, APPROACH_SPEED - corr)

            delta    = self._degree_to_mm(avg_deg - prev_deg)
            prev_deg = avg_deg
            yaw_rad  = math.radians(yaw)
            with self._state_lock:
                self._state["x"] = max(0, min(FIELD_MM,
                    self._state["x"] + delta * math.sin(yaw_rad)))
                self._state["y"] = max(0, min(FIELD_MM,
                    self._state["y"] + delta * math.cos(yaw_rad)))
                self._state["yaw"]      = yaw
                self._state["distance"] += delta
            time.sleep(0.02)

        self._stop()
        self.get_logger().info("[GRAB] 엔코더 전진 완료 → 0.5초 추가 전진")

        # 0.5초 추가 전진 (자이로 보정 유지)
        extra_start = time.time()
        while time.time() - extra_start < 0.5:
            yaw     = self._get_yaw()
            yaw_err = target_yaw - yaw
            if yaw_err > 180:  yaw_err -= 360
            elif yaw_err < -180: yaw_err += 360
            corr = max(-(APPROACH_SPEED - 1),
                       min(APPROACH_SPEED - 1, YAW_KP * yaw_err))
            self._drive(APPROACH_SPEED + corr, APPROACH_SPEED - corr)
            time.sleep(0.02)
        self._stop()
        self.get_logger().info("[GRAB] 추가 전진 완료 → Motor A 작동")

        # Motor A grab
        with self._state_lock:
            self._state["phase"] = "grab"
        if self._hw:
            from HandsON_BuildHat_API import Motor as SPMotor
            ma = SPMotor("C")
            ma.start(50)
            time.sleep(5)
            ma.stop()
        else:
            time.sleep(5)   # 시뮬레이션에서는 대기만

        self.get_logger().info(f"[GRAB] 완료 → navigate ({grabbed_label})")
        self._navigate_to_destination(grabbed_label)

    # ──────────────────────────────────────────────────────────────
    def _navigate_to_destination(self, label: str):
        """목적지로 이동 후 release."""
        key = label.strip().lower()
        if key not in DESTINATION_MAP:
            self.get_logger().warn(f"[NAV] 알 수 없는 label '{label}' → skip")
            self._post_release()
            return

        dest_x, dest_y = DESTINATION_MAP[key]
        self.get_logger().info(f"[NAV] {label.upper()} → ({dest_x}, {dest_y}) mm")

        with self._state_lock:
            self._state.update({
                "phase":      "navigate",
                "dest_label": label,
                "dest_x":     float(dest_x),
                "dest_y":     float(dest_y),
            })
            cur_x, cur_y = self._state["x"], self._state["y"]

        dx, dy   = dest_x - cur_x, dest_y - cur_y
        dist_mm  = math.sqrt(dx**2 + dy**2)

        if dist_mm > NAV_ARRIVE_THRESH:
            heading = math.degrees(math.atan2(dx, dy))
            if heading > 180:  heading -= 360
            if heading < -180: heading += 360
            self._turn_to(heading, speed=NAV_TURN_SPEED)
            time.sleep(0.3)
            self._straight_by_encoder(dist_mm, heading, NAV_SPEED)

        self._stop()
        time.sleep(0.3)

        # Release
        self.get_logger().info(f"[RELEASE] Motor A 역방향 {RELEASE_DURATION}s")
        with self._state_lock:
            self._state["phase"] = "release"
        if self._hw:
            from HandsON_BuildHat_API import Motor as SPMotor
            ma = SPMotor("C")
            ma.start(int(RELEASE_SPEED))
            time.sleep(RELEASE_DURATION)
            ma.stop()
        else:
            time.sleep(RELEASE_DURATION)

        self.get_logger().info("[RELEASE] 완료")
        self._post_release()

    # ──────────────────────────────────────────────────────────────
    def _post_release(self):
        """release 후 cycle 증가 → 중앙 스캔 or 원점 복귀."""
        self._release_lock()

        with self._state_lock:
            self._state["cycle"] += 1
            cycle = self._state["cycle"]
            self._state["dest_label"] = ""
            last_label = self._state.get("_last_grabbed_label", "")
            if last_label:
                if isinstance(self._state["grabbed_labels"], list):
                    if last_label.strip().lower() not in self._state["grabbed_labels"]:
                        self._state["grabbed_labels"].append(last_label.strip().lower())
                else:
                    self._state["grabbed_labels"].add(last_label.strip().lower())

        self.get_logger().info(f"[CYCLE] {cycle}/{TOTAL_CYCLES} 완료")

        if cycle < TOTAL_CYCLES:
            t = threading.Thread(
                target=self._move_to_center_and_scan, daemon=True)
            t.start()
        else:
            self.get_logger().info("[CYCLE] 모든 grab 완료 → 원점 복귀")
            self._return_to_origin()

    # ──────────────────────────────────────────────────────────────
    def _move_to_center_and_scan(self):
        """release 후 중앙 방향 선제 탐색 → 중앙 이동 → 45° 스캔."""
        CENTER_X, CENTER_Y = 450.0, 450.0

        with self._state_lock:
            self._state["phase"] = "spin180"
            cur_x, cur_y = self._state["x"], self._state["y"]

        # STEP 1: 중앙 방향 회전 후 1.5초 감지 시도
        dx, dy = CENTER_X - cur_x, CENTER_Y - cur_y
        heading = math.degrees(math.atan2(dx, dy))
        if heading > 180:  heading -= 360
        if heading < -180: heading += 360
        self._turn_to(heading, speed=SPIN_180_SPEED)
        time.sleep(0.5)

        grabbed = set(self._state["grabbed_labels"]) \
            if isinstance(self._state["grabbed_labels"], set) \
            else set(self._state["grabbed_labels"])

        scan_start = time.time()
        while time.time() - scan_start < 1.5:
            with self._tracking_lock:
                detected = self._tracking["detected"]
                label    = self._tracking["label"]
            if detected and label.strip().lower() not in grabbed:
                self.get_logger().info(f"[CENTER] 선제 감지: {label} → tracking 복귀")
                with self._state_lock:
                    self._state["phase"] = "tracking"
                self._resume_event.set()
                return
            time.sleep(0.05)

        # STEP 2: 중앙으로 이동
        with self._state_lock:
            self._state["phase"] = "navigate"
            cur_x, cur_y = self._state["x"], self._state["y"]

        dx, dy  = CENTER_X - cur_x, CENTER_Y - cur_y
        dist_mm = math.sqrt(dx**2 + dy**2)

        if dist_mm > NAV_ARRIVE_THRESH:
            self._straight_by_encoder(dist_mm, heading, NAV_SPEED)
        self._stop()

        # STEP 3: 45° 씩 8방향 스캔
        with self._state_lock:
            self._state["phase"] = "spin180"
        base_yaw = self._get_yaw()

        for i in range(8):
            scan_yaw = base_yaw + i * 45
            if scan_yaw > 180:  scan_yaw -= 360
            self._turn_to(scan_yaw, speed=35)

            t0 = time.time()
            while time.time() - t0 < 1.0:
                with self._tracking_lock:
                    detected = self._tracking["detected"]
                    label    = self._tracking["label"]
                if detected and label.strip().lower() not in grabbed:
                    self.get_logger().info(f"[SCAN] 발견: {label} → tracking 복귀")
                    with self._state_lock:
                        self._state["phase"] = "tracking"
                    self._resume_event.set()
                    return
                time.sleep(0.05)

        self.get_logger().info("[SCAN] 물체 없음 → tracking 복귀")
        with self._state_lock:
            self._state["phase"] = "tracking"
        self._resume_event.set()

    # ──────────────────────────────────────────────────────────────
    def _return_to_origin(self):
        """현재 위치에서 (0,0) 복귀."""
        self.get_logger().info("[RETURN] 원점 복귀 시작")
        with self._state_lock:
            self._state["phase"] = "returning"
            cur_x, cur_y = self._state["x"], self._state["y"]

        dest_x, dest_y = RETURN_ORIGIN
        dx, dy  = dest_x - cur_x, dest_y - cur_y
        dist_mm = math.sqrt(dx**2 + dy**2)

        if dist_mm <= RETURN_ARRIVE_THRESH:
            self.get_logger().info("[RETURN] 이미 원점 근처 → 생략")
        else:
            heading = math.degrees(math.atan2(dx, dy))
            if heading > 180:  heading -= 360
            if heading < -180: heading += 360
            self._turn_to(heading, speed=RETURN_TURN_SPEED)
            time.sleep(0.3)
            self._straight_by_encoder(dist_mm, heading, RETURN_SPEED)

        self._stop()
        self.get_logger().info("[RETURN] 완료. 모든 임무 종료.")
        with self._state_lock:
            self._state["phase"] = "done"

    # ==============================================================
    # Node 종료
    # ==============================================================

    def destroy_node(self):
        self._stop()
        super().destroy_node()


# ──────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()