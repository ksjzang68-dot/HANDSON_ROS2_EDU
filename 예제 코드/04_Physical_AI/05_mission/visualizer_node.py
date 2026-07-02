#!/usr/bin/env python3
"""
visualizer_node.py
==================
역할:
  - /agv/state  구독 → 궤적맵 + 패널 렌더링
  - /agv/frame  구독 → 카메라 프리뷰 렌더링
  - 키보드 입력 처리 (C: 궤적 초기화, F: 전체화면 토글, ESC: 종료)

Subscribe :
  /agv/state  (std_msgs/String)            ← mission_node
  /agv/frame  (sensor_msgs/CompressedImage) ← yolo_node

Publish   :
  /agv/cmd    (std_msgs/String)            → mission_node (start/reset)

Note: pygame 이벤트 루프는 메인 스레드에서만 동작해야 하므로
      ROS spin은 별도 스레드에서 실행.
"""

import json
import math
import threading

import cv2
import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import os
from agv_tracker.config import (
    FIELD_MM, FIELD_PX, PANEL_W, PREVIEW_W, PREVIEW_H,
    WIN_W, WIN_H, FPS, MAX_TRAIL,
    CLOSE_Y_THRESH_MAP,
    CLOSE_Y_THRESH_DEFAULT, DEAD_ZONE, BASE_SPEED,
    DESTINATION_MAP, TOTAL_CYCLES,
    C_BG, C_FIELD, C_GRID, C_BORDER, C_TRAIL,
    C_ROBOT, C_START, C_PANEL, C_DIM, C_ACCENT, C_ORANGE,
    C_RED, C_GREEN, C_WHITE, C_PREVIEW, C_LOCK,
    DEST_COLORS,
)

# config 기준 비율 (레이아웃 계산용)
_BASE_W   = WIN_W          # 880
_BASE_H   = WIN_H          # 870
_FIELD_R  = FIELD_PX / _BASE_W          # 필드 너비 비율
_PANEL_R  = PANEL_W  / _BASE_W          # 패널 너비 비율
_PREV_H_R = (PREVIEW_H + 10) / _BASE_H  # 하단 프리뷰 높이 비율


# ── 동적 레이아웃 계산 ─────────────────────────────────────────────
def _layout(sw, sh):
    """현재 화면 크기(sw, sh)로부터 레이아웃 치수를 반환."""
    field_px  = int(sw * _FIELD_R)
    panel_w   = sw - field_px
    prev_h    = int(sh * _PREV_H_R)
    field_h   = sh - prev_h          # 필드+패널 높이
    # 필드는 정사각형 유지
    field_sz  = min(field_px, field_h)
    return dict(
        sw=sw, sh=sh,
        field_sz=field_sz,
        field_px=field_px,
        panel_w=panel_w,
        prev_h=prev_h,
        field_h=field_h,
        scale=field_sz / FIELD_PX,   # 기준 대비 확대비
    )


# ── mm → 화면 px 변환 (동적 스케일) ───────────────────────────────
def mm_to_px(x_mm, y_mm, scale=1.0):
    IMG_TOTAL  = 1200.0
    IMG_OFFSET = 125.0
    IMG_FIELD  = 950.0
    base_scale       = FIELD_PX / IMG_TOTAL
    field_scale      = IMG_FIELD / FIELD_MM
    offset_px        = IMG_OFFSET * base_scale
    sx = offset_px + x_mm * field_scale * base_scale
    sy = offset_px + (FIELD_MM - y_mm) * field_scale * base_scale
    return int(sx * scale), int(sy * scale)


# ══════════════════════════════════════════════════════════════════
class VisualizerNode(Node):
    """pygame 기반 시각화 노드 (동적 스케일링)."""

    def __init__(self):
        super().__init__("visualizer_node")

        self._state_json: str | None = None
        self._frame_bgr:  np.ndarray | None = None
        self._data_lock = threading.Lock()

        # ── ROS 구독 / 발행 ───────────────────────────────────────
        self.create_subscription(String,          "/agv/state", self._state_cb, 10)
        self.create_subscription(CompressedImage, "/agv/frame", self._frame_cb, 10)
        self._pub_cmd = self.create_publisher(String, "/agv/cmd", 10)

        # ── 미션 UI 상태 ──────────────────────────────────────────
        self._mission_started = False

        # ── pygame 초기화 ─────────────────────────────────────────
        pygame.init()
        self._fullscreen = False
        self._screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("AGV Tracker — ROS2 Visualizer")
        self._clock = pygame.time.Clock()

        # ── 버튼 rect (클릭 판정용, 매 프레임 갱신) ──────────────
        self._btn_start_rect = None
        self._btn_reset_rect = None

        # ── 맵 원본 이미지 로드 (스케일은 매 프레임) ─────────────
        self._map_img_raw = self._load_map_image()

        # ── 궤적 (기준 좌표 mm 저장) ─────────────────────────────
        self._trail_mm: list[tuple[float, float]] = []

        self.get_logger().info("visualizer_node 준비 완료")

    # ==============================================================
    # 구독 콜백
    # ==============================================================

    def _state_cb(self, msg: String):
        with self._data_lock:
            self._state_json = msg.data

    def _frame_cb(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._data_lock:
                self._frame_bgr = frame

    # ==============================================================
    # 맵 이미지 로드
    # ==============================================================

    def _load_map_image(self):
        map_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "map.png")
        if os.path.exists(map_path):
            try:
                img = pygame.image.load(map_path).convert()
                self.get_logger().info(f"맵 이미지 로드 성공: {map_path}")
                return img
            except Exception as e:
                self.get_logger().warn(f"맵 이미지 로드 실패: {e}")
        return None

    # ==============================================================
    # 전체화면 토글
    # ==============================================================

    def _toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            self._screen = pygame.display.set_mode(
                (info.current_w, info.current_h), pygame.FULLSCREEN)
        else:
            self._screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)

    def _send_cmd(self, cmd: str):
        msg      = String()
        msg.data = cmd
        self._pub_cmd.publish(msg)
        self.get_logger().info(f"[CMD] {cmd}")

    # ==============================================================
    # 메인 루프
    # ==============================================================

    def run(self):
        running = True
        while running:
            sw, sh = self._screen.get_size()
            L = _layout(sw, sh)

            # ── 이벤트 처리 ───────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif event.key == pygame.K_c:
                        self._trail_mm.clear()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_button_click(event.pos)

            # ── 데이터 스냅샷 ─────────────────────────────────────
            with self._data_lock:
                state_json = self._state_json
                frame_bgr  = self._frame_bgr

            snap  = {}
            tsnap = {}
            if state_json:
                try:
                    data  = json.loads(state_json)
                    tsnap = data.pop("detection", {})
                    snap  = data
                except Exception:
                    pass

            # ── 궤적 갱신 (mm 단위로 저장) ────────────────────────
            x_mm = snap.get("x", 0.0)
            y_mm = snap.get("y", 0.0)
            if not self._trail_mm or self._trail_mm[-1] != (x_mm, y_mm):
                self._trail_mm.append((x_mm, y_mm))
            if len(self._trail_mm) > MAX_TRAIL:
                self._trail_mm.pop(0)

            # ── 렌더링 ────────────────────────────────────────────
            self._screen.fill(C_BG)
            self._render(self._screen, L, snap, tsnap, frame_bgr)

            pygame.display.flip()
            self._clock.tick(FPS)

        pygame.quit()
        rclpy.shutdown()

    # ==============================================================
    # 통합 렌더러
    # ==============================================================

    def _render(self, surf, L, snap, tsnap, frame_bgr):
        sc  = L["scale"]
        fsz = L["field_sz"]
        fpx = L["field_px"]
        pw  = L["panel_w"]
        sh  = L["sh"]
        sw  = L["sw"]
        prev_h = L["prev_h"]
        field_h = L["field_h"]

        # ── 필드 배경 ──────────────────────────────────────────────
        self._draw_field(surf, fsz, sc)

        # ── 궤적 ──────────────────────────────────────────────────
        trail_px = [mm_to_px(x, y, sc) for x, y in self._trail_mm]
        self._draw_trail(surf, trail_px)

        # ── 목적지/복귀 선 ─────────────────────────────────────────
        rx, ry = mm_to_px(snap.get("x", 0), snap.get("y", 0), sc)
        self._draw_dest_line(surf, snap, rx, ry, sc)
        self._draw_return_line(surf, snap, rx, ry, sc)

        # ── 시작점 마커 ───────────────────────────────────────────
        if trail_px:
            pygame.draw.circle(surf, C_START, trail_px[0], max(4, int(6*sc)))
            pygame.draw.circle(surf, C_WHITE, trail_px[0], max(4, int(6*sc)), 1)

        # ── 로봇 ──────────────────────────────────────────────────
        self._draw_robot(surf, rx, ry, snap.get("yaw", 0.0), sc)

        # ── 패널 ──────────────────────────────────────────────────
        self._draw_panel(surf, snap, tsnap, self._clock.get_fps(),
                         fpx, pw, field_h, sc)

        # ── START / RESET 버튼 ────────────────────────────────────
        self._draw_mission_buttons(surf, fpx, pw, field_h, sc)

        # ── 카메라 프리뷰 ──────────────────────────────────────────
        self._draw_camera_preview(surf, frame_bgr, tsnap,
                                  sw, sh, prev_h, sc)

        # ── 상태바 ─────────────────────────────────────────────────
        self._draw_statusbar(surf, sw, sh, sc)

    # ==============================================================
    # 필드 배경
    # ==============================================================

    def _draw_field(self, surf, fsz, sc):
        if self._map_img_raw:
            scaled = pygame.transform.scale(self._map_img_raw, (fsz, fsz))
            surf.blit(scaled, (0, 0))
        else:
            pygame.draw.rect(surf, C_FIELD, (0, 0, fsz, fsz))
            step = fsz // 10
            for i in range(1, 10):
                p = i * step
                pygame.draw.line(surf, C_GRID, (p, 0), (p, fsz))
                pygame.draw.line(surf, C_GRID, (0, p), (fsz, p))

        # 격자 반투명 오버레이
        grid = pygame.Surface((fsz, fsz), pygame.SRCALPHA)
        step = fsz // 10
        for i in range(1, 10):
            p = i * step
            pygame.draw.line(grid, (255,255,255,30), (p,0), (p,fsz))
            pygame.draw.line(grid, (255,255,255,30), (0,p), (fsz,p))
        pygame.draw.rect(grid, (0,180,220,160), (0,0,fsz,fsz), 2)
        surf.blit(grid, (0, 0))

        # 목적지 마커
        fnt = pygame.font.SysFont("monospace", max(8, int(10*sc)))
        for key, (dx_mm, dy_mm) in DESTINATION_MAP.items():
            px, py = mm_to_px(dx_mm, dy_mm, sc)
            col    = DEST_COLORS.get(key, C_DIM)
            r      = max(4, int(7*sc))
            pygame.draw.circle(surf, col, (px, py), r, 2)
            t = fnt.render(key.upper(), True, col)
            surf.blit(t, (px + int(10*sc), py - int(6*sc)))

        # 원점 마커
        opx, opy = mm_to_px(0, 0, sc)
        r = max(5, int(8*sc))
        pygame.draw.circle(surf, C_WHITE, (opx, opy), r, 2)
        lw = int(10*sc)
        pygame.draw.line(surf, C_WHITE, (opx-lw, opy), (opx+lw, opy), 1)
        pygame.draw.line(surf, C_WHITE, (opx, opy-lw), (opx, opy+lw), 1)
        t = fnt.render("ORIGIN", True, C_WHITE)
        surf.blit(t, (opx + int(10*sc), opy - int(6*sc)))

    # ==============================================================
    # 그리기 헬퍼
    # ==============================================================

    @staticmethod
    def _draw_trail(surf, trail):
        if len(trail) < 2:
            return
        for i in range(1, len(trail)):
            t = i / len(trail)
            color = (0, int(100+120*t), int(140+115*t))
            w = 1 if t < 0.5 else 2
            pygame.draw.line(surf, color, trail[i-1], trail[i], w)

    @staticmethod
    def _draw_robot(surf, px, py, yaw_deg, sc=1.0):
        R  = max(8, int(14*sc))
        yr = math.radians(-yaw_deg)
        glow = pygame.Surface((R*4, R*4), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255,107,53,45), (R*2, R*2), R*2)
        surf.blit(glow, (px-R*2, py-R*2))
        pygame.draw.circle(surf, C_ROBOT, (px, py), R, 2)
        tip = (px+int(math.sin(yr)*(R-2)),  py-int(math.cos(yr)*(R-2)))
        bl  = (px+int(math.sin(yr+2.4)*(R*.6)), py-int(math.cos(yr+2.4)*(R*.6)))
        br  = (px+int(math.sin(yr-2.4)*(R*.6)), py-int(math.cos(yr-2.4)*(R*.6)))
        pygame.draw.polygon(surf, C_ROBOT, [tip, bl, br])

    @staticmethod
    def _draw_dest_line(surf, snap, px, py, sc=1.0):
        if not snap.get("dest_label"):
            return
        key = snap["dest_label"].strip().lower()
        if key not in DESTINATION_MAP:
            return
        dx_mm, dy_mm = DESTINATION_MAP[key]
        dpx, dpy = mm_to_px(dx_mm, dy_mm, sc)
        col = DEST_COLORS.get(key, C_DIM)
        r   = max(8, int(12*sc))
        pygame.draw.circle(surf, col,     (dpx, dpy), r)
        pygame.draw.circle(surf, C_WHITE, (dpx, dpy), r, 2)
        lw = int(18*sc)
        pygame.draw.line(surf, C_WHITE, (dpx-lw, dpy), (dpx+lw, dpy), 1)
        pygame.draw.line(surf, C_WHITE, (dpx, dpy-lw), (dpx, dpy+lw), 1)
        pygame.draw.line(surf, col, (px, py), (dpx, dpy), 1)

    @staticmethod
    def _draw_return_line(surf, snap, px, py, sc=1.0):
        if snap.get("phase") != "returning":
            return
        rpx, rpy = mm_to_px(0, 0, sc)
        pygame.draw.line(surf, C_ORANGE, (px, py), (rpx, rpy), 1)

    def _draw_compass(self, surf, cx, cy, yaw_deg, sc=1.0):
        R = max(30, int(48*sc))
        pygame.draw.circle(surf, C_GRID,   (cx, cy), R)
        pygame.draw.circle(surf, C_BORDER, (cx, cy), R, 1)
        for i in range(36):
            a  = math.radians(i*10)
            ln = max(5, int(9*sc)) if i%9==0 else max(2, int(4*sc))
            col = C_ACCENT if i%9==0 else C_DIM
            x1 = cx+int(math.sin(a)*R);     y1 = cy-int(math.cos(a)*R)
            x2 = cx+int(math.sin(a)*(R-ln)); y2 = cy-int(math.cos(a)*(R-ln))
            pygame.draw.line(surf, col, (x1,y1), (x2,y2), 1)
        fnt = pygame.font.SysFont("monospace", max(8, int(11*sc)))
        for lbl, deg in [("N",0),("E",90),("S",180),("W",270)]:
            ar = math.radians(deg)
            lx = cx+int(math.sin(ar)*(R-int(16*sc)))
            ly = cy-int(math.cos(ar)*(R-int(16*sc)))
            col = C_ORANGE if lbl=="N" else C_DIM
            t = fnt.render(lbl, True, col)
            surf.blit(t, t.get_rect(center=(lx, ly)))
        yr = math.radians(yaw_deg)
        pygame.draw.line(surf, C_ORANGE, (cx, cy),
                         (cx+int(math.sin(yr)*(R-int(8*sc))),
                          cy-int(math.cos(yr)*(R-int(8*sc)))), max(2,int(3*sc)))
        pygame.draw.line(surf, C_DIM, (cx, cy),
                         (cx-int(math.sin(yr)*(R-int(16*sc))),
                          cy+int(math.cos(yr)*(R-int(16*sc)))), 2)
        pygame.draw.circle(surf, C_WHITE, (cx, cy), max(2, int(4*sc)))

    def _draw_panel(self, surf, snap, tsnap, fps,
                    fpx, pw, field_h, sc):
        panel_rect = pygame.Rect(fpx, 0, pw, field_h)
        pygame.draw.rect(surf, C_PANEL, panel_rect)
        pygame.draw.line(surf, C_BORDER, (fpx, 0), (fpx, field_h), 1)

        # 스케일에 맞는 폰트
        f_lg = pygame.font.SysFont("monospace", max(14, int(26*sc)), bold=True)
        f_md = pygame.font.SysFont("monospace", max(9,  int(15*sc)), bold=True)
        f_sm = pygame.font.SysFont("monospace", max(7,  int(11*sc)))

        m  = max(8, int(14*sc))   # 좌측 여백
        ox = fpx + m
        oy = max(8, int(14*sc))

        def sep(y):
            s = pygame.Surface((pw - m, 1)); s.fill(C_BORDER)
            surf.blit(s, (ox, y))

        def lbl(text, y):
            surf.blit(f_sm.render(text, True, C_DIM), (ox, y))

        def val(text, y, color=C_ACCENT, big=False):
            t = (f_lg if big else f_md).render(text, True, color)
            surf.blit(t, (ox, y))

        u = int(sc * 1.0)   # 줄 간격 스케일 단위

        surf.blit(f_md.render("AGV  TRACKER", True, C_ACCENT), (ox, oy))
        oy += int(20*sc); sep(oy); oy += int(10*sc)

        phase = snap.get("phase", "tracking")
        cycle = snap.get("cycle", 0)
        phase_colors = {
            "tracking": C_ACCENT,    "approach": C_ORANGE,
            "grab":     C_GREEN,     "navigate": (52,152,219),
            "release":  (155,89,182),"spin180":  (255,200,0),
            "returning":(255,150,50),"done":     C_DIM,
            "waiting":  C_DIM,
        }
        col = phase_colors.get(phase, C_DIM)
        surf.blit(f_md.render(f"PHASE: {phase.upper()}", True, col), (ox, oy))
        oy += int(20*sc)
        surf.blit(f_sm.render(f"CYCLE: {cycle}/{TOTAL_CYCLES}", True,
                  C_GREEN if cycle < TOTAL_CYCLES else C_DIM), (ox, oy))
        oy += int(16*sc)

        if snap.get("dest_label"):
            key  = snap["dest_label"].strip().lower()
            dcol = DEST_COLORS.get(key, C_DIM)
            txt  = f"→ {snap['dest_label'].upper()} ({int(snap.get('dest_x',0))},{int(snap.get('dest_y',0))})"
            surf.blit(f_sm.render(txt, True, dcol), (ox, oy)); oy += int(16*sc)
        if phase == "returning":
            surf.blit(f_sm.render("→ ORIGIN (0, 0)", True, C_ORANGE), (ox, oy))
            oy += int(16*sc)

        sep(oy); oy += int(8*sc)

        detected   = tsnap.get("detected", False)
        is_locked  = tsnap.get("locked", False)
        lock_label = tsnap.get("lock_label", "")

        if is_locked and phase == "tracking":
            lr = pygame.Rect(ox-4, oy-2, pw-m, int(20*sc))
            pygame.draw.rect(surf, C_LOCK, lr, 1)
            surf.blit(f_md.render(f"LOCKED: {lock_label.upper()}", True, C_LOCK), (ox, oy))
        else:
            c = C_GREEN if detected else C_RED
            surf.blit(f_md.render("TRACKING" if detected else "LOST   ", True, c), (ox, oy))
        oy += int(22*sc)

        if detected and phase == "tracking":
            lbl("TARGET", oy); oy += int(14*sc)
            val(tsnap.get("label","")[:12], oy, C_WHITE); oy += int(20*sc)
            lbl("CONF", oy); oy += int(14*sc)
            val(f"{tsnap.get('confidence',0):.2f}", oy, C_ACCENT); oy += int(20*sc)
            lbl("STEER", oy); oy += int(14*sc)
            st = tsnap.get("steer", 0)
            val(f"{st:+.1f}", oy, C_ORANGE if abs(st)>5 else C_GREEN); oy += int(14*sc)
            lbl("CY (px)", oy); oy += int(14*sc)
            cy_     = tsnap.get("cy", 0)
            label_  = tsnap.get("label", "")
            close_y = CLOSE_Y_THRESH_MAP.get(label_.strip().lower(), CLOSE_Y_THRESH_DEFAULT)
            val(f"{cy_}", oy, C_RED if cy_ > close_y else C_ACCENT); oy += int(22*sc)
        else:
            oy += int(94*sc)

        sep(oy); oy += int(10*sc)

        for lbl_txt, key_, big in [
            ("X POSITION (mm)", "x",        True),
            ("Y POSITION (mm)", "y",        True),
            ("YAW (deg)",        "yaw",      True),
            ("DISTANCE (mm)",    "distance", True),
        ]:
            lbl(lbl_txt, oy); oy += int(14*sc)
            raw = snap.get(key_, 0.0)
            fmt = f"{raw:+6.1f}" if key_ == "yaw" else f"{raw:6.0f}"
            val(fmt, oy, C_ORANGE if key_=="yaw" else C_ACCENT, big=True)
            oy += int(32*sc)

        sep(oy); oy += int(10*sc)
        lbl("SPEED L", oy); oy += int(14*sc)
        val(f"{snap.get('speed_l',0):+.0f}", oy); oy += int(22*sc)
        lbl("SPEED R", oy); oy += int(14*sc)
        val(f"{snap.get('speed_r',0):+.0f}", oy); oy += int(26*sc)

        sep(oy); oy += int(10*sc)
        lbl("HEADING", oy); oy += int(16*sc)
        compass_r = max(30, int(48*sc))
        compass_y = oy + compass_r + 4
        if compass_y + compass_r < field_h - int(60*sc):
            self._draw_compass(surf, fpx + pw//2, compass_y,
                               snap.get("yaw", 0), sc)
            oy = compass_y + compass_r + int(14*sc)

        sep(oy); oy += int(8*sc)
        surf.blit(f_sm.render(f"FPS  {fps:4.0f}", True, C_DIM), (ox, oy))

    def _draw_mission_buttons(self, surf, fpx, pw, field_h, sc):
        btn_w = max(80, int(106*sc))
        btn_h = max(24, int(34*sc))
        gap   = max(4, int(8*sc))
        total = btn_w*2 + gap
        lx    = fpx + (pw - total)//2
        by    = field_h - btn_h - max(6, int(10*sc))

        start_rect = pygame.Rect(lx, by, btn_w, btn_h)
        if self._mission_started:
            fill = (20,60,20); tc = C_DIM; label = "STARTED"
        else:
            fill = (20,120,40); tc = C_WHITE; label = "START"
        pygame.draw.rect(surf, fill,    start_rect, border_radius=6)
        pygame.draw.rect(surf, C_GREEN, start_rect, 2, border_radius=6)
        fnt = pygame.font.SysFont("monospace", max(9, int(15*sc)), bold=True)
        t = fnt.render(label, True, tc)
        surf.blit(t, t.get_rect(center=start_rect.center))

        reset_rect = pygame.Rect(lx + btn_w + gap, by, btn_w, btn_h)
        pygame.draw.rect(surf, (80,20,20), reset_rect, border_radius=6)
        pygame.draw.rect(surf, C_RED,      reset_rect, 2, border_radius=6)
        t = fnt.render("RESET", True, C_WHITE)
        surf.blit(t, t.get_rect(center=reset_rect.center))

        self._btn_start_rect = start_rect
        self._btn_reset_rect = reset_rect

    def _handle_button_click(self, pos):
        if self._btn_start_rect and self._btn_start_rect.collidepoint(pos):
            if not self._mission_started:
                self._mission_started = True
                self._send_cmd("start")
        if self._btn_reset_rect and self._btn_reset_rect.collidepoint(pos):
            self._mission_started = False
            self._trail_mm.clear()
            self._send_cmd("reset")

    def _draw_camera_preview(self, surf, frame_bgr, tsnap,
                             sw, sh, prev_h, sc):
        bar_y  = sh - prev_h
        prev_w = int(PREVIEW_W * sc)
        prev_h_img = prev_h - 10

        pygame.draw.rect(surf, C_PREVIEW, pygame.Rect(0, bar_y, sw, prev_h))
        pygame.draw.line(surf, C_BORDER, (0, bar_y), (sw, bar_y), 1)

        is_locked = tsnap.get("locked", False)

        if frame_bgr is not None:
            rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            small   = cv2.resize(rgb, (prev_w, prev_h_img))
            py_surf = pygame.surfarray.make_surface(
                          np.transpose(small, (1,0,2)))
            surf.blit(py_surf, (8, bar_y+4))
            if is_locked:
                pygame.draw.rect(surf, C_LOCK,
                                 (8, bar_y+4, prev_w, prev_h_img), 2)

        cx_px = 8 + prev_w//2
        pygame.draw.line(surf, (0,200,200),
                         (cx_px, bar_y+4), (cx_px, bar_y+4+prev_h_img), 1)

        fnt = pygame.font.SysFont("monospace", max(7, int(11*sc)))
        tx, ty = prev_w+24, bar_y+10
        def info(text, color=C_DIM):
            nonlocal ty
            surf.blit(fnt.render(text, True, color), (tx, ty))
            ty += max(12, int(18*sc))

        info("CAMERA FEED", C_ACCENT)
        if is_locked:
            info(f"LOCKED: {tsnap.get('lock_label','').upper()}", C_LOCK)
        else:
            info("OBJECT DETECTED" if tsnap.get("detected") else "NO TARGET",
                 C_GREEN if tsnap.get("detected") else C_RED)

        if tsnap.get("detected"):
            label_ = tsnap.get("label","")
            info(f"Label : {label_}")
            info(f"Conf  : {tsnap.get('confidence',0):.2f}")
            info(f"CX    : {tsnap.get('cx',0)} px")
            cy_     = tsnap.get("cy", 0)
            close_y = CLOSE_Y_THRESH_MAP.get(label_.strip().lower(), CLOSE_Y_THRESH_DEFAULT)
            info(f"CY    : {cy_} px  ({'CLOSE!' if cy_>close_y else 'far'})",
                 C_RED if cy_>close_y else C_DIM)
            info(f"Steer : {tsnap.get('steer',0):+.1f}")
            info(f"CLOSE Y > {close_y}px ({label_.upper() or 'DEFAULT'})", C_DIM)
        else:
            info(f"CLOSE Y > {CLOSE_Y_THRESH_DEFAULT}px (DEFAULT)", C_DIM)

    def _draw_statusbar(self, surf, sw, sh, sc):
        fnt   = pygame.font.SysFont("monospace", max(7, int(11*sc)))
        items = ["[C] CLEAR TRAIL", "[F] FULLSCREEN", "[ESC] QUIT",
                 f"TRAIL {len(self._trail_mm)} pts"]
        x = int((PREVIEW_W+24)*sc)
        y = sh - max(12, int(20*sc))
        for item in items:
            t = fnt.render(item, True, C_DIM)
            surf.blit(t, (x, y))
            x += t.get_width() + max(10, int(20*sc))


# ──────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()

    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()