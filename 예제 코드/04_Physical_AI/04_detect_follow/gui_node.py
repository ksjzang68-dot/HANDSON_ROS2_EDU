"""
GUI Node  (2단계 — 분할 화면)
==============================
왼쪽 절반 : YOLO 카메라 + 바운딩박스 오버레이
오른쪽 절반: 오도메트리 맵 (이동 경로 실시간 표시)

구독
  /yolo/image_raw    CompressedImage  → 왼쪽 카메라 화면
  /yolo/detections   String(JSON)     → 바운딩박스 + rank
  /odom              Pose2D           → 맵 위 로봇 위치
  /motor/phase       String           → 상태 표시

스레드 구조
  rclpy.spin  → 별도 데몬 스레드 (콜백 수신)
  pygame 루프 → 메인 스레드 (pygame 제약)
  공유 데이터 → threading.Lock 으로 보호
"""

import json
import math
import threading
import cv2
import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D


# ── 색상 팔레트 ───────────────────────────────────────────────────

C_BG        = (12,  18,  30)
C_FIELD     = (16,  24,  40)
C_GRID_MAJ  = (28,  46,  70)
C_GRID_MIN  = (20,  32,  52)
C_BORDER    = (0,  180, 220)
C_TRAIL     = (0,  200, 255)
C_ROBOT     = (255, 107,  53)
C_HEADING   = (255, 220,  80)
C_ACCENT    = (0,  200, 255)
C_DIM       = (70, 100, 130)
C_WHITE     = (240, 245, 255)
C_ORANGE    = (255, 150,  50)
C_GREEN     = (46,  204, 113)
C_RED       = (210,  60,  60)
C_PANEL     = (10,  14,  24)
C_DIVIDER   = (0,  160, 200)

RANK_COLORS = [
    (255,  60,  60),
    (255, 160,  30),
    (255, 230,   0),
    ( 80, 220,  80),
    ( 30, 180, 255),
    ( 80,  80, 255),
    (180,  80, 255),
    (200, 200, 200),
]

def rank_color(rank: int):
    return RANK_COLORS[min(rank - 1, len(RANK_COLORS) - 1)]


# ── 윈도우 설정 ───────────────────────────────────────────────────

WIN_W   = 1400
WIN_H   = 720
HALF_W  = WIN_W // 2        # 왼쪽/오른쪽 너비
MAP_SIZE = WIN_H             # 맵 높이 = 윈도우 높이
TRAIL_MAX = 8000
FPS       = 60

# 맵 스케일: 1픽셀 = 몇 mm  (초기값, 자동 확장)
MAP_SCALE_INIT = 1.0         # 1px = 1mm  (1000mm 공간을 720px에)
MAP_MARGIN     = 40          # 맵 여백(px)


# ── GUI Node ──────────────────────────────────────────────────────

class GuiNode(Node):

    def __init__(self):
        super().__init__('gui_node')

        # ── 공유 상태 ─────────────────────────────────────────────
        self._lock       = threading.Lock()
        self._frame      = None
        self._detections = []
        self._robot_x    = 0.0
        self._robot_y    = 0.0
        self._robot_yaw  = 0.0
        self._phase      = 'idle'
        self._trail      = []   # [(x_mm, y_mm), ...]

        # 맵 뷰 (자동 확장용)
        self._map_min_x  = 0.0
        self._map_max_x  = 1000.0
        self._map_min_y  = 0.0
        self._map_max_y  = 1000.0

        # ── 구독 ──────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage, '/yolo/image_raw',   self._cb_img,   10)
        self.create_subscription(
            String,          '/yolo/detections',  self._cb_det,   10)
        self.create_subscription(
            Pose2D,          '/odom',             self._cb_odom,  10)
        self.create_subscription(
            String,          '/motor/phase',      self._cb_phase, 10)

        # ── pygame 초기화 ─────────────────────────────────────────
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption('AGV 2단계 — YOLO 추적 + 맵핑')

        self.font_lg = pygame.font.SysFont('monospace', 22, bold=True)
        self.font_md = pygame.font.SysFont('monospace', 15, bold=True)
        self.font_sm = pygame.font.SysFont('monospace', 12)
        self.rank_font = pygame.font.SysFont('Arial', 26, bold=True)
        self.fps_clock = pygame.time.Clock()

        self.get_logger().info('gui_node 시작!')

    # ── ROS 콜백 (스핀 스레드) ────────────────────────────────────

    def _cb_img(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._lock:
                self._frame = frame

    def _cb_det(self, msg: String):
        try:
            dets = json.loads(msg.data)
            with self._lock:
                self._detections = dets
        except json.JSONDecodeError:
            pass

    def _cb_odom(self, msg: Pose2D):
        with self._lock:
            self._robot_x   = msg.x
            self._robot_y   = msg.y
            self._robot_yaw = msg.theta

            pt = (msg.x, msg.y)
            if not self._trail or self._trail[-1] != pt:
                self._trail.append(pt)
                if len(self._trail) > TRAIL_MAX:
                    self._trail.pop(0)

            # 맵 범위 자동 확장
            pad = 80
            self._map_min_x = min(self._map_min_x, msg.x - pad)
            self._map_max_x = max(self._map_max_x, msg.x + pad)
            self._map_min_y = min(self._map_min_y, msg.y - pad)
            self._map_max_y = max(self._map_max_y, msg.y + pad)

    def _cb_phase(self, msg: String):
        with self._lock:
            self._phase = msg.data

    # ── pygame 메인루프 (메인 스레드) ─────────────────────────────

    def run_pygame_loop(self):
        running = True
        while running and rclpy.ok():
            # 이벤트
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_c:
                        with self._lock:
                            self._trail.clear()

            # 공유 상태 스냅샷
            with self._lock:
                frame      = self._frame.copy() if self._frame is not None else None
                detections = list(self._detections)
                robot_x    = self._robot_x
                robot_y    = self._robot_y
                robot_yaw  = self._robot_yaw
                phase      = self._phase
                trail      = list(self._trail)
                map_range  = (self._map_min_x, self._map_max_x,
                              self._map_min_y, self._map_max_y)

            sw, sh = self.screen.get_size()
            half   = sw // 2

            self.screen.fill(C_BG)

            # ── 왼쪽: 카메라 ──────────────────────────────────────
            self._draw_camera(frame, detections, half, sh)

            # ── 중앙 구분선 ───────────────────────────────────────
            pygame.draw.line(self.screen, C_DIVIDER, (half, 0), (half, sh), 2)

            # ── 오른쪽: 맵 ───────────────────────────────────────
            map_rect = pygame.Rect(half + 1, 0, sw - half - 1, sh)
            self._draw_map(map_rect, trail, robot_x, robot_y,
                           robot_yaw, phase, map_range)

            # HUD (공통)
            self._draw_hud(phase, robot_x, robot_y, robot_yaw,
                           len(detections), sw, sh)

            pygame.display.flip()
            self.fps_clock.tick(FPS)

        pygame.quit()
        rclpy.shutdown()

    # ── 왼쪽: 카메라 + 바운딩박스 ────────────────────────────────

    def _draw_camera(self, frame, detections, half_w, sh):
        cam_rect = pygame.Rect(0, 0, half_w, sh)

        if frame is None:
            pygame.draw.rect(self.screen, (20, 28, 45), cam_rect)
            msg = self.font_md.render('카메라 신호 대기 중...', True, C_DIM)
            self.screen.blit(msg, (half_w // 2 - msg.get_width() // 2,
                                   sh // 2))
            return

        fh, fw = frame.shape[:2]
        scale  = min(half_w / fw, sh / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        ox     = (half_w - nw) // 2
        oy     = (sh - nh) // 2

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh))
        surf    = pygame.surfarray.make_surface(resized.transpose(1, 0, 2))
        self.screen.blit(surf, (ox, oy))

        # 바운딩박스
        for d in detections:
            rank  = d.get('rank', 1)
            color = rank_color(rank)
            x1 = int(d['x1'] * scale) + ox
            y1 = int(d['y1'] * scale) + oy
            x2 = int(d['x2'] * scale) + ox
            y2 = int(d['y2'] * scale) + oy
            bw = 3 if rank == 1 else 1

            pygame.draw.rect(self.screen, color,
                             pygame.Rect(x1, y1, x2-x1, y2-y1), bw)

            # 레이블
            lbl = self.font_sm.render(
                f"#{rank} {d['label']} {d['conf']:.2f}", True, C_WHITE)
            lw, lh = lbl.get_size()
            bg_y   = max(y1 - lh - 6, 0)
            bg     = pygame.Surface((lw+8, lh+6), pygame.SRCALPHA)
            bg.fill((*color, 200))
            self.screen.blit(bg,  (x1, bg_y))
            self.screen.blit(lbl, (x1+4, bg_y+3))

            # rank 숫자 (박스 내부)
            rsurf = self.rank_font.render(str(rank), True, color)
            rw, rh = rsurf.get_size()
            nbg = pygame.Surface((rw+10, rh+6), pygame.SRCALPHA)
            nbg.fill((0, 0, 0, 160))
            self.screen.blit(nbg,   (x1+4, y1+4))
            self.screen.blit(rsurf, (x1+9, y1+7))

            # rank=1: 화면 중심선 → 박스 중심 벡터 표시
            if rank == 1:
                cx_screen = int(d['cx'] * scale) + ox
                center_x  = half_w // 2
                cy_screen = oy + nh // 2
                pygame.draw.line(self.screen, C_ORANGE,
                                 (center_x, cy_screen),
                                 (cx_screen, cy_screen), 2)
                pygame.draw.circle(self.screen, C_ORANGE,
                                   (cx_screen, cy_screen), 6)

        # 카메라 중심선
        pygame.draw.line(self.screen, (*C_DIM, 120),
                         (half_w // 2, 0), (half_w // 2, sh), 1)

        # 정지 기준선 (박스 높이 70% 위치)
        stop_y = int(sh * (1 - 0.70) / 2)
        pygame.draw.line(self.screen, C_RED,
                         (0, stop_y), (half_w, stop_y), 1)
        pygame.draw.line(self.screen, C_RED,
                         (0, sh - stop_y), (half_w, sh - stop_y), 1)
        hint = self.font_sm.render('STOP LINE (70%)', True, C_RED)
        self.screen.blit(hint, (4, stop_y + 4))

    # ── 오른쪽: 오도메트리 맵 ────────────────────────────────────

    def _draw_map(self, rect, trail, rx, ry, ryaw, phase, map_range):
        """
        rect       : 맵이 그려질 pygame.Rect (오른쪽 절반)
        trail      : [(x_mm, y_mm), ...]
        map_range  : (min_x, max_x, min_y, max_y) in mm — 자동 확장
        """
        # 배경
        pygame.draw.rect(self.screen, C_FIELD, rect)

        mx, my  = rect.x, rect.y
        mw, mh  = rect.width, rect.height
        pad     = MAP_MARGIN

        min_x, max_x, min_y, max_y = map_range
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)

        # 종횡비 유지 스케일
        scale = min((mw - pad*2) / span_x, (mh - pad*2) / span_y)

        # 실제 그리기 영역 오프셋
        draw_w = span_x * scale
        draw_h = span_y * scale
        ox = mx + pad + (mw - pad*2 - draw_w) / 2
        oy = my + pad + (mh - pad*2 - draw_h) / 2

        def to_screen(fx, fy):
            sx = ox + (fx - min_x) * scale
            sy = oy + draw_h - (fy - min_y) * scale
            return int(sx), int(sy)

        # 그리드 (100mm 간격)
        step = 100
        gx = (int(min_x) // step) * step
        while gx <= max_x + step:
            p1 = to_screen(gx, min_y)
            p2 = to_screen(gx, max_y)
            col = C_GRID_MAJ if gx % 500 == 0 else C_GRID_MIN
            pygame.draw.line(self.screen, col, p1, p2)
            # 눈금 레이블
            if gx % 500 == 0:
                lbl = self.font_sm.render(f'{gx}', True, C_DIM)
                self.screen.blit(lbl, (p1[0]-10, p1[1]+2))
            gx += step

        gy = (int(min_y) // step) * step
        while gy <= max_y + step:
            p1 = to_screen(min_x, gy)
            p2 = to_screen(max_x, gy)
            col = C_GRID_MAJ if gy % 500 == 0 else C_GRID_MIN
            pygame.draw.line(self.screen, col, p1, p2)
            if gy % 500 == 0:
                lbl = self.font_sm.render(f'{gy}', True, C_DIM)
                self.screen.blit(lbl, (p1[0]+2, p1[1]-12))
            gy += step

        # 테두리
        border = pygame.Rect(int(ox), int(oy),
                             int(draw_w)+1, int(draw_h)+1)
        pygame.draw.rect(self.screen, C_BORDER, border, 1)

        # 경로
        if len(trail) >= 2:
            n = len(trail)
            pts = [to_screen(p[0], p[1]) for p in trail]
            for i in range(1, n):
                t   = i / n
                col = (0, int(100 + 100*t), int(180 + 75*t))
                w   = 2 if t > 0.6 else 1
                pygame.draw.line(self.screen, col, pts[i-1], pts[i], w)
            # 시작점
            pygame.draw.circle(self.screen, C_GREEN, pts[0], 5)
            pygame.draw.circle(self.screen, C_WHITE, pts[0], 5, 1)

        # 로봇
        rsx, rsy = to_screen(rx, ry)
        R   = 12
        pygame.draw.circle(self.screen, C_ROBOT, (rsx, rsy), R, 2)

        rad   = math.radians(ryaw)
        tip_x = rsx + int(math.sin(rad) * (R+6))
        tip_y = rsy - int(math.cos(rad) * (R+6))
        bl_x  = rsx + int(math.sin(rad+2.3) * (R-2))
        bl_y  = rsy - int(math.cos(rad+2.3) * (R-2))
        br_x  = rsx + int(math.sin(rad-2.3) * (R-2))
        br_y  = rsy - int(math.cos(rad-2.3) * (R-2))
        pygame.draw.polygon(self.screen, C_HEADING,
                            [(tip_x,tip_y),(bl_x,bl_y),(br_x,br_y)])
        pygame.draw.circle(self.screen, C_WHITE, (rsx, rsy), 3)

        # 패널 정보 (맵 우상단)
        info_x = mx + mw - 170
        info_y = my + 12
        def minfo(s, color=C_DIM):
            nonlocal info_y
            self.screen.blit(self.font_sm.render(s, True, color), (info_x, info_y))
            info_y += 15

        minfo('[ 매핑 ]', C_ACCENT)
        minfo(f'X  {rx:8.1f} mm', C_WHITE)
        minfo(f'Y  {ry:8.1f} mm', C_WHITE)
        minfo(f'YAW  {ryaw:+6.1f}°', C_ORANGE)
        minfo(f'궤적  {len(trail)} pts', C_DIM)
        minfo(f'[ C ] 경로 지우기', C_DIM)

    # ── HUD ──────────────────────────────────────────────────────

    def _draw_hud(self, phase, rx, ry, ryaw, n_det, sw, sh):
        phase_col = C_DIM if phase == 'idle' \
            else C_GREEN if phase == 'arrived' \
            else C_ORANGE

        texts = [
            (f'FPS: {self.fps_clock.get_fps():.1f}', C_ACCENT, 10, 10),
            (f'Objects: {n_det}', C_WHITE, 10, 28),
            (f'Phase: {phase.upper()}', phase_col, 10, 46),
        ]
        for t, col, tx, ty in texts:
            s = self.font_sm.render(t, True, col)
            self.screen.blit(s, (tx, ty))

    def destroy_node(self):
        pygame.quit()
        super().destroy_node()


# ── 엔트리포인트 ──────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = GuiNode()

    # rclpy.spin → 별도 데몬 스레드
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_pygame_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
