import json
import threading
import cv2
import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


# ── 색상 팔레트 ───────────────────────────────────────────────────

C_BG      = (12,  18,  30)
C_DIM     = (70, 100, 130)
C_WHITE   = (240, 245, 255)
C_ORANGE  = (255, 150,  50)
C_ACCENT  = (0,  200, 255)

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

WIN_W = 960
WIN_H = 720
FPS   = 60


# ── GUI Node ──────────────────────────────────────────────────────

class GuiNode(Node):

    def __init__(self):
        super().__init__('gui_node')

        # ── 공유 상태 ─────────────────────────────────────────────
        self._lock       = threading.Lock()
        self._frame      = None
        self._detections = []

        # ── 구독 ──────────────────────────────────────────────────
        self.create_subscription(
            CompressedImage, '/yolo/image_raw',  self._cb_img, 10)
        self.create_subscription(
            String,          '/yolo/detections', self._cb_det, 10)

        # ── pygame 초기화 ─────────────────────────────────────────
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption('AGV — YOLO 박스 뷰어')

        self.font_sm   = pygame.font.SysFont('monospace', 12)
        self.rank_font = pygame.font.SysFont('Arial', 26, bold=True)
        self.fps_clock = pygame.time.Clock()

        self.get_logger().info('gui_node 시작!')

    # ── ROS 콜백 (스핀 스레드) ────────────────────────────────────

    def _cb_img(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
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

    # ── pygame 메인루프 (메인 스레드) ─────────────────────────────

    def run_pygame_loop(self):
        running = True
        while running and rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

            with self._lock:
                frame      = self._frame.copy() if self._frame is not None else None
                detections = list(self._detections)

            sw, sh = self.screen.get_size()

            self.screen.fill(C_BG)
            self._draw_camera(frame, detections, sw, sh)
            self._draw_hud(len(detections))

            pygame.display.flip()
            self.fps_clock.tick(FPS)

        pygame.quit()
        rclpy.shutdown()

    # ── 카메라 + 바운딩박스 ─────────────────────────────────────

    def _draw_camera(self, frame, detections, sw, sh):
        cam_rect = pygame.Rect(0, 0, sw, sh)

        if frame is None:
            pygame.draw.rect(self.screen, (20, 28, 45), cam_rect)
            msg = self.font_sm.render('카메라 신호 대기 중...', True, C_DIM)
            self.screen.blit(msg, (sw // 2 - msg.get_width() // 2, sh // 2))
            return

        fh, fw = frame.shape[:2]
        scale  = min(sw / fw, sh / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        ox     = (sw - nw) // 2
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

            # 무게중심 표시 (cx, cy)
            if 'cx' in d and 'cy' in d:
                cxs = int(d['cx'] * scale) + ox
                cys = int(d['cy'] * scale) + oy
                pygame.draw.circle(self.screen, C_ORANGE, (cxs, cys), 5)
                pygame.draw.circle(self.screen, C_WHITE,  (cxs, cys), 5, 1)

    # ── HUD ──────────────────────────────────────────────────────

    def _draw_hud(self, n_det):
        texts = [
            (f'FPS: {self.fps_clock.get_fps():.1f}', C_ACCENT, 10, 10),
            (f'Objects: {n_det}',                    C_WHITE,  10, 28),
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