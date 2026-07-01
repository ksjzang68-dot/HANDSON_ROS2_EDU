import json
import threading
import cv2
import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

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

def rank_color(rank):
    return RANK_COLORS[min(rank - 1, len(RANK_COLORS) - 1)]

class PygameDisplayNode(Node):
    def __init__(self):
        super().__init__("gui_node")

        self.declare_parameter("window_width",   1280)
        self.declare_parameter("window_height",  720)

        w = self.get_parameter("window_width").value
        h = self.get_parameter("window_height").value

        self._lock       = threading.Lock()
        self._frame      = None
        self._detections = []

        self.create_subscription(CompressedImage, "/yolo/image_raw",   self._img_cb,  10)
        self.create_subscription(String,          "/yolo/detections",  self._det_cb,  10)

        pygame.init()
        self.screen     = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption("YOLO Detection")
        self.font       = pygame.font.SysFont("Arial", 18, bold=True)
        self.rank_font  = pygame.font.SysFont("Arial", 28, bold=True)
        self.clock      = pygame.time.Clock()
        self.get_logger().info("pygame_display_node 시작!")

    def _img_cb(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._lock:
                self._frame = frame

    def _det_cb(self, msg):
        try:
            with self._lock:
                self._detections = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def run_pygame_loop(self):
        running = True
        while running and rclpy.ok():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            with self._lock:
                frame      = self._frame.copy() if self._frame is not None else None
                detections = list(self._detections)

            if frame is None:
                self.screen.fill((30, 30, 30))
                msg = self.font.render("카메라 신호 대기 중...", True, (180, 180, 180))
                sw, sh = self.screen.get_size()
                self.screen.blit(msg, (sw // 2 - msg.get_width() // 2, sh // 2))
                pygame.display.flip()
                self.clock.tick(10)
                continue

            # 프레임 → Surface (창 크기에 맞게 스케일)
            sw, sh = self.screen.get_size()
            fh, fw = frame.shape[:2]
            scale  = min(sw / fw, sh / fh)
            nw, nh = int(fw * scale), int(fh * scale)
            ox, oy = (sw - nw) // 2, (sh - nh) // 2

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (nw, nh))
            surface = pygame.surfarray.make_surface(resized.transpose(1, 0, 2))

            self.screen.fill((0, 0, 0))
            self.screen.blit(surface, (ox, oy))

            self._draw(self.screen, detections, ox, oy, scale)

            # HUD
            self.screen.blit(self.font.render(f"FPS: {self.clock.get_fps():.1f}", True, (255, 220, 0)), (10, 10))
            self.screen.blit(self.font.render(f"Objects: {len(detections)}", True, (200, 255, 200)), (10, 32))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        rclpy.shutdown()

    def _draw(self, surface, detections, ox, oy, scale):
        for d in detections:
            rank  = d["rank"]
            color = rank_color(rank)
            x1 = int(d["x1"] * scale) + ox
            y1 = int(d["y1"] * scale) + oy
            x2 = int(d["x2"] * scale) + ox
            y2 = int(d["y2"] * scale) + oy

            pygame.draw.rect(surface, color, pygame.Rect(x1, y1, x2-x1, y2-y1), 2)

            # 레이블 (박스 위)
            label_surf = self.font.render(f"#{rank}  {d['label']}  {d['conf']:.2f}", True, (255,255,255))
            lw, lh = label_surf.get_size()
            bg_y   = max(y1 - lh - 6, 0)
            bg     = pygame.Surface((lw+8, lh+6), pygame.SRCALPHA)
            bg.fill((*color, 200))
            surface.blit(bg, (x1, bg_y))
            surface.blit(label_surf, (x1+4, bg_y+3))

            # 순위 숫자 (박스 내부)
            rank_surf = self.rank_font.render(str(rank), True, color)
            rw, rh    = rank_surf.get_size()
            num_bg    = pygame.Surface((rw+10, rh+6), pygame.SRCALPHA)
            num_bg.fill((0, 0, 0, 160))
            surface.blit(num_bg,    (x1+4, y1+4))
            surface.blit(rank_surf, (x1+9, y1+7))

    def destroy_node(self):
        pygame.quit()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PygameDisplayNode()

    # rclpy.spin → 별도 스레드 (Pygame은 메인 스레드 필수)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run_pygame_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()