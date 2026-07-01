"""
GUI Node
========
역할
  - /odom      구독 → 로봇 위치·경로 맵에 표시
  - /goal_pose 퍼블리시 → 마우스 클릭 좌표 전송
  - /motor/phase 구독 → 상태 패널 표시

교육 포인트
-----------
- pygame 메인루프를 ROS2 타이머로 대체 (30Hz)
- pygame.event.get() 을 콜백 안에서 호출 (블로킹 X)
- ROS2 spin_once 대신 타이머+spin 조합 사용

주의사항
--------
- pygame은 메인 스레드에서만 실행 가능
- rclpy.spin() 이 메인 스레드를 점유하므로
  MultiThreadedExecutor + pygame timer 콜백으로 처리
"""

import math
import sys
import pygame
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Pose2D

# ── 설정 ──────────────────────────────────────────────────────────

MAP_W      = 1000
MAP_H      = 1000
FIELD_MM   = 1000.0
PANEL_W    = 280
WIN_W      = MAP_W + PANEL_W
WIN_H      = MAP_H
FPS        = 30
TRAIL_MAX  = 6000

# 색상
C_BG       = (15,  20,  30)
C_FIELD    = (18,  26,  42)
C_GRID_MAJ = (30,  50,  75)
C_GRID_MIN = (22,  35,  55)
C_BORDER   = (0,  180, 220)
C_TRAIL    = (0,  200, 255)
C_ROBOT    = (255, 107,  53)
C_HEADING  = (255, 220,  80)
C_TARGET   = (46,  204, 113)
C_PANEL    = (10,  16,  28)
C_ACCENT   = (0,  200, 255)
C_DIM      = (70, 100, 130)
C_WHITE    = (240, 245, 255)
C_ORANGE   = (255, 140,  60)
C_GREEN    = (46,  204, 113)
C_RED      = (210,  60,  60)


# ── 좌표 변환 ─────────────────────────────────────────────────────

def field_to_screen(fx, fy):
    return int(fx), int(MAP_H - fy)

def screen_to_field(sx, sy):
    return float(sx), float(MAP_H - sy)


# ── GUI Node ──────────────────────────────────────────────────────

class GuiNode(Node):

    def __init__(self):
        super().__init__('gui_node')

        # ── 상태 ──────────────────────────────────────────────────
        self.robot_x    = 100.0
        self.robot_y    = 100.0
        self.robot_yaw  = 0.0
        self.phase      = 'idle'
        self.trail      = []
        self.target_x   = None
        self.target_y   = None
        self.pending_goal = None   # 클릭 후 GO 전 대기

        # ── 구독 ──────────────────────────────────────────────────
        self.sub_odom  = self.create_subscription(
            Pose2D, '/odom', self._cb_odom, 10)

        self.sub_phase = self.create_subscription(
            String, '/motor/phase', self._cb_phase, 10)

        # ── 퍼블리셔 ─────────────────────────────────────────────
        self.pub_goal = self.create_publisher(Pose2D, '/goal_pose', 10)

        # ── pygame 초기화 ─────────────────────────────────────────
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption('AGV 1단계 ROS2 — GUI Node')
        self.clock  = pygame.font.SysFont('monospace', 11)   # 임시

        self.font_lg = pygame.font.SysFont('monospace', 24, bold=True)
        self.font_md = pygame.font.SysFont('monospace', 14, bold=True)
        self.font_sm = pygame.font.SysFont('monospace', 11)
        self.fps_clock = pygame.time.Clock()

        # 배경 (정적)
        self.map_bg = pygame.Surface((MAP_W, MAP_H))
        self.map_bg.fill(C_FIELD)
        self._draw_grid(self.map_bg)
        self._draw_axes(self.map_bg)
        pygame.draw.rect(self.map_bg, C_BORDER, (0, 0, MAP_W, MAP_H), 2)

        self._running = True

        # ── pygame 루프를 30Hz 타이머로 실행 ─────────────────────
        self.render_timer = self.create_timer(1.0 / FPS, self._render_tick)

        self.get_logger().info('gui_node 시작')

    # ── ROS2 구독 콜백 ────────────────────────────────────────────

    def _cb_odom(self, msg: Pose2D):
        """motor_node에서 오도메트리 수신"""
        self.robot_x   = msg.x
        self.robot_y   = msg.y
        self.robot_yaw = msg.theta

        pt = field_to_screen(self.robot_x, self.robot_y)
        if not self.trail or self.trail[-1] != pt:
            self.trail.append(pt)
            if len(self.trail) > TRAIL_MAX:
                self.trail.pop(0)

    def _cb_phase(self, msg: String):
        self.phase = msg.data

    # ── 목표 좌표 퍼블리시 ────────────────────────────────────────

    def _publish_goal(self, fx: float, fy: float):
        msg   = Pose2D()
        msg.x = fx
        msg.y = fy
        self.pub_goal.publish(msg)
        self.target_x = fx
        self.target_y = fy
        self.get_logger().info(f'목표 발행: ({fx:.0f}, {fy:.0f}) mm')

    # ── 렌더 타이머 콜백 (30Hz) ──────────────────────────────────

    def _render_tick(self):
        """
        ★ 핵심 패턴 ★
        pygame 이벤트 처리 + 화면 그리기를 ROS2 타이머 콜백으로.
        블로킹 없이 매 tick 빠르게 반환.
        """
        if not self._running:
            return

        go_rect = pygame.Rect(
            MAP_W + (PANEL_W - 120) // 2,
            WIN_H - 60, 120, 44)

        # ── pygame 이벤트 처리 ────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                rclpy.shutdown()
                return

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._running = False
                    rclpy.shutdown()
                    return
                elif event.key == pygame.K_c:
                    self.trail.clear()
                elif event.key in (pygame.K_SPACE, pygame.K_g):
                    if self.pending_goal and self.phase == 'idle':
                        self._publish_goal(*self.pending_goal)
                        self.pending_goal = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if mx < MAP_W:
                    # 맵 클릭 → 대기 목표로 설정
                    fx, fy = screen_to_field(mx, my)
                    fx = max(0.0, min(FIELD_MM, fx))
                    fy = max(0.0, min(FIELD_MM, fy))
                    self.pending_goal = (fx, fy)
                    self.target_x = fx
                    self.target_y = fy
                elif go_rect.collidepoint(mx, my):
                    if self.pending_goal and self.phase == 'idle':
                        self._publish_goal(*self.pending_goal)
                        self.pending_goal = None

        # ── 화면 렌더링 ───────────────────────────────────────────
        self.screen.fill(C_BG)
        self.screen.blit(self.map_bg, (0, 0))

        self._draw_trail()
        self._draw_target()
        self._draw_robot()
        self._draw_panel(go_rect)

        pygame.display.flip()
        self.fps_clock.tick(FPS)

    # ── 그리기 함수 ───────────────────────────────────────────────

    def _draw_grid(self, surf):
        for i in range(0, int(FIELD_MM) + 1, 50):
            col = C_GRID_MAJ if i % 200 == 0 else C_GRID_MIN
            sx  = int(i)
            sy  = int(MAP_H - i)
            pygame.draw.line(surf, col, (sx, 0),    (sx, MAP_H))
            pygame.draw.line(surf, col, (0, sy),    (MAP_W, sy))

    def _draw_axes(self, surf):
        ox, oy = field_to_screen(0, 0)
        pygame.draw.line(surf, (180, 60, 60),  (ox, oy), (ox+40, oy), 2)
        pygame.draw.line(surf, (60, 180, 60),  (ox, oy), (ox, oy-40), 2)
        f = pygame.font.SysFont('monospace', 11)
        surf.blit(f.render('X', True, (200,80,80)),  (ox+44, oy-8))
        surf.blit(f.render('Y', True, (80,200,80)),  (ox+4,  oy-52))
        surf.blit(f.render('(0,0)', True, C_DIM),    (ox+4,  oy+4))

    def _draw_trail(self):
        if len(self.trail) < 2:
            return
        n = len(self.trail)
        for i in range(1, n):
            t = i / n
            col = (0, int(120+80*t), int(180+75*t))
            w   = 2 if t > 0.6 else 1
            pygame.draw.line(self.screen, col,
                             self.trail[i-1], self.trail[i], w)
        pygame.draw.circle(self.screen, C_TARGET, self.trail[0], 6)
        pygame.draw.circle(self.screen, C_WHITE,  self.trail[0], 6, 1)

    def _draw_target(self):
        if self.target_x is None:
            return
        tx, ty = field_to_screen(self.target_x, self.target_y)
        pygame.draw.circle(self.screen, C_TARGET, (tx, ty), 10, 2)
        pygame.draw.line(self.screen, C_TARGET, (tx-14,ty), (tx+14,ty), 1)
        pygame.draw.line(self.screen, C_TARGET, (tx,ty-14), (tx,ty+14), 1)
        lbl = self.font_sm.render(
            f'({self.target_x:.0f}, {self.target_y:.0f})', True, C_TARGET)
        self.screen.blit(lbl, (tx+14, ty-8))

    def _draw_robot(self):
        rx, ry = field_to_screen(self.robot_x, self.robot_y)
        R = 14
        pygame.draw.circle(self.screen, C_ROBOT, (rx, ry), R, 2)
        rad    = math.radians(self.robot_yaw)
        tip_x  = rx + int(math.sin(rad) * (R+6))
        tip_y  = ry - int(math.cos(rad) * (R+6))
        bl_x   = rx + int(math.sin(rad+2.3) * (R-2))
        bl_y   = ry - int(math.cos(rad+2.3) * (R-2))
        br_x   = rx + int(math.sin(rad-2.3) * (R-2))
        br_y   = ry - int(math.cos(rad-2.3) * (R-2))
        pygame.draw.polygon(self.screen, C_HEADING,
                            [(tip_x,tip_y),(bl_x,bl_y),(br_x,br_y)])
        pygame.draw.circle(self.screen, C_WHITE, (rx, ry), 3)

    def _draw_panel(self, go_rect):
        ox, oy = MAP_W + 12, 14
        pygame.draw.rect(self.screen, C_PANEL, (MAP_W, 0, PANEL_W, WIN_H))
        pygame.draw.line(self.screen, C_BORDER, (MAP_W,0),(MAP_W,WIN_H), 1)

        def sep(y):
            pygame.draw.line(self.screen, C_BORDER,
                             (MAP_W+8, y), (MAP_W+PANEL_W-8, y), 1)

        def txt(s, y, color=C_DIM, big=False):
            f = self.font_md if big else self.font_sm
            self.screen.blit(f.render(s, True, color), (ox, y))

        txt('AGV  ROS2  1단계', oy, C_ACCENT, big=True); oy += 24
        sep(oy); oy += 10

        txt('[ ROS2 토픽 ]', oy); oy += 14
        txt('/imu/yaw   → motor_node', oy); oy += 14
        txt('/odom      → gui_node',   oy); oy += 14
        txt('/goal_pose → motor_node', oy); oy += 18
        sep(oy); oy += 10

        txt('현재 위치 (mm)', oy); oy += 14
        txt(f'X  {self.robot_x:7.1f}', oy, C_ACCENT, big=True); oy += 28
        txt(f'Y  {self.robot_y:7.1f}', oy, C_ACCENT, big=True); oy += 28
        txt(f'YAW  {self.robot_yaw:+6.1f} deg', oy, C_ORANGE, big=True); oy += 32
        sep(oy); oy += 10

        txt('목표 좌표 (mm)', oy); oy += 14
        if self.target_x is not None:
            txt(f'X  {self.target_x:7.1f}', oy, C_TARGET, big=True); oy += 28
            txt(f'Y  {self.target_y:7.1f}', oy, C_TARGET, big=True); oy += 28
        else:
            txt('클릭으로 지정', oy, C_DIM); oy += 56
        sep(oy); oy += 10

        phase_col = {'idle':'(70,100,130)','turning':None,'driving':None}.get(
            self.phase)
        pc = C_DIM if self.phase=='idle' else \
             C_ORANGE if self.phase=='turning' else C_GREEN
        txt(f'상태: {self.phase.upper()}', oy, pc, big=True); oy += 32
        sep(oy); oy += 10

        txt('[ 클릭 ]  목표 지정',     oy); oy += 15
        txt('[ SPACE/G ]  이동 시작',  oy); oy += 15
        txt('[ C ]  경로 지우기',      oy); oy += 15
        txt('[ ESC/Q ]  종료',         oy); oy += 15

        # GO 버튼
        active  = (self.pending_goal is not None) and (self.phase == 'idle')
        btn_col = C_GREEN if active else C_DIM
        txt_col = C_WHITE if active else (40,60,80)
        pygame.draw.rect(self.screen, btn_col, go_rect, border_radius=8)
        pygame.draw.rect(self.screen, C_WHITE if active else C_DIM,
                         go_rect, 1, border_radius=8)
        lbl = self.font_md.render('GO  ▶', True, txt_col)
        self.screen.blit(lbl, lbl.get_rect(center=go_rect.center))

    def destroy_node(self):
        pygame.quit()
        super().destroy_node()


# ── 엔트리포인트 ──────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = GuiNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()