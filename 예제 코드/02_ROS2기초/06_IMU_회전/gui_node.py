import pygame
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class PygameNode(Node):
    def __init__(self):
        super().__init__('pygame_node')
        self.pub = self.create_publisher(Float32, 'target_angle', 10)
        self.get_logger().info('pygame_node 시작!')

    def publish_angle(self, angle):
        msg = Float32()
        msg.data = float(angle)
        self.pub.publish(msg)
        print(f'목표 각도 발행: {angle}°')

def main(args=None):
    rclpy.init(args=args)
    node = PygameNode()

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption('Angle Controller')
    font_large = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    input_text = ''   # 입력 중인 문자열
    active = True     # 입력창 활성화 여부

    # 버튼 영역
    btn_rect = pygame.Rect(140, 200, 120, 50)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_RETURN:
                        # 엔터도 start와 동일하게 동작
                        if input_text:
                            try:
                                angle = float(input_text)
                                node.publish_angle(angle)
                            except ValueError:
                                print('숫자를 입력하세요')
                    elif event.unicode.lstrip('-').replace('.','',1).isdigit() or event.unicode == '-':
                        input_text += event.unicode  # 숫자 + 음수 입력 허용

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_rect.collidepoint(event.pos):
                        if input_text:
                            try:
                                angle = float(input_text)
                                node.publish_angle(angle)
                            except ValueError:
                                print('숫자를 입력하세요')

            rclpy.spin_once(node, timeout_sec=0)

            # 화면 그리기
            screen.fill((30, 30, 30))

            # 안내 텍스트
            label = font_small.render('Target Angle (deg):', True, (200, 200, 200))
            screen.blit(label, (60, 60))

            # 입력창
            input_box = pygame.Rect(60, 100, 280, 50)
            pygame.draw.rect(screen, (255, 255, 255), input_box, 2)
            value_text = font_large.render(input_text or '0', True, (255, 255, 255))
            screen.blit(value_text, (input_box.x + 10, input_box.y + 8))

            # START 버튼
            pygame.draw.rect(screen, (0, 180, 100), btn_rect, border_radius=8)
            btn_text = font_small.render('START', True, (255, 255, 255))
            screen.blit(btn_text, (btn_rect.x + 22, btn_rect.y + 12))

            pygame.display.flip()
            clock.tick(30)

    except KeyboardInterrupt:
        pass

    pygame.quit()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()