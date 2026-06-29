import pygame
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

KEY_MAP = {
    pygame.K_UP:    'f',
    pygame.K_DOWN:  'b',
    pygame.K_LEFT:  'l',
    pygame.K_RIGHT: 'r',
}

class KeyboardNode(Node):
    def __init__(self):
        super().__init__('keyboard_node')
        self.pub = self.create_publisher(String, 'motor_ctrl', 10)  # ← 토픽 이름
        self.last_cmd = None
        self.get_logger().info('keyboard_node 시작!')

    def publish_cmd(self, cmd):
        if cmd == self.last_cmd:
            return
        msg = String()
        msg.data = cmd
        self.pub.publish(msg)
        self.last_cmd = cmd
        print(f'발행: {cmd}')

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardNode()

    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption('ROS2 Teleop')
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    text = font.render('Press arrow keys', True, (255, 255, 255))
    text_rect = text.get_rect(center=(200, 100))

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            keys = pygame.key.get_pressed()
            cmd = 's'
            for key, direction in KEY_MAP.items():
                if keys[key]:
                    cmd = direction
                    break

            node.publish_cmd(cmd)
            rclpy.spin_once(node, timeout_sec=0)

            screen.fill((30, 30, 30))
            screen.blit(text, text_rect)
            pygame.display.flip()
            clock.tick(30)

    except KeyboardInterrupt:
        pass

    node.publish_cmd('s')
    pygame.quit()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()