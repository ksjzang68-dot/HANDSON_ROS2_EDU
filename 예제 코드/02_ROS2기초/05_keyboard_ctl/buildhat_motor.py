from HandsON_BuildHat_API import *
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

robot = MotorPair('E', 'F')
SPEED = 50

class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')
        self.subscription = self.create_subscription(
            String,
            'motor_ctrl',    # ← 토픽 이름 (keyboard_pkg와 반드시 동일)
            self.listener_callback,
            10
        )
        self.get_logger().info('motor_node 시작!')

    def listener_callback(self, msg):
        cmd = msg.data
        self.get_logger().info(f'명령 받음: "{cmd}"')
        self.motor_move(cmd)

    def motor_move(self, cmd):
        if cmd == 'f':
            robot.start_tank(SPEED, SPEED)
            print('전진')
        elif cmd == 'b':
            robot.start_tank(-SPEED, -SPEED)
            print('후진')
        elif cmd == 'l':
            robot.start_tank(-SPEED, SPEED)
            print('좌회전')
        elif cmd == 'r':
            robot.start_tank(SPEED, -SPEED)
            print('우회전')
        elif cmd == 's':
            robot.stop()
            print('정지')
        else:
            self.get_logger().warn(f'모르는 명령: "{cmd}"')

def main(args=None):
    rclpy.init(args=args)
    node = MotorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()