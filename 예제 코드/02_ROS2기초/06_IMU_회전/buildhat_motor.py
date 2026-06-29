from SpikePI import *
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

robot = MotorPair('E', 'F')
SPEED = 30
THRESHOLD = 5.0

class MotorNode(Node):
    def __init__(self):
        super().__init__('motor_node')
        self.target  = None
        self.current = None
        self.reached = False   # ← 도달 여부 플래그 추가

        self.sub_target  = self.create_subscription(
            Float32, 'target_angle',  self.target_callback,  10)
        self.sub_current = self.create_subscription(
            Float32, 'current_angle', self.current_callback, 10)

        self.get_logger().info('motor_node 시작!')

    def target_callback(self, msg):
        self.target  = msg.data
        self.reached = False   # ← 새 목표 오면 플래그 리셋
        self.get_logger().info(f'새 목표 각도: {self.target}°')

    def current_callback(self, msg):
        self.current = msg.data
        self.control()

    def control(self):
        if self.target is None or self.current is None:
            return

        if self.reached:       # ← 이미 도달했으면 아무것도 안 함
            return

        error = self.target - self.current

        # -180 ~ 180 정규화
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360

        if abs(error) <= THRESHOLD:
            robot.stop()
            self.reached = True   # ← 도달 플래그 세팅
            self.get_logger().info(f'목표 도달! 현재: {self.current:.1f}° (오차: {error:.1f}°)')

        elif error > 0:
            robot.start_tank(SPEED, -SPEED)
            self.get_logger().info(f'우회전 중... 오차: {error:.1f}°')

        else:
            robot.start_tank(-SPEED, SPEED)
            self.get_logger().info(f'좌회전 중... 오차: {error:.1f}°')

def main(args=None):
    rclpy.init(args=args)
    node = MotorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()