import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from bno055 import BNO055

class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        # ── 토픽 이름 ─────────────────────────────────────────────
        # motor_node 와 gui_node 가 '/imu/yaw' 를 구독하므로
        # 반드시 '/imu/yaw' 로 맞춰야 세 노드가 연결됨
        self.pub = self.create_publisher(Float32, '/imu/yaw', 10)

        self.sensor = BNO055(bus=7)   # Jetson Orin Nano I2C bus 번호

        # 0.1초마다 각도 발행 (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('imu_node 시작!')

    def timer_callback(self):
        yaw, pitch, roll = self.sensor.euler
        msg = Float32()
        msg.data = float(yaw)
        self.pub.publish(msg)
        self.get_logger().info(f'현재 yaw: {yaw:.2f}°')

    def destroy_node(self):
        self.sensor.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()