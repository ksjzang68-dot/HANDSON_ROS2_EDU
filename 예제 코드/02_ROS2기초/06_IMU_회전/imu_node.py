import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from IMU_회전.bno055 import BNO055   # ← 커스텀 라이브러리

class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')
        self.pub = self.create_publisher(Float32, 'current_angle', 10)

        self.sensor = BNO055(bus=7)   # ← bus=7 고정

        # 0.1초마다 각도 발행 (10Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('imu_node 시작!')

    def timer_callback(self):
        yaw, pitch, roll = self.sensor.euler   # ← (yaw, pitch, roll) 언패킹
        msg = Float32()
        msg.data = float(yaw)
        self.pub.publish(msg)
        self.get_logger().info(f'현재 yaw: {yaw:.2f}°')

    def destroy_node(self):
        self.sensor.close()   # ← 노드 종료 시 센서 close
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()