"""
IMU Node  (1단계와 동일)
========================
BNO055 → /imu/yaw (Float32, 10Hz)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    from bno055 import BNO055
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False


class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        self.pub = self.create_publisher(Float32, '/imu/yaw', 10)

        if HW_AVAILABLE:
            self.sensor = BNO055(bus=7)
        else:
            self.sensor = None
            self._sim_yaw = 0.0
            self.get_logger().warn('BNO055 없음 → 시뮬레이션 모드 (yaw=0)')

        self.timer = self.create_timer(0.1, self._timer_cb)
        self.get_logger().info('imu_node 시작!')

    def _timer_cb(self):
        if self.sensor:
            yaw, _, _ = self.sensor.euler
        else:
            yaw = self._sim_yaw

        msg      = Float32()
        msg.data = float(yaw)
        self.pub.publish(msg)

    def destroy_node(self):
        if self.sensor:
            self.sensor.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
