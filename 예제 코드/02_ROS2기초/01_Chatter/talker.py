import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.i = 0
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, world!: {0}'.format(self.i)
        self.i += 1
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.publisher_.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)

    node = Talker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    
if __name__=='__main__':
    main()