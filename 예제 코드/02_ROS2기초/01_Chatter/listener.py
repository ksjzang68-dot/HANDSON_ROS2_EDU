import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String

class Listener(Node):
    
    def __init__(self):
        super().__init__('listener')
        self.sub = self.creat_subscription(String, 'chatter', self.chatter_callback, 10)


    def chatter_callback(self, msg):
        self.get_logger().info('I heard: [%s]' % msg.data)
        
def main(args=None):
    rclpy.init(args=args)

    node = Listener()
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