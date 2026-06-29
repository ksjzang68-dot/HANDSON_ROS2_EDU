import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MyTalker(Node):
    def __init__(self):
        super().__init__('my_talker')
        self.publisher_ = self.create_publisher(String, 'my_chatter', 10)
        self.count = 0
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('My Talker Start!')

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, ROS@: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.count += 1
        
def main(args=None):
    rclpy.init(args=args)
    node = MyTalker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__=='__main__':
    main()