import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MyListener(Node):
    def __init__(self):
        super().__init__('my_listener')
        self.subscription = self.create_subscription(
            String, 
            'my_chatter', 
            self.listener_callback,
            10
        )
        self.get_logger().info('My Listener Start!')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: "{msg.data}"')
        
def main(args=None):
    rclpy.init(args=args)
    node = MyListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__=='__main__':
    main()