# action_server.py
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci

class MyActionServer(Node):
    def __init__(self):
        super().__init__('action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'countdown',
            self.execute_callback
        )
        self.get_logger().info('액션 서버 준비 완료!')

    def execute_callback(self, goal_handle):
        count = goal_handle.request.order
        self.get_logger().info(f'목표 받음: {count}부터 카운트다운')

        result_list = []
        for i in range(count, 0, -1):
            self.get_logger().info(f'카운트다운: {i}')
            feedback_msg = Fibonacci.Feedback()
            feedback_msg.sequence = [i]  # ← sequence로 수정
            goal_handle.publish_feedback(feedback_msg)
            result_list.append(i)
            time.sleep(1.0)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = result_list
        self.get_logger().info('카운트다운 완료!')
        return result

def main(args=None):
    rclpy.init(args=args)
    node = MyActionServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()