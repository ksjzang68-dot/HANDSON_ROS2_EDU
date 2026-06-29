import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts  # 기본 제공 srv 재활용

from std_srvs.srv import Trigger  # request: 없음, response: success(bool) + message(string)

class MyServiceServer(Node):
    def __init__(self):
        super().__init__('my_service_server')
        # 서비스 이름: 'say_hello', Trigger 타입
        self.srv = self.create_service(Trigger, 'say_hello', self.handle_request)
        self.get_logger().info('서비스 서버 준비 완료! 클라이언트를 실행하세요.')

    def handle_request(self, request, response):
        self.get_logger().info('요청 받음!')
        response.success = True
        response.message = 'Hello from Service Server!'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = MyServiceServer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()