import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class MyServiceClient(Node):
    def __init__(self):
        super().__init__('my_service_client')
        self.cli = self.create_client(Trigger, 'say_hello')

        # 서버가 준비될 때까지 대기
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('서버 기다리는 중...')
            
        self.get_logger().info('서버 연결됨!')

    def send_request(self):
        req = Trigger.Request()  # Trigger는 request 필드가 없음
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    node = MyServiceClient()

    # 키보드 입력 받기
    print('엔터를 누르면 서버에 요청을 보냅니다. (종료: Ctrl+C)')
    try:
        while True:
            input('>>> 엔터 입력: ')
            result = node.send_request()
            if result.success:
                print(f'서버 응답: {result.message}')
            else:
                print('서버 응답 실패')
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()