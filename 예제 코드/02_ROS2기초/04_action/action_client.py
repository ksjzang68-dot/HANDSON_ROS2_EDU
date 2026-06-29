# action_client.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from example_interfaces.action import Fibonacci

class MyActionClient(Node):
    def __init__(self):
        super().__init__('action_client')
        self._client = ActionClient(self, Fibonacci, 'countdown')

    def send_goal(self, number):
        self._client.wait_for_server()
        goal_msg = Fibonacci.Goal()
        goal_msg.order = number
        print(f'{number}부터 카운트다운 시작!')
        
        self._send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('목표 거절됨')
            return
        
        print('목표 수락됨! 피드백 기다리는 중...')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        count = feedback_msg.feedback.sequence[0]  # ← sequence로 수정
        print(f' [피드백] 카운트다운: {count}')

    def result_callback(self, future):
        result = future.result().result
        print(f'[완료] 카운트다운 순서: {result.sequence}')
        print('액션 종료!')

def main(args=None):
    rclpy.init(args=args)
    node = MyActionClient()
    print('카운트다운할 숫자를 입력하세요 (종료: Ctrl+C)')
    try:
        while True:
            text = input('>>> 숫자 입력: ')
            try:
                number = int(text)
                if number <= 0:
                    print('1 이상의 숫자를 입력하세요')
                    continue
                node.send_goal(number)
                
                import time
                time.sleep(number + 1.5)
                
            except ValueError:
                print('숫자만 입력해주세요')
                
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()