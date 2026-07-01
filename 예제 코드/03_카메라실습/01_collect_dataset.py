import cv2
import os

# 저장할 디렉토리 설정 (~/Desktop/dataset)
SAVE_DIR = os.path.expanduser("~/Desktop/dataset")
os.makedirs(SAVE_DIR, exist_ok=True)

# 카메라 초기화 (0번 카메라, V4L2 백엔드 사용)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라를 읽을 수 없습니다.")
        break

    # 화면에 카메라 영상 출력
    cv2.imshow("AI Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # 스페이스바(32) 또는 'c' 키를 누를 때 이미지 저장
    if key == 32 or key == ord('c'):
        count += 1
        # 파일명 형식: stop_001.jpg, stop_002.jpg ...
        filename = os.path.join(SAVE_DIR, f"stop_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"{count}장 촬영, {filename} 저장 완료")
        cv2.waitKey(50)  # 0.05초 대기

    # 'q' 키를 누르면 종료
    elif key == ord('q'):
        break

# 자원 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()