import cv2

CAMERA_INDEX = 0        # /dev/video0 이 아니면 1, 2 등으로 변경
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다. /dev/video* 번호를 확인하세요.")
        return

    print("스트리밍 시작. 종료하려면 화면에서 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[경고] 프레임을 읽지 못했습니다.")
            break

        cv2.imshow("Camera Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()