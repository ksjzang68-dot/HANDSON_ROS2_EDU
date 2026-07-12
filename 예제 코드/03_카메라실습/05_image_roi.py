import cv2
import numpy as np

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ROI(사다리꼴) 꼭짓점 비율 (0~1), 해상도가 바뀌어도 그대로 동작
ROI_TOP_Y = 0.55            # ROI 윗변의 y 위치 (화면 세로 비율)
ROI_TOP_LEFT_X = 0.40
ROI_TOP_RIGHT_X = 0.60
ROI_BOTTOM_LEFT_X = 0.05
ROI_BOTTOM_RIGHT_X = 0.95


def make_roi_vertices(frame_shape):
    h, w = frame_shape[:2]
    top_y = int(h * ROI_TOP_Y)
    vertices = np.array([[
        (int(w * ROI_BOTTOM_LEFT_X), h),
        (int(w * ROI_TOP_LEFT_X), top_y),
        (int(w * ROI_TOP_RIGHT_X), top_y),
        (int(w * ROI_BOTTOM_RIGHT_X), h),
    ]], dtype=np.int32)
    return vertices


def apply_roi_mask(frame, vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, vertices, (255, 255, 255))
    return cv2.bitwise_and(frame, mask)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
        return

    print("ROI 적용 스트리밍 시작. 종료하려면 화면에서 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[경고] 프레임을 읽지 못했습니다.")
            break

        vertices = make_roi_vertices(frame.shape)

        # 원본 + ROI 테두리 표시용
        preview = frame.copy()
        cv2.polylines(preview, vertices, isClosed=True, color=(0, 255, 255), thickness=2)

        # 실제 ROI 마스킹 결과
        roi_masked = apply_roi_mask(frame, vertices)

        cv2.imshow("Original + ROI", preview)
        cv2.imshow("ROI Masked", roi_masked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()