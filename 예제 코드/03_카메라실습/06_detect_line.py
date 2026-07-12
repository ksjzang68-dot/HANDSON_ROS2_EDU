import cv2
import numpy as np

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ROI(사다리꼴) 꼭짓점 비율 (0~1)
ROI_TOP_Y = 0.55
ROI_TOP_LEFT_X = 0.40
ROI_TOP_RIGHT_X = 0.60
ROI_BOTTOM_LEFT_X = 0.05
ROI_BOTTOM_RIGHT_X = 0.95

# Canny 파라미터
CANNY_LOW = 50
CANNY_HIGH = 150

# Hough 변환 파라미터
HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 100


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


def apply_roi_mask(edges, vertices):
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return edges


def detect_lines(roi_edges):
    return cv2.HoughLinesP(
        roi_edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )


def draw_lines(frame, lines):
    result = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return result


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
        return

    print("라인 검출 스트리밍 시작. 종료하려면 화면에서 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[경고] 프레임을 읽지 못했습니다.")
            break

        vertices = make_roi_vertices(frame.shape)

        edges = detect_edges(frame)
        roi_edges = apply_roi_mask(edges, vertices)
        lines = detect_lines(roi_edges)

        result = draw_lines(frame, lines)
        cv2.polylines(result, vertices, isClosed=True, color=(255, 255, 0), thickness=1)

        cv2.imshow("ROI Edges", roi_edges)
        cv2.imshow("Detected Lines", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()