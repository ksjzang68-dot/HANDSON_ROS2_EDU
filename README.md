본 교육 자료는 실습 중심으로 구성되어 있으며, 각 예제는 실제 Jetson Orin Nano 환경에서 바로 실행할 수 있도록 제작되었습니다. ROS2와 Physical AI를 처음 접하는 학습자도 단계적으로 따라갈 수 있도록 기초부터 프로젝트까지 순차적으로 학습할 수 있습니다.


1# Jetson Nano Wi-Fi 연결 오류 해결

오류 메시지:
> Failed to add/activate connection — (1) not authorized to control networking

## 1단계: netdev 그룹 추가

```bash
sudo usermod -aG netdev $USER
```

→ 로그아웃 후 재로그인

## 2단계: 그래도 안 되면 PolicyKit 파일 생성

```bash
sudo nano /etc/polkit-1/localauthority/50-local.d/networkmanager.pkla
```

아래 내용 붙여넣기:

```ini
[Network Manager all users]
Identity=unix-group:netdev
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes
```

```bash
sudo systemctl restart polkit
sudo systemctl restart NetworkManager
```

→ 재로그인 후 GUI에서 다시 시도


## 예시 동작 영상
'''
https://youtu.be/s6dDlIa_Oxo
'''


# YOLOv8 커스텀 데이터셋 학습 (Google Colab)

라벨링이 완료된 데이터셋(zip)을 Colab에 직접 업로드하여 YOLOv8 모델을 학습시키고,
학습된 `best.pt`를 다운로드하는 과정을 정리한 문서입니다.

## 사전 준비물

- 라벨링이 끝난 데이터셋 zip 파일 (`images/`, `labels/`, `data.yaml` 포함)
- Colab 런타임 유형: **GPU**로 설정 (런타임 → 런타임 유형 변경 → 하드웨어 가속기 → GPU)

---

## 1. 데이터셋 업로드 및 압축 해제

```python
from google.colab import files
uploaded = files.upload()  # zip 파일 업로드
!unzip dataset.zip -d /content/dataset
```

- 실행하면 파일 선택 창이 뜨고, 준비한 데이터셋 zip 파일을 선택해 업로드합니다.
- 업로드한 파일명이 정확히 `dataset.zip`이어야 `unzip dataset.zip` 명령이 정상 동작합니다. 다른 이름으로 올릴 경우 해당 줄의 파일명을 맞춰서 수정해야 합니다.
- 압축 해제 후 `/content/dataset` 안에 `data.yaml`이 실제로 있는지, 그 안의 `train`/`val` 경로가 올바른지 확인이 필요합니다.

## 2. 필요한 패키지 설치

```python
!pip install ultralytics
```

- YOLOv8 학습/추론에 필요한 `ultralytics` 라이브러리를 설치합니다.

## 3. 모델 학습

```python
from ultralytics import YOLO

# 사전학습된 기본 모델(yolov8n.pt, yolov8s.pt 등)을 base로 사용
model = YOLO('yolov8n.pt')

results = model.train(
    data='/content/dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    project='/content/runs',
    name='my_train'
)
```

| 파라미터 | 설명 |
|---|---|
| `data` | 데이터셋 정보가 담긴 `data.yaml` 경로 |
| `epochs` | 전체 데이터셋을 반복 학습할 횟수 (기본 100) |
| `imgsz` | 입력 이미지 크기 (기본 640) |
| `batch` | 한 번에 처리할 이미지 수 (GPU 메모리 부족 시 줄이기) |
| `project` | 결과가 저장될 상위 폴더 |
| `name` | 이번 학습 결과가 저장될 하위 폴더 이름 |

- `yolov8n.pt`는 로컬에 없으면 자동으로 다운로드되므로 별도 준비가 필요 없습니다.
- 학습이 끝나면 결과는 `/content/runs/my_train/weights/` 아래에 저장됩니다.
  - `best.pt` : 검증 성능이 가장 좋았던 시점의 모델
  - `last.pt` : 마지막 epoch의 모델

## 4. 학습된 모델(best.pt) 다운로드

```python
from google.colab import files
files.download('/content/runs/my_train/weights/best.pt')
```

- 로컬 컴퓨터로 `best.pt` 파일을 바로 다운로드합니다.

## 5. 구글 드라이브에 백업 (선택)

```python
# 구글드라이브에 저장하면 코랩 세션이 종료되어도 안전하게 저장 가능하다.

!cp /content/runs/my_train/weights/best.pt /content/drive/MyDrive/best.pt
```

- Colab 세션이 끊기면 `/content` 아래의 모든 파일이 사라지므로, 구글 드라이브가 마운트되어 있다면 이 단계를 통해 결과를 안전하게 보관할 수 있습니다.
- 드라이브 마운트가 되어 있지 않다면 아래 코드를 먼저 실행해야 합니다.
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

---

## 전체 흐름 요약

```
1. 데이터셋 zip 업로드 → 압축 해제
2. ultralytics 설치
3. YOLO('yolov8n.pt')로 모델 로드 후 model.train() 실행
4. /content/runs/my_train/weights/best.pt 생성 확인
5. best.pt 로컬 다운로드
6. (선택) 구글 드라이브에 복사하여 백업
```

## 주의사항

- 업로드 파일명이 `dataset.zip`이 아니면 `unzip` 명령에서 오류가 발생합니다.
- `data.yaml` 내부의 `train`/`val` 경로가 실제 압축 해제된 폴더 구조와 일치해야 합니다.
- Colab 세션이 종료되면 업로드했던 파일과 학습 결과가 모두 삭제되므로, 학습 완료 후에는 반드시 `best.pt`를 다운로드하거나 드라이브에 백업해야 합니다.