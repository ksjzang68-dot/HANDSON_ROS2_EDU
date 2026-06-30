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