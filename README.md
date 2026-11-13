# 실차 주행 실행

TB1 저장소는 `/home/sj/Desktop/TB`다. 다른 컴퓨터에서는 `cd` 경로를 실제 저장소 위치에 맞춘다.
주행 파일은 `codes/stepbystep/test.py`이며, `current.lut`와 `images/left.png`, `images/right.png`는 기존 위치에 둔다.
터널 설정은 `codes/stepbystep/tunnel.real.yaml`을 사용한다. 외곽 가로 1800mm × 깊이 1820mm, 벽 두께 10mm, 입출구 폭 300mm 기준이다.

## 1. 센서 시작 — 터미널 A

```bash
cd /home/sj/Desktop/TB
bash start_robot.sh
```

카메라·LiDAR·OpenCR·장착 TF를 시작하고 카메라를 수동 노출 `16000µs`(16ms)로 설정한다.
기존 센서 실행이 있으면 먼저 종료하고, 이 터미널은 열어 둔다.

## 2. 전체 주행 — 터미널 B

**Windows에서 카메라·LUT 화면을 보려면**, X 서버를 켜고 터미널 B로 사용할 새 PowerShell 창에서 접속한다.

```powershell
$env:DISPLAY = "127.0.0.1:0.0"
ssh -Y -C TB3
```

경기 출발 위치에 로봇을 놓고 센서가 준비되면, 접속한 터미널에서 실행한다.

```bash
cd /home/sj/Desktop/TB
rostopic pub -1 /motor_power std_msgs/Bool 'data: true'
python3 codes/stepbystep/test.py
```

STEP 0에서 신호등을 기다린 뒤 전체 주행을 시작한다.
화면을 표시할 수 없는 SSH 터미널에서는 마지막 명령을 다음으로 바꾼다.

```bash
python3 codes/stepbystep/test.py _show_lut_debug:=false
```

STEP 8부터 구간 시험할 때만 주행 명령 뒤에 `_start_step:=8`을 붙인다.

STEP 9부터 터널만 시험할 때는 **바퀴 축 중앙을 입구선에서 안쪽 9.5cm, 입구 중앙에서 오른쪽 2.2cm**에 놓고 터널 안쪽을 똑바로 향하게 한다.
주행 명령 뒤에 `_start_step:=9`를 붙여 실행한다. 시작 후 코드가 왼쪽으로 5° 회전하므로 미리 꺾어 놓지 않는다.

## 3. 종료

미리보기 창을 닫아도 주행은 계속된다. 주행 터미널에서 `Ctrl+C`를 누른 뒤 모터 토크를 해제한다.

```bash
rostopic pub -1 /motor_power std_msgs/Bool 'data: false'
```

정지 후 센서 터미널에서도 `Ctrl+C`를 누른다. 다시 시험할 때는 토크를 `data: true`로 켠다.

## 터널 센서 기록 — 영상 제외, 별도 터미널

센서를 켠 뒤 주행 시작 전에 실행한다.

```bash
mkdir -p ~/tb_logs
rosbag record --lz4 -o ~/tb_logs/run \
  /livox/lidar /odom /tf /tf_static \
  /cmd_vel /step /rosout /motor_power /sensor_state
```

시험이 끝나면 센서 종료 전에 기록 터미널에서 `Ctrl+C`를 누르고 `.bag.active`가 `.bag`으로 저장될 때까지 기다린다.

[Gazebo 실행](DOCKER_NOETIC.md)
