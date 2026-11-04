# 실차 주행 실행

기본 실행 파일은 `codes/stepbystep/1104.py`다. 센서를 시작한 뒤 직접 실행한다.
일반 주행과 터널 단독 시험 모두 같은 파일을 사용하며, 터널은 V2로 주행한다.
색상 인식에는 `codes/stepbystep/current.lut`와 `images/left.png`, `images/right.png`를 사용하므로 저장소 경로를 유지한다.

아래 `cd` 경로는 실제 저장소 위치에 맞춘다.

## 센서 시작 — 터미널 A

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
roslaunch launch/sensors.launch enable_depth:=true
```

카메라·LiDAR·OpenCR·장착 TF를 시작한다. 기존 센서 실행이 있으면 먼저 종료한다.
카메라는 자동 노출을 끄고 노출 `20000µs`·게인 `16`으로 시작한다.

## 주행 기록 — 선택, 별도 터미널

ROS 문자 로그는 `~/.ros/log/`에 자동 저장된다. 영상·센서 데이터·속도 명령은 아래 기록을 실행해야 저장된다.
기록용 터미널에서는 `cd`가 필요 없다.

**최초 한 번 — 저장 폴더 만들기**

```bash
mkdir -p ~/tb_logs
```

폴더를 만든 뒤에는 이 명령을 다시 실행할 필요 없다.

**시험마다 — 기록 시작**

센서를 켠 뒤 **주행 시작 전에** 기록용 터미널에서 실행하고 창을 열어 둔다.

```bash
rosbag record --lz4 -o ~/tb_logs/run \
  /camera/color/image_raw /livox/lidar /odom /tf /tf_static \
  /cmd_vel /step /rosout /motor_power /sensor_state
```

`--lz4`는 영상·LiDAR 원본값을 유지하면서 압축한다. 시험이 끝나면 기록도 `Ctrl+C`로 종료한다.
다음 시험에는 같은 기록 터미널에서 `↑` 키로 이전 `rosbag record` 명령을 불러와 `--lz4`가 있는지 확인한 뒤 Enter를 누르면 된다.
기록을 다시 시작할 때마다 날짜·시각이 붙은 새 파일을 만들어 이전 기록을 덮어쓰지 않는다.

시험마다 `~/tb_logs/run_날짜_시각.bag`로 저장된다. `/cmd_vel`과 `/odom`을 비교하고
당시 카메라 영상·LiDAR·ROS 로그를 함께 확인할 수 있다. Python 소스와 YAML 파일은 자동 저장되지 않는다.

## STEP 8부터 터널 진입 시험 — 터미널 B

차단기 이후 터널로 이어지는 차선이 보이는 위치에 로봇을 놓고 실행한다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
rostopic pub -1 /motor_power std_msgs/Bool 'data: true'
python3 codes/stepbystep/1104.py _start_step:=8
```

STEP 8 차선 주행 → STEP 8.5 추가 전진 → STEP 9 위치 정합·진입 → STEP 10 경로 주행 → STEP 11 차선 주행으로 이어진다.

## 전체 주행 시작 — 터미널 B

로봇을 출발 위치에 놓고 센서가 준비된 뒤 실행한다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
rostopic pub -1 /motor_power std_msgs/Bool 'data: true'
python3 codes/stepbystep/1104.py
```

기본 시작은 STEP 0(신호등 대기)이다. 모터 토크는 위 명령으로 직접 활성화한다.
터널 단독 시험은 [터널 시작 위치](README_TUNNEL_TEST.md)에 배치하고 마지막 명령을 다음으로 바꾼다.

```bash
python3 codes/stepbystep/1104.py _start_step:=9
```

## 종료

터널 종료 뒤에도 STEP 11 차선 주행이 계속된다. 주행 터미널에서 `Ctrl+C`를 누른 뒤 토크를 해제한다.

```bash
rostopic pub -1 /motor_power std_msgs/Bool 'data: false'
```

로봇이 멈춘 것을 확인한다. 기록 중이면 기록 터미널에서 `Ctrl+C`를 누르고,
`~/tb_logs/`의 `.bag.active`가 `.bag`으로 저장될 때까지 기다린다.
그다음 센서 터미널에서 `Ctrl+C`를 누른다.
토크를 해제한 뒤 다시 주행하려면 시작 전에 `data: true` 명령을 다시 실행한다.

[터널 단독 직접 실행·설정·선택적 기록](README_TUNNEL_TEST.md) · [Gazebo 실행](DOCKER_NOETIC.md)
