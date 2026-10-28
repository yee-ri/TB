# 터널 단독 시험

기본 실행 파일은 `codes/stepbystep/try_maze.py`이며, `_start_step:=9`로 터널부터 직접 시작한다.
아래 `cd` 경로는 실제 저장소 위치에 맞춘다.

## 시작 위치

연습 터널은 외경 1,820mm, 벽 2mm, 출입구 폭 300mm 설정을 사용한다.
전체 주행의 STEP 8.5 추가 전진이 끝난 위치·방향에 놓고 STEP 9로 시작한다.
이미 입구 안쪽에 들어온 위치에서도 시작할 수 있다.

위치 정합 후보 범위는 바퀴축 중앙 기준 입구 바깥 약 49cm부터 안쪽 약 31cm까지,
입구 중앙선 좌우 7cm, 방향 ±20°다. 벽 관측으로 자세가 확정될 때까지 정지한다.
입구 밖 140mm는 지도 기준점이며 고정된 필수 출발점은 아니다.

STEP 9 위치 정합·진입 → STEP 10 경로 주행 → 차체 전체 출구 통과 후 STEP 11 차선 주행으로 이어진다.
차선 주행은 자동으로 종료되지 않는다. 일반 주행은 [README.md](README.md)를 따른다.

## 직접 실행 — 기본

터미널 A: 카메라·LiDAR·OpenCR·장착 TF를 시작한다. 기존 센서 실행이 있으면 먼저 종료한다.
터널 단독 시험에는 depth가 필요 없다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
roslaunch launch/sensors.launch
```

터미널 B: 시작 위치에 로봇을 놓고 센서가 준비된 뒤 STEP 9부터 주행한다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
rostopic pub -1 /motor_power std_msgs/Bool 'data: true'
python3 codes/stepbystep/try_maze.py _start_step:=9
```

## 직접 실행 종료

주행 터미널에서 `Ctrl+C`를 누른 뒤 토크를 해제한다.

```bash
rostopic pub -1 /motor_power std_msgs/Bool 'data: false'
```

로봇이 멈춘 것을 확인한다. 기록 중이면 기록 터미널에서 `Ctrl+C`로 저장한 뒤 센서 터미널에서 `Ctrl+C`를 누른다.
다시 주행하려면 센서 준비 후 `data: true` 명령을 다시 실행한다.

## 로그 확인·선택적 센서 기록

ROS 로그의 기본 저장 위치는 `~/.ros/log/`다. 직접 실행한 제어기의 로그도 이 경로에서 확인한다.

```bash
ls -lt ~/.ros/log
```

센서 데이터를 남기려면 센서를 시작한 뒤, 주행 시작 전에 별도 터미널에서 기록한다.
주행 시작·종료는 위 직접 실행 절차와 같다.

```bash
mkdir -p "$HOME/tb_logs"
rosbag record -o "$HOME/tb_logs/tunnel" \
  /camera/color/image_raw /livox/lidar /odom /tf /tf_static \
  /cmd_vel /step /rosout /motor_power /sensor_state
```

`~/tb_logs/tunnel_날짜_시각.bag` 형식으로 저장된다.
제어기를 종료하고 토크를 해제한 뒤 로봇이 멈춘 것을 확인한다.
기록 터미널에서 `Ctrl+C`를 누르고 `.bag.active`가 `.bag`으로 저장된 뒤 센서를 종료한다.

## 설정 변경

| 파일 | 설정 |
| --- | --- |
| [tunnel.real.yaml](codes/stepbystep/tunnel.real.yaml) | 실차 지도·차체·센서·진입·속도·출구 |
| [navigation.yaml](codes/tunnel/navigation.yaml) | 전역 갱신 간격, 지역 목표 거리·후보 경로 |
| [tunnel.yaml](codes/stepbystep/tunnel.yaml) | Gazebo `normal`·`fast` 설정 |
| [sensors.launch](launch/sensors.launch) | 실차 센서·장착 TF |

실물 YAML의 `navigation:` 항목은 `navigation.yaml`의 같은 설정을 덮어쓴다.
주행을 종료하고 파일을 수정한 뒤 다시 실행한다. 기록 중이었다면 기록도 종료하고 새로 시작한다.
Python/YAML은 별도 빌드가 필요 없다.
`tunnel.competition.yaml`은 지도·목표가 미입력 상태이므로 실행에 사용하지 않는다.

현재 실차의 `sensors.lidar_maximum_age: null`은 LiDAR 지연 정지 검사를 해제한다.
LiDAR가 끊겨도 마지막 지도·전방 거리로 계산할 수 있다. `0.30`으로 바꾸거나 이 항목을 삭제하면
0.30초 제한으로 복원된다. odom·TF·미래 timestamp·초기 위치 정합 결과의 유효시간 검사는 유지된다.

실물 설정의 경로 속도 `0.40m/s`와 진입 속도 `0.30m/s`는 미션 내부 목표값이다.
선속도 출력은 `odometry.linear_scale`, 각속도 출력은 `odometry.angular_command_scale`로 나눈다.

## 원격 RViz

SSH 연결에서 X11 forwarding을 켜고, 센서가 실행 중인 상태에서 실행한다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
rviz -d rviz/tunnel_sensors.rviz
```

`DISPLAY`가 비어 있으면 X11 forwarding을 켜서 새로 연결한다.
