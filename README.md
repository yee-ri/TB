# 실차 주행 실행

기본 실행 파일은 `codes/stepbystep/try_maze.py`다. 센서를 시작한 뒤 직접 실행한다.
일반 주행과 터널 단독 시험 모두 같은 파일을 사용하며, 터널은 V2로 주행한다.

아래 `cd` 경로는 실제 저장소 위치에 맞춘다.

## 센서 시작 — 터미널 A

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
roslaunch launch/sensors.launch enable_depth:=true
```

카메라·LiDAR·OpenCR·장착 TF를 시작한다. 기존 센서 실행이 있으면 먼저 종료한다.

## 직접 주행 시작 — 터미널 B

로봇을 출발 위치에 놓고 센서가 준비된 뒤 실행한다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
rostopic pub -1 /motor_power std_msgs/Bool 'data: true'
python3 codes/stepbystep/try_maze.py
```

기본 시작은 STEP 1이다. 모터 토크는 위 명령으로 직접 활성화한다.
터널 단독 시험은 [터널 시작 위치](README_TUNNEL_TEST.md)에 배치하고 마지막 명령을 다음으로 바꾼다.

```bash
python3 codes/stepbystep/try_maze.py _start_step:=9
```

## 종료

터널 종료 뒤에도 STEP 11 차선 주행이 계속된다. 주행 터미널에서 `Ctrl+C`를 누른 뒤 토크를 해제한다.

```bash
rostopic pub -1 /motor_power std_msgs/Bool 'data: false'
```

로봇이 멈춘 것을 확인한다. 기록 중이면 기록 터미널에서 `Ctrl+C`로 저장한 뒤 센서 터미널에서 `Ctrl+C`를 누른다.
토크를 해제한 뒤 다시 주행하려면 시작 전에 `data: true` 명령을 다시 실행한다.

[터널 단독 직접 실행·설정·선택적 기록](README_TUNNEL_TEST.md) · [Gazebo 실행](DOCKER_NOETIC.md)
