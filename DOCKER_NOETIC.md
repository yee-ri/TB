# Gazebo 터널 단독 실행

저장소로 이동한다. 경로는 실제 위치에 맞춘다.

```bash
cd /home/sj/TB_tunnel_gazebo
```

Gazebo와 launch에 연결된 제어기를 실행한다. 기본값은 `normal` 속도·A 배치·STEP 9다.

```bash
docker compose -f compose.gazebo.yaml up
```

`fast` 속도·A 배치로 실행한다.

```bash
TB_TUNNEL_PROFILE=fast TB_TUNNEL_LAYOUT=layout_a docker compose -f compose.gazebo.yaml up
```

`fast` 속도·B 배치로 실행한다.

```bash
TB_TUNNEL_PROFILE=fast TB_TUNNEL_LAYOUT=layout_b docker compose -f compose.gazebo.yaml up
```

`fast` 속도·C 배치로 실행한다.

```bash
TB_TUNNEL_PROFILE=fast TB_TUNNEL_LAYOUT=layout_c docker compose -f compose.gazebo.yaml up
```

화면 없이 실행한다.

```bash
TB_GAZEBO_GUI=false docker compose -f compose.gazebo.yaml up -d
```

시뮬레이션을 종료한다.

```bash
docker compose -f compose.gazebo.yaml down
```

주행 코드 단위 검사·실행 파일 선택 검사·ROS 연결 검사를 실행한다.

```bash
docker compose -f compose.gazebo.yaml run --rm --no-deps gazebo bash -lc 'source /opt/ros/noetic/setup.bash && source /workspace/devel/setup.bash && python3 -m unittest discover -s tests && python3 tests/ros_smoke.py'
```

TB1에서 카메라·LiDAR·OpenCR·장착 TF를 시작한다. 카메라 수동 노출은 `16000µs`다.

```bash
cd /home/sj/Desktop/TB
bash start_robot.sh
```

실차에서 STEP 8 차선 주행부터 터널 진입을 시험한다.

```bash
cd /home/sj/Desktop/TB
rostopic pub -1 /motor_power std_msgs/Bool 'data: true'
python3 codes/stepbystep/1105_lut.py _start_step:=8
```

실차에서 위치 정합·터널 주행부터 시험한다.

```bash
python3 codes/stepbystep/1105_lut.py _start_step:=9
```

실차 영상·LiDAR·주행 상태를 무손실 압축으로 기록한다. 시험 종료 시 기록 터미널에서 `Ctrl+C`로 마친다.

```bash
mkdir -p ~/tb_logs
rosbag record --lz4 -o ~/tb_logs/run \
  /camera/color/image_raw /livox/lidar /odom /tf /tf_static \
  /cmd_vel /step /rosout /motor_power /sensor_state
```
