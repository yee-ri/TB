# Gazebo 터널 단독 실행

저장소로 이동한다. 경로는 실제 위치에 맞춘다.

```bash
cd /home/sj/TB_tunnel_gazebo
```

Gazebo와 `codes/stepbystep/try_maze.py` 제어기를 launch로 함께 실행한다. 기본값은 `normal` 속도·A 배치·STEP 9다.

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

실차 카메라(노출 `20000µs`·게인 `16`)·LiDAR·OpenCR·장착 TF를 시작한다.

```bash
roslaunch launch/sensors.launch enable_depth:=true
```
