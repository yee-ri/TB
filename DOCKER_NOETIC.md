# Noetic 단위 검사

기존 `custom-autorace:noetic` 이미지로 이식본의 테스트를 실행한다.

```bash
docker run --rm --network none \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/sj/TB_tunnel_port:/target -w /target \
  custom-autorace:noetic bash -lc \
  'source /opt/ros/noetic/setup.bash && python3 -m unittest discover -s tests -v'
```

외부 로봇과 격리된 ROS에서 실제 노드 초기화·명령 발행자·센서 없는 정지를 검사한다.

```bash
docker run --rm --network none \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/sj/TB_tunnel_port:/target -w /target \
  custom-autorace:noetic bash -lc \
  'source /opt/ros/noetic/setup.bash && python3 tests/ros_smoke.py'
```

센서 확인 후 TB 공식 미션 순서로 실행한다.

```bash
cd /home/sj/TB_tunnel_port/codes/stepbystep
python3 try_maze.py _start_step:=0
```

설정된 터널 진입 자세에서 단독 실행한다.

```bash
python3 try_maze.py _start_step:=9
```
