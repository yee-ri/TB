# Gazebo 터널 단독 실행

분리된 작업 폴더에서 기존 경기장·로봇을 읽기 전용으로 사용한다. 공식 시작점 전체 코스 시험이 아니다.

```bash
cd /home/sj/TB_tunnel_gazebo
docker compose -f compose.gazebo.yaml up
```

원본 FAST 속도 설정으로 터널 단독 시험한다. 기본값은 `normal`이다.

```bash
TB_TUNNEL_PROFILE=fast docker compose -f compose.gazebo.yaml up
```

B 배치에서 FAST 터널 단독 시험을 실행한다. 기본 배치는 `layout_a`이다.

```bash
TB_TUNNEL_PROFILE=fast TB_TUNNEL_LAYOUT=layout_b docker compose -f compose.gazebo.yaml up
```

C 배치에서 FAST 터널 단독 시험을 실행한다.

```bash
TB_TUNNEL_PROFILE=fast TB_TUNNEL_LAYOUT=layout_c docker compose -f compose.gazebo.yaml up
```

화면 없이 실행한다.

```bash
TB_GAZEBO_GUI=false docker compose -f compose.gazebo.yaml up -d
```

이 시뮬레이션만 종료한다.

```bash
docker compose -f compose.gazebo.yaml down
```

이식본 단위 검사와 ROS 실행 검사를 한다.

```bash
docker compose -f compose.gazebo.yaml run --rm --no-deps gazebo bash -lc 'python3 -m unittest discover -s tests -v && python3 tests/ros_smoke.py'
```
