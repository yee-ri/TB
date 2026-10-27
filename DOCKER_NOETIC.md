# Gazebo 터널 단독 실행

분리된 작업 폴더에서 기존 경기장·로봇을 읽기 전용으로 사용한다. 공식 시작점 전체 코스 시험이 아니다.

```bash
cd /home/sj/TB_tunnel_gazebo
docker compose -f compose.gazebo.yaml up
```

V1 터널 단독 시험을 선택한다.

```bash
TB_TUNNEL_VERSION=v1 docker compose -f compose.gazebo.yaml up
```

V2 터널 단독 시험을 선택한다.

```bash
TB_TUNNEL_VERSION=v2 docker compose -f compose.gazebo.yaml up
```

검증한 A 배치에서 V2 FAST 터널 단독 시험을 실행한다.

```bash
TB_TUNNEL_VERSION=v2 TB_TUNNEL_PROFILE=fast TB_TUNNEL_LAYOUT=layout_a docker compose -f compose.gazebo.yaml up
```

같은 FAST 속도와 B 배치에서 V2를 시험한다.

```bash
TB_TUNNEL_VERSION=v2 TB_TUNNEL_PROFILE=fast TB_TUNNEL_LAYOUT=layout_b docker compose -f compose.gazebo.yaml up
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

# 실차 V2 실행

센서를 켠 뒤 `try_maze.py`만 직접 실행하는 명령은 [기본 주행 실행](README.md)에 정리했다.
직접 실행은 ROS 환경을 자동으로 준비하고 기본 V2·실물 설정·STEP 1을 사용한다.
`_start_step:=9`를 붙이면 터널 단독으로 시작한다.
이 NUC의 저장소는 `/home/sj/ParkCars/TB-tunnel-nuc`이며 다른 컴퓨터에서는 해당 저장소 경로를 쓴다.

STEP 9 시작 위치와 `nuc_trial.py`의 기록·정지·저장 순서는 [터널 테스트 안내](README_TUNNEL_TEST.md)를 따른다.
위 Gazebo 명령의 시뮬레이션 설정과 실행 방식은 그대로 유지한다.
