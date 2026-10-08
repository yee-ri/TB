# TB — tunnel port

`yee-ri/TB`의 `b683f7c`에서 분리한 터널 이식본. 다른 미션은 TB 코드를 사용한다.

## 분리된 Gazebo 시험

`codex/tunnel-port`는 이식 기준본, `codex/tunnel-gazebo`는 별도 작업 폴더의 시뮬레이션 브랜치다.
`main`에는 병합하지 않는다. 실행 명령은 [DOCKER_NOETIC.md](DOCKER_NOETIC.md)를 참고한다.

- 기존 custom robot·경기장·빌드 결과를 `/workspace`에 읽기 전용으로 연결한다.
- 전용 컨테이너 내부에서만 ROS를 실행하며, TB 노드 하나만 `/cmd_vel`을 발행한다.
- 터널 입구에서 step 9로 시작한다. 공식 시작점 전체 주행이 아니다.
- Gazebo `LaserScan`을 TB 노드가 직접 받는다. 기본 실물 입력은 기존 `PointCloud2`다.
- Gazebo는 원본 통합 설정과 같은 world odometry + EKF를 사용한다. 실물 위치추정 검증이 아니다.
- 전체 코스는 입구 기준 정합과 다른 TB 미션의 센서 연결도 필요하다. 현재 Gazebo RGB 카메라는 TB 차단봉용 깊이 영상을 제공하지 않는다.

2026-10-08 연결 시험: Gazebo GUI 프로세스·RGB·LiDAR·EKF odom·TB 단독 명령 발행을 확인했다.
터널 입구 단독 출발에서는 `ENTRY: waiting for an observed entry corridor with turning room`에 머물렀다.
30초 관측 중 전진 명령은 없었으며, 터널 통과·차선 복귀는 미검증이다. 안전 조건이나 경로 코어는 변경하지 않았다.

## 실행

ROS Noetic, 정상 빌드한 센서 드라이버 작업공간, `numpy`, `PyYAML`, OpenCV가 필요하다.
저장소에 들어 있는 과거 `catkin_ws/devel`의 절대경로 설정은 그대로 사용하지 않는다.

```bash
# 터미널 1: 카메라·OpenCR·Mid-360 시작 (센서 작업공간을 먼저 source)
./start_robot.sh
```

```bash
# 터미널 2: 공식 미션 순서로 시작
cd codes/stepbystep
python3 try_maze.py _start_step:=0
```

```bash
# 터널 진입 기준 자세에서만 단독 시험. 전체 코스 검증이 아님.
python3 try_maze.py _start_step:=9
```

설정은 `codes/stepbystep/tunnel.yaml`. 다른 설정은 `_tunnel_config:=/절대경로/설정.yaml`로 선택한다.

## 이식 범위

- TB의 `/odom`, 영상 판단, 미션 순서와 단일 `/cmd_vel` 발행자를 유지한다.
- step 9: LiDAR로 확인한 진입로에서 정지·회전 공간을 남기고 전진한다.
- step 10: 측정 속도가 멈추면 Hybrid A*로 출구 경로를 계산하고 odom으로 추종한다.
- 출구 위치·방향과 새 카메라 차선 검출이 확인되면 step 11의 기존 차선 제어로 돌아간다.
- 지도 갱신만으로 경로를 버리지 않는다. 남은 경로의 실제 충돌은 재탐색, 미관측·제동 위험은 정지로 처리한다.
- LiDAR 취득 시각의 TF를 사용한다. 없는 방향의 포인트를 빈 공간으로 꾸미지 않는다.

`tunnel/planner.py`, `costmap.py`는 원본 프로젝트의 코드를 출처 주석 외 그대로 이식했다.
`tracking.py`는 원본의 추종·속도 제한 함수만 추출했다. `mission.py`가 이들을 TB 입력에 연결한다.
원본의 AMCL·미션 관리자·별도 터널 ROS 노드·다른 미션용 라이브러리는 필요하지 않다.
대체된 `maze_navigation.py`는 제거했으며 이전 내용은 Git의 원본 커밋에 남아 있다.

## 좌표와 실물 적용 전 확인

기본 목표는 사용자가 지정한 **원본 Gazebo 터널 출구**다.

| 기준 | 원본 map 좌표 | 이식본의 진입 기준 상대 좌표 |
| --- | --- | --- |
| 진입 자세 | `(-1.7475895, 0.140, -90°)` | `(0, 0, 0°)` |
| 터널 밖 출구 | `(0.200, -1.748373, 0°)` | `(1.888373, 1.9475895, 90°)` |

이식본은 시작 odom을 고정하므로 **step 9가 이 진입 자세에서 시작한다는 전제**가 있다.
TB의 카메라 차선 소실 조건은 이 자세와의 일치를 보증하지 않는다.
이식본에는 원본의 LiDAR 입구 정렬·정적 벽 지도·AMCL 정렬을 넣지 않았다.
관측 전의 공간으로도 탐색 후보는 만들지만 실제 진입과 제동 영역은 관측된 공간만 허용한다.

차체 수치는 원본 custom robot의 비대칭 직사각형 기준이다. TB 저장소만으로는 실제 차체의 앞·뒤·폭을 확인할 수 없다.
실물 실행 전 footprint, 센서 TF, 입구 기준 자세, 출구 위치를 맞춰야 한다.

## 검증

```bash
python3 -m unittest discover -s tests -v
```

검사 범위: 경로 탐색·직사각형 충돌·추종, 센서 시각·제어권 연결, 이상적 LiDAR/차동구동 폐루프 시험.
이상적 폐루프는 실제 Gazebo나 실물 주행이 아니다. 이식본의 공식 시작점 전체 주행·실물 주행은 아직 미검증이다.
Noetic Docker 검사 명령은 `DOCKER_NOETIC.md`를 참고한다.
