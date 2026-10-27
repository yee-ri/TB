# 주행 실행

실행 파일은 `codes/stepbystep/try_maze.py`다. 로봇 전원을 켜고 카메라·LiDAR·OpenCR를 연결한 뒤, 로봇을 코스 출발 위치에 놓는다.
별도 버전 인자 없이 **V2 · 실물 연습장 설정**을 사용한다. 직접 실행하면 센서와 같은 ROS 환경을 자동으로 준비한다.
아래는 현재 NUC 경로이며, 다른 컴퓨터에서는 `cd` 경로를 실제 저장소 위치로 바꾼다.
센서 드라이버 작업공간은 `scripts/nuc_env.sh`에서 지정한다. 저장소의 과거 `catkin_ws/build·devel`은 사용하지 않는다.

## 1. 센서 실행

첫 번째 터미널에서 실행하고 창을 열어 둔다. 기존 센서가 실행 중이면 먼저 종료한다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
bash scripts/nuc_sensors.sh --mounts --depth
```

## 2. 주행 시작

센서가 준비되면 두 번째 터미널에서 실행한다.

```bash
cd /home/sj/ParkCars/TB-tunnel-nuc
python3 codes/stepbystep/try_maze.py
```

`try_maze.py`의 `~start_step` 기본값에 설정된 단계부터 주행한다. 현재 기본값은 **1**이다.
터널에 도달하면 V2가 실행되며, `TB_TUNNEL_VERSION=v2`나 별도의 `source` 명령은 필요 없다.
직접 실행은 기존 OpenCR 시작 검사로 토크 상태를 확인하고 활성화하므로, 이전 시험에서 토크를 껐어도 다시 시작할 수 있다.
시작 단계만 바꾸려면 다음처럼 지정한다.

```bash
python3 codes/stepbystep/try_maze.py _start_step:=9
```

STEP 9 시작 위치와 선택적 기록 방법은 [터널 테스트 안내](README_TUNNEL_TEST.md)를 참고한다.
이 직접 실행 방식은 rosbag을 자동으로 기록하지 않는다.

## 3. 종료

터널을 나온 뒤에도 차선 주행이 계속되므로 완주 후 직접 종료한다.

1. 주행 터미널에서 `Ctrl+C`를 누른다.
2. 로봇이 멈춘 것을 확인한다.
3. 센서 터미널에서 `Ctrl+C`를 누른다.
