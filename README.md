# 주행 실행

실행 파일은 `codes/stepbystep/try_maze.py`다. 로봇 전원을 켜고 카메라·LiDAR·OpenCR를 연결한 뒤, 로봇을 코스 출발 위치에 놓는다.

## 1. 센서 실행

첫 번째 터미널에서 실행하고 창을 열어 둔다. 기존 센서가 실행 중이면 먼저 종료한다.

```bash
cd /home/sj/Desktop/TB
bash scripts/nuc_sensors.sh --mounts --depth
```

## 2. 주행 시작

센서가 준비되면 두 번째 터미널에서 실행한다.

```bash
cd /home/sj/Desktop/TB
python3 codes/stepbystep/try_maze.py
```

`try_maze.py`에 설정된 시작 단계부터 주행한다. 현재 시작 단계는 3이다.

## 3. 종료

터널을 나온 뒤에도 차선 주행이 계속되므로 완주 후 직접 종료한다.

1. 주행 터미널에서 `Ctrl+C`를 누른다.
2. 로봇이 멈춘 것을 확인한다.
3. 센서 터미널에서 `Ctrl+C`를 누른다.
