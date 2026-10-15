# NUC 실물 터널 준비 기록 — 2026-10-09 KST

이 문서의 `/home/sj/ParkCars` 경로와 네트워크·메시지 호환 설정은 당시 NUC의 준비 이력이다. 현재 `/home/sj/Desktop/TB`에서의 환경 준비와 실행은 [README_TUNNEL_TEST.md](README_TUNNEL_TEST.md)를 따른다.

2026-10-10 추가 확인: 사용자가 설명한 스위치는 PC 전원을 즉시 끈다. 모터 전원 차단 기능은 확인되지 않았으며, 이 스위치를 모터 정지 수단으로 기록한 이전 해석은 정정한다. 정상 시험 종료는 PC를 유지한 채 기존 드라이버의 소프트웨어 정지 요청과 rosbag 정상 종료를 사용한다. 실행 순서는 [README_TUNNEL_TEST.md](README_TUNNEL_TEST.md)에만 있다.

사용자가 직접 시험할 때의 순서별 안내: [터널 테스트 시작 안내 — README_TUNNEL_TEST.md](README_TUNNEL_TEST.md) (2026-10-10 갱신).

**환경·실제 센서·장착 TF·연습용 지도와 실행 도구를 준비했다.** 2026-10-10 사용자가 스위치를 포함한 물리적 준비 완료를 확인했다. 수동 확인 플래그를 요구하던 `physical_setup` 항목을 제거했으며 실행 시 실제 센서·TF·명령 발행자·기록 상태를 검사한다. 대회 규격은 새 안내에 따라 1.8m로 정정했으며, 벽 두께·치수 기준이 없어 대회 지도는 미완성이다. 주행은 사용자가 직접 시작한다. 이 준비 작업에서는 실제 `/cmd_vel`에 어떤 명령도 발행하지 않았으며 펌웨어·게인·터널 알고리즘을 변경하지 않았다.

## 환경과 보존 범위

- NUC `g2-nuc`, Ubuntu 20.04.6, Linux 5.15, ROS Noetic. 기존 센서 작업공간 `/home/sj/ParkCars`의 소스/build/devel과 수정된 Livox IP 설정을 보존했다.
- 별도 clone: `/home/sj/ParkCars/TB-tunnel-nuc`, 브랜치 `codex/tunnel-gazebo`, HEAD `6a035104ec391753845faa0466b03a8ae5918cae`; 참고 `230d0ae`는 조상이다. main 수정·commit·push는 하지 않았다. token 파일은 checkout·열람·인증에 사용하지 않았다.
- 기존 shell은 master `192.168.0.10`, hostname `192.168.0.10`을 가리키지만 이 NUC는 Wi-Fi `192.168.0.3`, 유선 `192.168.1.5`다. 해당 원격 master 연결은 응답하지 않았고 조사 시 로컬 ROS 노드는 없었다. 시험 안내의 환경 파일만 로컬 `127.0.0.1:11311`을 사용한다. 기존 shell 설정은 수정하지 않았다.
- 저장소의 과거 `catkin_ws/build/devel`과 token은 sparse checkout에서 제외했다. `.gitmodules` 없는 gitlink는 clone만으로 드라이버가 준비된 것이 아니다. 카메라·LiDAR는 실제 NUC에 있던 정상 빌드본을 재사용한다.
- 새 `nuc_ws`는 필요한 메시지/라이브러리만 이 NUC에서 빌드했다. `turtlebot3_msgs`는 기존 `/home/sj/Desktop/ARAICodes/catkin_ws/src/turtlebot3_msgs`를 사용한다. `tf2_sensor_msgs`는 공식 `ros/geometry2`의 noetic-devel 커밋 `8fef92a6b56a26e2b5be58d4d5c84dc4de5ee696` 소스를 사용한다. apt 캐시의 해당 패키지 URL은 404이고 sudo 비밀번호 없이 설치할 수 없었으므로 시스템 설치를 변경하지 않았다.
- HEAD에서 `start_step` 파라미터가 무시되고 step 3이 고정되어 있던 한 줄만 파라미터 읽기로 복원했다. 상위 브랜치의 다른 미션 동작은 유지했다.

## 정지 상태 센서 검사

카메라 D405는 USB `8086:0b5b`, 펌웨어 `05.15.01.55`, RealSense ROS 2.3.2/SDK 2.53이다. `/camera/color/image_raw`는 실제 640×480 RGB8, 29.996Hz, 수신 지연 중앙값 14.3ms를 확인했다. 12초 동안 360장의 서로 다른 영상을 수신했고 실제 색 영상 파일은 `records/camera_check.png`다. 터널/step 11에 필요한 RGB만 실행하며 depth는 꺼 둔다. 전체 코스 차단봉 검증은 이 준비 범위가 아니다.

Mid-360은 `192.168.1.172`에 응답했고 NUC 유선 주소 `192.168.1.5`는 기존 설정과 일치했다. 기존 driver2 1.2.7 및 SDK2를 사용한다. `/livox/lidar`는 **sensor_msgs/PointCloud2**, frame `livox_frame`, 약 10.000Hz, 약 20,064점/메시지, timestamp 증가, 수신 지연 중앙값 100.7ms를 확인했다. 기본 CustomMsg launch 대신 `xfer_format=0`을 명시했다. `/livox/imu`도 실제 발행된다. 기존 extrinsic 0과 IP 설정은 변경하지 않았다.

OpenCR은 실제 USB serial-by-id에서 인식되고 `/turtlebot3_core`의 rosserial/115200으로 연결됐다. `core v1.2.6 / TB3 SuperBurger` 로그를 확인했다. 초기 검사에서 구형 `sensor_msgs/BatteryState` MD5 `476f837fa6771f6e16e3bf4ef96f8770`과 설치된 Noetic MD5 `4ddae7f048e32fda22cac764685e3974`의 불일치로 반복 협상이 발생했다. 이는 firmware VersionInfo/SensorState 메시지 문제가 아니다. 초기 `/odom`은 토픽만 등록되고 실제 메시지는 0개였다. 재검사 및 호스트 메시지 정의 조치 결과는 `records/opencr_audit.json`과 이 문서 아래의 최종 검사 결과를 따른다. timeout을 늘리지 않았다.

`use_sim_time=false`를 명시하고 확인했다. 실제 `/cmd_vel` 발행자는 없었으며 구동계 subscriber는 `/turtlebot3_core`였다. 초기 검사는 장착 TF 없이 진행했고, 사용자에게 현재 하드웨어가 시트와 같다는 확인을 받은 뒤 바닥 기준으로 변환한 장착 TF를 적용했다. 실제 camera/LiDAR→base_footprint 및 측정 시각 odom 연결 조회 50건이 모두 통과했다. 카메라 내부 TF는 RealSense manager, 장착 TF는 `/nuc_camera_mount`와 `/nuc_lidar_mount`, odom TF는 `/turtlebot3_core`가 발행하며 같은 child의 중복 authority는 없다. 현재 결과는 `records/mount_diagnostics_20261009.json`이다.

## 설정·구동계 확인 결과

확정된 차체·장착값과 지도 좌표의 근거는 [NUC_MEASUREMENTS.md](NUC_MEASUREMENTS.md)에 보관했다. 연습용 지도는 외측 1820mm·벽 2mm·출입구 300mm 기준으로 생성해 정적 검사를 통과했다. 대회 안내의 1.8m 규격은 벽 두께·내측/외측 기준 등이 미확인이라 대회 지도는 미완성이다. 2026-10-09 센서 장착 TF 검사는 통과했다. 2026-10-10에는 검증된 장착 변환을 `launch/nuc_sensors.launch`에 직접 반영하고 YAML의 수동 준비 확인 항목을 제거했다. 터널 수치·게인·지도는 그대로다.

후보 소스 `/home/sj/Desktop/ARAICodes/Arduino/package/tb3bigwheel_core/tb3bigwheel_core_config.h`는 `CONTROL_MOTOR_TIMEOUT=500ms`이며 `.ino`는 시간이 지나면 0 속도를 제어한다. v1.2.6 문구만으로 업로드된 binary와 이 소스가 동일하다고 확정할 수 없다. RC100/버튼 입력도 watchdog 시각을 갱신하고 구동 명령에 합산된다. 후보 소스의 `/motor_power=false`는 torque off를 유지하지만, 실제 업로드본의 효과·제동과 활성 timeout은 정지 상태 준비에서 입증하지 못했다. torque off는 기계식 브레이크가 아니다.

정지 상태에서는 odom 이동 방향·거리 스케일·실제 제동거리/성능을 검증하지 못했다. 후보 펌웨어의 바퀴 반지름/기어 값도 서로 달라 소스만으로 실제 odom 스케일을 확정하지 않았다. 실제 구동 시험이나 펌웨어 변경은 수행하지 않았다.

시험 준비값 변경·시작 배치·설정 전환·실행·정지·기록 확인 절차는 [README_TUNNEL_TEST.md](README_TUNNEL_TEST.md)에만 기록한다.

## 검증 범위

2026-10-10 PC 전원 스위치 확인 후 `stop`에 새 `/sensor_state`의 `torque=false` 응답 확인을 추가했다. 출처·요청 이후 timestamp·데이터 신선도를 확인하며, 응답은 드라이버 보고 상태이고 실제 바퀴 제동/전원 차단 증명은 아니다. 관련 mock 테스트는 14개 통과했고 로그는 `records/stop_feedback_tests.log`다. 이 검증에서 실제 ROS 연결이나 하드웨어 메시지 발행은 하지 않았다.

2026-10-10 실행 절차 간소화 후 지도·normal 설정·기록·실시간 센서·발행자 충돌 검사 단위 테스트 8개가 통과했다. 연습 YAML의 정적 검사와 ROS launch 구문/노드 목록 검사도 통과했으며, launch에 직접 넣은 센서 변환은 이전 실제 TF 검사 값과 정확히 일치한다. 이 변경을 확인하면서 실제 제어기나 센서 노드를 실행하지 않았다.

구동계와 분리한 master `127.0.0.1:11329`에서 모든 제어기 입출력/TF를 `/tb_isolated_test/*`로 remap한 연결 검사는 통과했다. 실제 normal/step 9 초기화, 하나의 격리된 출력 발행자, 센서 대기 중 10회 이상 0 명령, 실제 구동 subscriber 없음, 정상 종료를 확인했다. 실제 11311 `/cmd_vel`에는 발행하지 않았다. 로그는 `records/isolated_smoke.log`다.

2026-10-09 당시 새 실물 실행 검사 테스트 6개가 모두 통과했고, 전체 단위 테스트는 116개 중 115개 통과했다. `test_entry_transition_frame_does_not_emit_lane_command` 한 개는 상위 HEAD의 step 8.5 추가와 이전 step 9 기대의 차이로 실패한다. 시험 범위인 명시적 step 9 시작은 격리 연결 검사로 확인했다. 새 테스트는 연습 지도 정적 통과, 미완성 대회 거부, ready/정지/별도 입력 확인 우회 거부, 원본 Gazebo 지도의 실물 사용 거부를 확인한다. `records/unit_tests_after_layout.log`에 현재 출력, `records/unit_tests.log`에 초기 110개 출력이 있으며 실물 성공/전체 코스 검증을 뜻하지 않는다.

## 재검사와 기록 검증

호스트 프로토콜 조치: `deps/common_msgs`의 공식 Noetic 커밋 `092bc230c8153dd4cfee0c84c9796cc7e6b09fdb`에서 **BatteryState.msg 하나만** 과거 공식 커밋 `ea6c65e8846d9873b8d6e44435c52fb43d6093a3`의 정의로 복원해 별도 `nuc_ws`에 빌드했다. MD5가 현재 펌웨어와 일치하며 다른 27개 센서 메시지/서비스 및 TurtleBot 메시지 MD5는 기존 설치와 일치한다. `/opt/ros`, 기존 드라이버, 펌웨어는 변경하지 않았고 새 호환 노드나 cmd_vel 중계 노드는 없다. 출처와 전체 확인은 `deps/common_msgs/NUC_PROTOCOL_PROVENANCE.json`, 빌드 출력은 `records/native_message_build.log`다. 이 환경 파일을 source해야 구형 BatteryState 연결이 유지된다.

| 장착 TF 적용 후 정지 상태 검사 | 실제 결과 |
| --- | --- |
| D405 color | 640×480 RGB8, 29.996Hz, 12초 360장, 지연 중앙값 14.2ms |
| Mid-360 PointCloud2 | 10.000Hz, 12초 120개, 약 19,872~20,160점, `livox_frame`, 지연 중앙값 100.7ms |
| OpenCR odom | 28.988Hz, 12초 348개, `odom→base_footprint`, 유효 위치/quaternion/twist/stamp; 지연 중앙값 2.11ms |
| 정지 odom 최신 값 | 위치 약 `(0.7e-6,-5.6e-6)m`, yaw 약 0.1699rad, linear x 약 -0.00209m/s, angular z 약 0.00280rad/s. 작은 정지 잡음/IMU 드리프트가 있으며 방향·거리 스케일 검증이 아님 |
| 센서 측정 시각의 odom TF | camera/LiDAR/odom 표본 총 30회 모두 조회 통과 |
| 측정 시각의 센서 장착 TF | camera optical/livox→base_footprint 총 20회 모두 조회 통과. odom 포함 전체 50회 통과 |
| 중복 TF/구동 명령 | 관측한 child 중 중복 authority 없음. 실제 `/cmd_vel` 발행자 없음 |
| 실제 OpenCR telemetry | `/firmware_version` 1.2.6, `/sensor_state` torque=true/button=0, `/cmd_vel_rc100` 175개 모두 0, 구형 `/battery_state` 실제 수신. RC100 꺼짐이나 watchdog/torque 해제 검증을 뜻하지 않음 |
| 프로토콜 재검사 오류 | 새 serial 실행에서 checksum/lost-sync 오류 0. 초기 오류 로그는 보존 |

현재 센서 진단은 `records/mount_diagnostics_20261009.json`이며 상태는 **empirical_sensor_tf_pass**, failures는 빈 목록이다. 이 진단은 TF 연결을 검사하며 물리 장착값의 별도 실측을 대신하지 않는다. 장착값 근거는 시트와 사용자 확인이다. 이전 장착 TF 적용 전 실패 기록 `records/sensor_diagnostics_final.json`과 이전 요약 `records/preparation_result.json`은 이력으로 보존했다. 현재 설정 요약은 `records/layout_setup_result.json`, OpenCR 자료는 `records/opencr_audit.json`이다. 센서 검사 통과와 실제 주행 성공은 별개다. 이후 2026-10-10 물리적 준비 완료는 사용자 확인에 근거하며, 실제 주행은 사용자가 시작한다.

센서만 **56.3초** 기록하여 33,703개 메시지와 color/LiDAR/odom/tf/tf_static/ROS 로그를 확인했고 rosbag SIGINT 정상 종료, exit 0, `.bag.active` 없음, `bag_finalized=true`를 확인했다. 실제 `/cmd_vel` 메시지는 0개이며 제어기를 시작하지 않았다. 검증 기록은 `records/20261009_082851_494130_KST/sensors.bag`, 상세는 같은 폴더의 `verified_bag_info.txt`다. 계획 경로/costmap 전체는 기록되지 않는다.

기록은 raw color를 LZ4로 보관한다. 실제 시험 기록 크기는 1.2GB/56.3초(약 1.3GB/분)였다. raw 영상을 보관하므로 색 영상의 compressedDepth 광고 토픽에는 구독하지 않는다. 센서 로그·원본 측정/프로토콜 검증 자료는 `records/`에 남겨 두었다. 준비에 사용한 센서/ROS 프로세스는 검사 후 종료하며 이후 실행 방법은 터널 테스트 안내를 따른다.
