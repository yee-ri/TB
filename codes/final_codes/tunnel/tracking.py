"""ROS-independent forward tracking helpers for a sampled tunnel path.

Tracking functions come from custom_autorace_bringup/zigzag_path.py;
the moving-exit quintic and sampled geometry come from parking_geometry.py
and path_following.py in the same source workspace. Only the used helpers
are copied; complete parking and other mission modules are not dependencies.
"""

from dataclasses import dataclass
import math
import operator

import numpy as np


@dataclass(frozen=True)
class TrackingPath:
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    station: np.ndarray
    speed: np.ndarray




# Extracted from source parking_geometry.py and path_following.py. Keep the
# same quintic geometry and sampled heading-gradient curvature at the exit.
def quintic_pose_path(
    start_pose,
    end_pose,
    start_tangent_length,
    end_tangent_length,
    sample_count,
):
    """Sample a zero-end-curvature quintic between two planar poses.

    The first and last three control points are equally spaced and collinear
    with their respective pose headings.  Position, heading and curvature are
    therefore continuous when this path is joined to another path with the
    same endpoint pose and zero endpoint curvature.
    """
    try:
        start = np.asarray(start_pose, dtype=np.float64)
        end = np.asarray(end_pose, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "start_pose and end_pose must contain three finite values"
        ) from error
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
    ):
        raise ValueError(
            "start_pose and end_pose must contain three finite values"
        )

    try:
        start_tangent_length = float(start_tangent_length)
        end_tangent_length = float(end_tangent_length)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("tangent lengths must be finite and positive") from error
    if not (
        math.isfinite(start_tangent_length)
        and math.isfinite(end_tangent_length)
        and start_tangent_length > 0.0
        and end_tangent_length > 0.0
    ):
        raise ValueError("tangent lengths must be finite and positive")

    if isinstance(sample_count, (bool, np.bool_)):
        raise ValueError("sample_count must be an integer of at least three")
    try:
        sample_count = operator.index(sample_count)
    except TypeError as error:
        raise ValueError(
            "sample_count must be an integer of at least three"
        ) from error
    if sample_count < 3:
        raise ValueError("sample_count must be an integer of at least three")

    start_direction = np.asarray(
        [math.cos(start[2]), math.sin(start[2])], dtype=np.float64
    )
    end_direction = np.asarray(
        [math.cos(end[2]), math.sin(end[2])], dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        control_points = np.asarray(
            [
                start[:2],
                start[:2] + start_tangent_length * start_direction,
                start[:2] + 2.0 * start_tangent_length * start_direction,
                end[:2] - 2.0 * end_tangent_length * end_direction,
                end[:2] - end_tangent_length * end_direction,
                end[:2],
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(control_points)):
        raise ValueError("path dimensions produce non-finite control points")

    parameter = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    complement = 1.0 - parameter
    positions = np.zeros((sample_count, 2), dtype=np.float64)
    derivatives = np.zeros((sample_count, 2), dtype=np.float64)
    derivative_controls = 5.0 * np.diff(control_points, axis=0)
    for index in range(6):
        weight = (
            math.comb(5, index)
            * complement ** (5 - index)
            * parameter ** index
        )
        positions += weight[:, None] * control_points[index]
    for index in range(5):
        weight = (
            math.comb(4, index)
            * complement ** (4 - index)
            * parameter ** index
        )
        derivatives += weight[:, None] * derivative_controls[index]

    derivative_norm = np.linalg.norm(derivatives, axis=1)
    if not np.all(np.isfinite(derivative_norm)) or np.any(
        derivative_norm <= np.finfo(np.float64).tiny
    ):
        raise ValueError("path dimensions produce a degenerate path")
    headings = np.arctan2(derivatives[:, 1], derivatives[:, 0])
    headings[0] = normalize_angle(start[2])
    headings[-1] = normalize_angle(end[2])
    return np.column_stack((positions, headings))


def geometry_from_poses(poses):
    """Return body heading, curvature, and station for explicit SE(2) poses.

    Unlike :func:`geometry_from_xy`, this keeps the supplied body heading.
    That distinction is required for reverse paths, whose body heading is
    opposite their direction of travel.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] < 2:
        raise ValueError("path poses must be an Nx3 array with N >= 2")
    if not np.all(np.isfinite(poses)):
        raise ValueError("path poses must be finite")
    segment = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    if np.any(segment <= 1e-9):
        raise ValueError("path contains duplicate consecutive points")
    station = np.concatenate(([0.0], np.cumsum(segment)))
    unwrapped_heading = np.unwrap(poses[:, 2])
    curvature = np.gradient(unwrapped_heading, station, edge_order=1)
    heading = np.asarray(
        [normalize_angle(value) for value in poses[:, 2]], dtype=np.float64
    )
    return heading, curvature, station


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))



def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))



@dataclass(frozen=True)
class TrackingResult:
    path_index: int
    target_index: int
    position_error: float
    heading_error: float
    target_speed: float
    curvature_command: float
    angular_velocity: float



@dataclass(frozen=True)
class LimitedCommand:
    linear_velocity: float
    angular_velocity: float



def build_speed_profile(
    station,
    curvature,
    cruise_velocity,
    minimum_velocity,
    entry_velocity,
    exit_velocity,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
    linear_acceleration,
    linear_deceleration,
    maximum_angular_acceleration=math.inf,
):
    count = int(station.size)
    if count < 2:
        return np.full(count, minimum_velocity, dtype=np.float64)
    segment = np.diff(station)
    absolute_curvature = np.maximum(np.abs(curvature), 1e-6)
    angular_limit = maximum_angular_velocity / absolute_curvature
    lateral_limit = np.sqrt(maximum_lateral_acceleration / absolute_curvature)
    speed = np.minimum.reduce(
        (
            np.full(count, cruise_velocity, dtype=np.float64),
            angular_limit,
            lateral_limit,
        )
    )
    speed = np.maximum(speed, minimum_velocity)
    speed[0] = min(speed[0], entry_velocity)
    speed[-1] = min(speed[-1], exit_velocity)

    def apply_linear_limits():
        for index in range(count - 2, -1, -1):
            reachable = math.sqrt(
                max(
                    0.0,
                    speed[index + 1] ** 2
                    + 2.0 * linear_deceleration * segment[index],
                )
            )
            speed[index] = min(speed[index], reachable)
        for index in range(count - 1):
            reachable = math.sqrt(
                max(
                    0.0,
                    speed[index] ** 2
                    + 2.0 * linear_acceleration * segment[index],
                )
            )
            speed[index + 1] = min(speed[index + 1], reachable)

    # The backward pass is the curvature look-ahead: a low corner speed is
    # propagated toward earlier samples by the configured deceleration bound.
    for _ in range(3):
        apply_linear_limits()

    maximum_angular_acceleration = float(maximum_angular_acceleration)
    if math.isfinite(maximum_angular_acceleration):
        maximum_angular_acceleration = max(0.05, maximum_angular_acceleration)
        # omega = v * curvature. Reducing both end speeds of an offending
        # segment by sqrt(limit / measured) reduces d(omega)/dt by the same
        # ratio squared. Longitudinal passes then propagate the braking ahead.
        for _ in range(100):
            duration = 2.0 * segment / np.maximum(
                speed[:-1] + speed[1:], 1e-6
            )
            omega = speed * curvature
            angular_acceleration = np.abs(np.diff(omega)) / np.maximum(
                duration, 1e-6
            )
            offending = np.flatnonzero(
                angular_acceleration
                > maximum_angular_acceleration * (1.0 + 1e-9)
            )
            if offending.size == 0:
                break
            for index in offending:
                scale = math.sqrt(
                    maximum_angular_acceleration
                    / max(float(angular_acceleration[index]), 1e-12)
                )
                scale *= 1.0 - 1e-6
                speed[index] *= scale
                speed[index + 1] *= scale
            apply_linear_limits()
        else:
            raise ValueError(
                "zigzag speed profile cannot satisfy angular acceleration"
            )
    return speed



def nearest_path_index(
    path,
    x,
    y,
    previous_index=0,
    search_back=3,
    search_ahead_distance=0.30,
):
    if path.x.size == 0:
        return 0
    first = max(0, min(int(previous_index) - int(search_back), path.x.size - 1))
    maximum_station = (
        float(path.station[max(0, min(int(previous_index), path.x.size - 1))])
        + max(0.05, float(search_ahead_distance))
    )
    last = int(np.searchsorted(path.station, maximum_station, side="right"))
    last = max(first + 1, min(last, path.x.size))
    distance = np.hypot(path.x[first:last] - x, path.y[first:last] - y)
    candidate = first + int(np.argmin(distance))
    return max(int(previous_index), candidate)



def calculate_tracking(
    path,
    x,
    y,
    yaw,
    previous_index,
    lookahead_distance,
    maximum_angular_velocity,
    heading_gain,
    path_curvature_weight,
    search_ahead_distance=0.30,
):
    index = nearest_path_index(
        path,
        x,
        y,
        previous_index,
        search_ahead_distance=search_ahead_distance,
    )
    position_error = math.hypot(path.x[index] - x, path.y[index] - y)
    heading_error = normalize_angle(float(path.heading[index]) - yaw)
    target_station = float(path.station[index]) + max(
        0.01, float(lookahead_distance)
    )
    target = min(
        int(np.searchsorted(path.station, target_station, side="left")),
        path.x.size - 1,
    )
    dx = float(path.x[target]) - x
    dy = float(path.y[target]) - y
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    target_x = cosine * dx + sine * dy
    target_y = -sine * dx + cosine * dy
    distance_squared = max(0.0025, target_x * target_x + target_y * target_y)
    pure_pursuit_curvature = 2.0 * target_y / distance_squared
    weight = clamp(float(path_curvature_weight), 0.0, 1.0)
    curvature_command = (
        (1.0 - weight) * pure_pursuit_curvature
        + weight * float(path.curvature[target])
    )
    target_speed = float(np.min(path.speed[index : target + 1]))
    target_heading_error = normalize_angle(float(path.heading[target]) - yaw)
    angular_velocity = clamp(
        target_speed * curvature_command + heading_gain * target_heading_error,
        -maximum_angular_velocity,
        maximum_angular_velocity,
    )
    return TrackingResult(
        path_index=index,
        target_index=target,
        position_error=position_error,
        heading_error=heading_error,
        target_speed=target_speed,
        curvature_command=curvature_command,
        angular_velocity=angular_velocity,
    )



def limit_tracking_command(
    target_speed,
    reference_speed,
    target_angular_velocity,
    last_linear_velocity,
    last_angular_velocity,
    elapsed,
    linear_acceleration,
    linear_deceleration,
    angular_acceleration,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
):
    """Rate-limit a tracking command without changing its intended curve.

    ``target_angular_velocity`` was calculated at ``reference_speed``. If a
    line or tracking guard lowers the linear target, the same scale must be
    applied to angular velocity; otherwise slowing down makes the driven path
    turn more sharply. When steering cannot ramp up quickly enough, linear
    speed is reduced so the robot does not understeer into the paint.
    """
    target_speed = max(0.0, float(target_speed))
    reference_speed = max(1e-6, abs(float(reference_speed)))
    curvature = float(target_angular_velocity) / reference_speed
    elapsed = max(0.0, float(elapsed))
    linear_acceleration = max(0.0, float(linear_acceleration))
    linear_deceleration = max(0.0, float(linear_deceleration))
    angular_acceleration = max(0.0, float(angular_acceleration))
    maximum_angular_velocity = max(0.0, float(maximum_angular_velocity))
    maximum_lateral_acceleration = max(
        0.0, float(maximum_lateral_acceleration)
    )

    if abs(curvature) > 1e-9:
        target_speed = min(
            target_speed,
            math.sqrt(maximum_lateral_acceleration / abs(curvature)),
        )

    acceleration = (
        linear_acceleration
        if target_speed >= last_linear_velocity
        else linear_deceleration
    )
    linear = clamp(
        target_speed,
        float(last_linear_velocity) - acceleration * elapsed,
        float(last_linear_velocity) + acceleration * elapsed,
    )
    linear = max(0.0, linear)
    if abs(curvature) > 1e-9:
        linear = min(
            linear,
            math.sqrt(maximum_lateral_acceleration / abs(curvature)),
        )

    desired_angular = clamp(
        linear * curvature,
        -maximum_angular_velocity,
        maximum_angular_velocity,
    )
    angular = clamp(
        desired_angular,
        float(last_angular_velocity) - angular_acceleration * elapsed,
        float(last_angular_velocity) + angular_acceleration * elapsed,
    )

    # If the wheel-speed ramp cannot yet create the requested bend, wait with
    # linear motion rather than cutting the corner. Excess angular velocity
    # while unwinding cannot be fixed this way, so it is only bounded below by
    # the lateral-acceleration guard.
    if (
        abs(curvature) > 1e-9
        and abs(angular) + 1e-12 < abs(desired_angular)
    ):
        if angular * curvature > 0.0:
            linear = min(linear, abs(angular / curvature))
            desired_angular = linear * curvature
            angular = clamp(
                desired_angular,
                float(last_angular_velocity) - angular_acceleration * elapsed,
                float(last_angular_velocity) + angular_acceleration * elapsed,
            )
        else:
            linear = 0.0

    if abs(angular) > 1e-9:
        linear = min(linear, maximum_lateral_acceleration / abs(angular))
    return LimitedCommand(linear, angular)
