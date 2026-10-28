"""Velocity commands and acceleration limits."""
from dataclasses import dataclass
import math
import numpy as np


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class LimitedCommand:
    linear_velocity: float
    angular_velocity: float
    reason: str = ''


def turn_command(error, linear_velocity, previous_angular_velocity, elapsed,
                 control, gain):
    """Limit a turn by angular ramp and lateral acceleration.

    The caller supplies its already-limited linear speed.
    Collision checking stays with the geometry-aware caller.
    """
    angular_limit = control['maximum_angular_velocity']
    if linear_velocity > 1e-9:
        angular_limit = min(angular_limit, control['maximum_lateral_acceleration']/linear_velocity)
    desired = math.copysign(min(angular_limit, gain*abs(error)), error)
    delta = control['angular_acceleration']*elapsed
    angular = float(np.clip(desired, previous_angular_velocity-delta,
                            previous_angular_velocity+delta))
    angular = float(np.clip(angular, -angular_limit, angular_limit))
    return LimitedCommand(linear_velocity, angular)


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
    speed limit lowers the linear target, angular velocity scales with it to
    retain curvature. Linear speed is reduced when the angular ramp cannot
    produce the requested curvature.
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
