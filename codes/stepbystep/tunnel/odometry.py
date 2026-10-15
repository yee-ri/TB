"""Motion calibration at the tunnel's ROS boundary only.

Raw /odom and TF stay unchanged for other missions. Tunnel positions and
measured linear speed use this scale; outgoing tunnel linear commands use
its inverse. An independent output scale corrects angular commands.
Measured IMU yaw/angular speed and LiDAR geometry remain unchanged.
"""

import math
from numbers import Real


def _read_scale(config, key):
    settings = config.get('odometry', {})
    if not isinstance(settings, dict):
        raise ValueError('odometry must be a mapping')
    scale = settings.get(key, 1.0)
    if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('odometry/%s must be finite and positive' % key)
    return float(scale)


def read_linear_scale(config):
    return _read_scale(config, 'linear_scale')


def read_angular_command_scale(config):
    return _read_scale(config, 'angular_command_scale')


def scale_pose(pose, scale):
    return (pose[0] * scale, pose[1] * scale, pose[2])
