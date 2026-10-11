"""PointCloud2 preprocessing without constructing another PointCloud2 message.

No ROS imports: the adapter supplies sensor_msgs' existing point reader and
its acquisition-time rigid transform. Unsupported layouts use the old adapter
path unless explicit Livox processing requires verified per-point metadata.
"""

import numpy as np


def _livox_point_times(cloud, fields):
    """Read verified Livox absolute FLOAT64 nanoseconds, never offset times."""
    try:
        integers = (cloud.width, cloud.height, cloud.point_step, cloud.row_step)
        if any(not isinstance(value, (int, np.integer)) for value in integers):
            raise ValueError('cloud dimensions must be integers')
        if (cloud.width < 0 or cloud.height < 0 or cloud.point_step < 20
                or cloud.row_step != cloud.width * cloud.point_step):
            raise ValueError('cloud dimensions or strides are invalid')
        sizes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
        names, previous_end = set(), 0
        for field in fields:
            size = sizes.get(field.datatype)
            if (field.name in names or size is None
                    or not isinstance(field.count, (int, np.integer)) or field.count < 1
                    or not isinstance(field.offset, (int, np.integer))
                    or field.offset < previous_end
                    or field.offset + size * field.count > cloud.point_step):
                raise ValueError('point fields are duplicate, overlapping or out of bounds')
            names.add(field.name)
            previous_end = field.offset + size * field.count
        timestamps = [field for field in fields if field.name == 'timestamp']
        if (len(timestamps) != 1 or timestamps[0].datatype != 8
                or timestamps[0].count != 1):
            raise ValueError('timestamp must be FLOAT64 with count 1')
        if memoryview(cloud.data).nbytes != cloud.row_step * cloud.height:
            raise ValueError('data length does not match complete cloud rows')
        header = float(cloud.header.stamp.to_sec())
        if not np.isfinite(header) or header <= 0.0:
            raise ValueError('a positive acquisition header time is required')
        if cloud.width * cloud.height == 0:
            return header, np.empty(0, dtype=np.float64)
        times = np.ndarray((cloud.height, cloud.width), dtype='<f8',
                           buffer=cloud.data, offset=timestamps[0].offset,
                           strides=(cloud.row_step, cloud.point_step)).reshape(-1) / 1e9
        # The physical driver emits 10 Hz clouds. Permit packet reordering at
        # the beginning, but reject wrong units, corrupt or long-span scans.
        # This is metadata validation, not an extension of freshness limits.
        if (not np.isfinite(times).all() or np.min(times) < header - .001
                or np.max(times) > header + .2):
            raise ValueError('point times must be absolute nanoseconds within one acquisition')
        return header, times
    except (AttributeError, TypeError, ValueError, BufferError) as error:
        raise ValueError('Livox motion compensation requires valid timestamps: ' + str(error)) from error


def _compensate_motion(points, header, point_times, motion_poses):
    """Express acquisition-time base points in the cloud header's base frame."""
    if not callable(motion_poses):
        raise ValueError('Livox motion compensation requires acquisition-time odometry TF')
    if not len(points):
        return points
    low, high = min(header, np.min(point_times)), max(header, np.max(point_times))
    # Sorted knots tolerate packet-order timestamp jitter. The adapter waits
    # for the newest TF once; all other lookups must be immediately available.
    intervals = max(1, int(np.ceil((high - low) / .01)))
    knots = np.unique(np.append(np.linspace(low, high, intervals + 1), header))
    poses = np.asarray(motion_poses(knots), dtype=np.float64)
    if poses.shape != (len(knots), 3) or not np.isfinite(poses).all():
        raise ValueError('Livox motion compensation received invalid odometry poses')
    poses[:, 2] = np.unwrap(poses[:, 2])
    reference = poses[np.searchsorted(knots, header)]
    acquired = np.column_stack([np.interp(point_times, knots, poses[:, i]) for i in range(3)])
    dx, dy = acquired[:, 0] - reference[0], acquired[:, 1] - reference[1]
    c0, s0 = np.cos(reference[2]), np.sin(reference[2])
    angle = acquired[:, 2] - reference[2]
    cosine, sine = np.cos(angle), np.sin(angle)
    x, y = points[:, 0].copy(), points[:, 1].copy()
    points[:, 0] = cosine * x - sine * y + c0 * dx + s0 * dy
    points[:, 1] = sine * x + cosine * y - s0 * dx + c0 * dy
    return points


def _require_livox_tag_layout(cloud, fields):
    """Validate the tag byte and full rows before an enabled filter reads data."""
    try:
        tags = [field for field in fields if field.name == 'tag']
        if len(tags) != 1:
            raise ValueError('exactly one tag field is required')
        tag = tags[0]
        if tag.datatype != 2 or tag.count != 1:  # PointField.UINT8
            raise ValueError('tag must be UINT8 with count 1')
        integers = (cloud.width, cloud.height, cloud.point_step, cloud.row_step,
                    tag.offset)
        if any(not isinstance(value, (int, np.integer)) for value in integers):
            raise ValueError('cloud dimensions and tag offset must be integers')
        if (cloud.width < 0 or cloud.height < 0 or cloud.point_step < 13
                or not 12 <= tag.offset < cloud.point_step):
            raise ValueError('tag byte or cloud dimensions are out of bounds')
        sizes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
        for field in fields:
            if field is tag:
                continue
            size = sizes.get(field.datatype)
            if (size is None or not isinstance(field.count, (int, np.integer))
                    or field.count < 1
                    or not isinstance(field.offset, (int, np.integer))
                    or field.offset < 0
                    or field.offset + size * field.count > cloud.point_step):
                raise ValueError('point field extends outside the supported layout')
            if field.offset <= tag.offset < field.offset + size * field.count:
                raise ValueError('tag byte overlaps another field')
        if memoryview(cloud.data).nbytes != cloud.row_step * cloud.height:
            raise ValueError('data length does not match complete cloud rows')
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Livox particle filter requires a valid tag layout: ' + str(error)) from error


def _require_livox_intensity_layout(cloud, fields):
    """Require the driver's FLOAT32 intensity at byte 12 and complete rows."""
    try:
        intensities = [field for field in fields if field.name == 'intensity']
        if (len(intensities) != 1 or intensities[0].datatype != 7
                or intensities[0].count != 1 or intensities[0].offset != 12):
            raise ValueError('intensity must be FLOAT32 at offset 12 with count 1')
        dimensions = (cloud.width, cloud.height, cloud.point_step, cloud.row_step)
        if (any(not isinstance(value, (int, np.integer)) for value in dimensions)
                or cloud.width < 0 or cloud.height < 0 or cloud.point_step < 16
                or cloud.row_step != cloud.width * cloud.point_step):
            raise ValueError('cloud dimensions or strides are invalid')
        sizes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
        names, previous_end = set(), 0
        for field in fields:
            size = sizes.get(field.datatype)
            if (field.name in names or size is None
                    or not isinstance(field.count, (int, np.integer)) or field.count < 1
                    or not isinstance(field.offset, (int, np.integer))
                    or field.offset < previous_end
                    or field.offset + size * field.count > cloud.point_step):
                raise ValueError('point fields are duplicate, overlapping or out of bounds')
            names.add(field.name)
            previous_end = field.offset + size * field.count
        if memoryview(cloud.data).nbytes != cloud.row_step * cloud.height:
            raise ValueError('data length does not match complete cloud rows')
    except (AttributeError, TypeError, ValueError, BufferError) as error:
        raise ValueError('Minimum intensity requires a valid Livox layout: ' + str(error)) from error


def tunnel_cloud_xy(cloud, rotation, translation, settings, read_points, motion_poses=None,
                    wall_height_band=None):
    """Return the existing height/range-filtered XY values, or None for fallback.

    The fast path is deliberately limited to contiguous, little-endian XYZ
    FLOAT32 clouds. Reading still uses sensor_msgs, including its field offsets.
    The intermediate FLOAT32 rounding matches tf2's transformed cloud storage;
    filtering then uses doubles, like the previous Python float loop.

    ``livox_particle_filter`` defaults to False. When explicitly enabled for
    Mid-360, discard points whose tag bits 2-3 (mask 0x0C) are nonzero; all
    other tag bits are preserved. See the manufacturer's Tag Information:
    https://livox-wiki-en.readthedocs.io/en/latest/tutorials/new_product/mid360/livox_eth_protocol_mid360.html#tag-information
    This mode raises ValueError for unsupported data rather than falling back
    to an unfiltered cloud. Rejected points do not become free-space rays.

    ``livox_motion_compensation`` also defaults to False. Its ``motion_poses``
    callback receives sorted acquisition timestamps (seconds, <=10 ms apart)
    and returns matching (x, y, yaw) odometry poses in one fixed frame. Points
    are moved into the header base frame before the existing FLOAT32 rounding
    and filters. Missing TF or malformed timestamps reject the whole cloud.

    Optional ``horizontal_half_angle_deg`` replaces the base-height band with
    a slice around the raw LiDAR's horizontal plane. The angle is evaluated
    before mounting transforms or motion compensation; range, finite-point
    and enabled Livox quality checks still apply. An absent setting preserves
    the height-band behavior. Zero selects raw z == 0 exactly (the driver's
    recorded zero-height coordinate), without adding an angular tolerance.

    ``minimum_intensity`` defaults to zero (disabled). A positive threshold
    retains only finite intensity values between the threshold and 255,
    inclusive, independently of tag filtering. Invalid intensity metadata
    rejects the cloud; invalid or weaker values reject only their points,
    without supplying free-space rays.

    ``wall_height_band`` optionally returns (obstacle_xy, wall_xy) from the
    same transform. Wall XY is used only for stopped initial registration:
    its ground-height band replaces the horizontal slice and includes weak
    intensity 0/1 returns from black walls. Particle rejection, acquisition
    transforms, finite coordinates and range checks still apply. It must
    never be passed to the obstacle costmap in place of obstacle_xy.
    """
    if wall_height_band is not None:
        band = np.asarray(wall_height_band, dtype=float)
        if band.shape != (2,) or not np.isfinite(band).all() or not 0. <= band[0] < band[1]:
            raise ValueError('wall_height_band must contain two increasing finite ground heights')
    particle_filter = settings.get('livox_particle_filter', False)
    if not isinstance(particle_filter, bool):
        raise ValueError('livox_particle_filter must be a boolean')
    motion_compensation = settings.get('livox_motion_compensation', False)
    if not isinstance(motion_compensation, bool):
        raise ValueError('livox_motion_compensation must be a boolean')
    minimum_intensity = settings.get('minimum_intensity', 0.)
    if (isinstance(minimum_intensity, (bool, np.bool_))
            or not isinstance(minimum_intensity, (int, float, np.integer, np.floating))):
        raise ValueError('minimum_intensity must be a finite number between 0 and 255')
    try:
        minimum_intensity = float(minimum_intensity)
    except (ValueError, OverflowError) as error:
        raise ValueError('minimum_intensity is out of range') from error
    if not np.isfinite(minimum_intensity) or not 0. <= minimum_intensity <= 255.:
        raise ValueError('minimum_intensity must be a finite number between 0 and 255')
    intensity_filter = minimum_intensity > 0.
    horizontal_slice = 'horizontal_half_angle_deg' in settings
    if horizontal_slice:
        angle = settings['horizontal_half_angle_deg']
        if (isinstance(angle, (bool, np.bool_))
                or not isinstance(angle, (int, float, np.integer, np.floating))):
            raise ValueError('horizontal_half_angle_deg must be a finite number >= 0 and < 90')
        try:
            angle = float(angle)
        except (ValueError, OverflowError) as error:
            raise ValueError('horizontal_half_angle_deg is out of range') from error
        if not np.isfinite(angle) or not 0. <= angle < 90.:
            raise ValueError('horizontal_half_angle_deg must be a finite number >= 0 and < 90')
        # Include the mathematical angle boundary despite one-ULP rounding
        # in tan (for example, tan(45 degrees) may be just below 1).
        horizontal_slope = 0. if angle == 0. else np.nextafter(np.tan(np.radians(angle)), np.inf)
    strict = particle_filter or motion_compensation or horizontal_slice or intensity_filter or wall_height_band is not None
    try:
        fields = sorted(cloud.fields, key=lambda field: field.offset)
        supported = (
            not cloud.is_bigendian
            and cloud.point_step >= 12
            and cloud.row_step == cloud.width * cloud.point_step
            and len(fields) >= 3
            and [(f.name, f.offset, f.datatype, f.count) for f in fields[:3]]
                == [('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1)]
            and all(f.offset >= 12 and f.name not in ('x', 'y', 'z') for f in fields[3:])
        )
    except (AttributeError, TypeError, ValueError) as error:
        if strict:
            raise ValueError('Livox processing requires supported XYZ metadata') from error
        return None
    if not supported:
        if strict:
            raise ValueError('Livox processing requires a supported XYZ layout')
        return None

    if horizontal_slice:
        dimensions = (cloud.width, cloud.height, cloud.point_step, cloud.row_step)
        if (any(not isinstance(value, (int, np.integer)) for value in dimensions)
                or cloud.width < 0 or cloud.height < 0
                or memoryview(cloud.data).nbytes != cloud.row_step * cloud.height):
            raise ValueError('Horizontal LiDAR slice requires complete cloud rows')

    if motion_compensation:
        header, point_times = _livox_point_times(cloud, fields)

    field_names = ('x', 'y', 'z')
    if intensity_filter:
        _require_livox_intensity_layout(cloud, fields)
        field_names += ('intensity',)
    if particle_filter:
        _require_livox_tag_layout(cloud, fields)
        field_names += ('tag',)
    rows = np.asarray(list(read_points(cloud, field_names=field_names,
                                      skip_nans=False)), dtype=np.float64).reshape(-1, len(field_names))
    if strict and len(rows) != cloud.width * cloud.height:
        raise ValueError('Livox processing did not read every cloud point')
    points = rows[:, :3]
    if particle_filter:
        particle_keep = (rows[:, -1].astype(np.uint8) & 0x0C) == 0
    if intensity_filter:
        intensity = rows[:, 3]
        with np.errstate(invalid='ignore'):
            intensity_keep = (np.isfinite(intensity) & (intensity >= minimum_intensity)
                              & (intensity <= 255.))
    # Nonfinite input is rejected by the same post-transform filter as before.
    # Let the original serializer handle finite values outside FLOAT32 range,
    # rather than silently turning them into omitted infinite obstacles.
    with np.errstate(invalid='ignore', over='raise'):
        if horizontal_slice:
            if angle == 0.:
                horizontal_keep = points[:, 2] == 0.
            else:
                horizontal_keep = (np.abs(points[:, 2])
                                   <= np.hypot(points[:, 0], points[:, 1]) * horizontal_slope)
        points = points.dot(rotation.T) + translation
        if motion_compensation:
            points = _compensate_motion(points, header, point_times, motion_poses)
        if np.any(np.isfinite(points) & (np.abs(points) > np.finfo(np.float32).max)):
            if strict:
                raise ValueError('Livox filtered transform exceeds FLOAT32 coordinates')
            return None
        points = points.astype(np.float32).astype(np.float64)
        keep = (np.isfinite(points).all(axis=1)
                & (np.hypot(points[:, 0], points[:, 1]) <= settings['range_max']))
        if wall_height_band is not None:
            wall_keep = keep & (points[:, 2] > band[0]) & (points[:, 2] < band[1])
        if horizontal_slice:
            keep &= horizontal_keep
        else:
            keep &= ((points[:, 2] > settings['z_min'])
                     & (points[:, 2] < settings['z_max']))
    if particle_filter:
        keep &= particle_keep
        if wall_height_band is not None:
            wall_keep &= particle_keep
    if intensity_filter:
        keep &= intensity_keep
        if wall_height_band is not None:
            with np.errstate(invalid='ignore'):
                wall_keep &= np.isfinite(intensity) & (intensity >= 0.) & (intensity <= 255.)
    if wall_height_band is not None:
        return points[keep, :2], points[wall_keep, :2]
    return points[keep, :2]
