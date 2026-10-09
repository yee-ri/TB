"""ROS-free cloud preprocessing regression tests; no transport or publishers."""
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes/stepbystep'))
from tunnel.cloud import tunnel_cloud_xy


SETTINGS = {'z_min': .03, 'z_max': .30, 'range_max': 3.}


def cloud(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    fields = [SimpleNamespace(name=name, offset=index*4, datatype=7, count=1)
              for index, name in enumerate(('x', 'y', 'z'))]
    return SimpleNamespace(fields=fields, is_bigendian=False, point_step=12,
                           row_step=12*len(points), width=len(points), height=1,
                           data=points.astype('<f4').tobytes())


def read_points(message, field_names, skip_nans):
    assert field_names in (('x', 'y', 'z'), ('x', 'y', 'z', 'tag'),
                           ('x', 'y', 'z', 'intensity'), ('x', 'y', 'z', 'intensity', 'tag'))
    assert not skip_nans
    # Match the ROS reader's per-record field-offset traversal, including
    # organized clouds and the Livox padding between XYZ and tag.
    fields = sorted((f for f in message.fields if f.name in field_names),
                    key=lambda f: f.offset)
    formats = {2: 'B', 7: 'f'}
    for row in range(message.height):
        for column in range(message.width):
            offset = row * message.row_step + column * message.point_step
            yield tuple(struct.unpack_from('<' + formats[f.datatype], message.data,
                                           offset + f.offset)[0] for f in fields)


def tagged_cloud(points, tags, height=1, intensities=None):
    message = cloud(points)
    assert len(tags) == message.width
    assert not len(tags) % height
    message.width //= height
    message.height = height
    message.point_step = 26
    message.row_step = message.width * message.point_step
    message.fields.extend([SimpleNamespace(name='intensity', offset=12, datatype=7, count=1),
                           SimpleNamespace(name='tag', offset=16, datatype=2, count=1)])
    payload = bytearray(len(tags) * message.point_step)
    intensities = [15.] * len(tags) if intensities is None else intensities
    assert len(intensities) == len(tags)
    for index, (xyz, tag) in enumerate(zip(points, tags)):
        struct.pack_into('<ffffB', payload, index * message.point_step,
                         *xyz, intensities[index], tag)
    message.data = bytes(payload)
    return message


def timed_cloud(points, times, header=42., tags=None, height=1, intensities=None):
    message = tagged_cloud(points, [0] * len(points) if tags is None else tags,
                           height, intensities)
    message.header = SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: header))
    message.fields.extend([SimpleNamespace(name='line', offset=17, datatype=2, count=1),
                           SimpleNamespace(name='timestamp', offset=18, datatype=8, count=1)])
    payload = bytearray(message.data)
    for index, stamp in enumerate(times):
        struct.pack_into('<d', payload, index * message.point_step + 18, stamp * 1e9)
    message.data = bytes(payload)
    return message


def reference(message, rotation, translation, settings):
    """Scalar transform, FLOAT32 wire rounding, then the old scalar filter."""
    retained = []
    for point in struct.iter_unpack('<fff', message.data):
        xyz = tuple(sum(float(rotation[i, j])*point[j] for j in range(3)) + translation[i]
                    for i in range(3))
        x, y, z = struct.unpack('<fff', struct.pack('<fff', *xyz))
        if (all(math.isfinite(v) for v in (x, y, z))
                and settings['z_min'] < z < settings['z_max']
                and math.hypot(x, y) <= settings['range_max']):
            retained.append((x, y))
    return np.asarray(retained, dtype=float).reshape(-1, 2)


def rotation(roll, pitch, yaw):
    cr,sr,cp,sp,cy,sy = (math.cos(roll),math.sin(roll),math.cos(pitch),
                         math.sin(pitch),math.cos(yaw),math.sin(yaw))
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class TunnelCloudTests(unittest.TestCase):
    def transform(self, message, matrix=None, translation=None, settings=None, motion_poses=None):
        return tunnel_cloud_xy(message, np.eye(3) if matrix is None else matrix,
                               np.zeros(3) if translation is None else translation,
                               SETTINGS if settings is None else settings, read_points, motion_poses)

    def test_matches_scalar_transform_filter_for_generic_rotations_and_nonfinite_data(self):
        rng = np.random.RandomState(947)
        values = np.concatenate([rng.uniform(-4, 4, (3000, 3)),
                                 [[float('nan'), 0, .1], [float('inf'), 0, .1],
                                  [-float('inf'), 0, .1], [0, float('inf'), .1],
                                  [0, 0, float('nan')], [0, 0, float('inf')]]])
        message = cloud(values)
        for rpy, xyz in [((0,0,0),(-.033073,0,.19165)),
                         ((0,math.radians(17),0),(.058026,.009,.131384)),
                         ((.7,-.9,1.2),(.17,-.11,.04)),
                         ((math.pi/2,0,math.pi/2),(.2,.4,.01))]:
            with self.subTest(rpy=rpy):
                matrix, translation = rotation(*rpy), np.asarray(xyz)
                expected = reference(message, matrix, translation, SETTINGS)
                np.testing.assert_array_equal(self.transform(message, matrix, translation), expected)

    def test_preserves_strict_height_inclusive_range_and_rear_side_returns(self):
        message = cloud([[3,0,.1],[-3,0,.1],[0,3,.1],[0,-3,.1],
                         [np.nextafter(np.float32(3),np.float32(4)),0,.1],
                         [0,0,.03],[0,0,.30],
                         [0,0,np.nextafter(np.float32(.03),np.float32(1))],
                         [0,0,np.nextafter(np.float32(.30),np.float32(0))]])
        actual = self.transform(message)
        np.testing.assert_array_equal(actual, reference(message,np.eye(3),np.zeros(3),SETTINGS))
        np.testing.assert_array_equal(actual, [[3,0],[-3,0],[0,3],[0,-3],[0,0],[0,0]])

    def test_rounds_transformed_coordinates_to_float32_before_height_filter(self):
        # In double this is below .30; the old cloud storage rounds it above .30.
        message = cloud([[1,0,.125]])
        actual = self.transform(message, translation=np.array([0.,0.,.174999999]))
        self.assertEqual(actual.shape, (0,2))

    def test_empty_and_entirely_invalid_clouds_do_not_invent_points(self):
        for points in ([], [[float('nan'),0,.1],[0,0,float('inf')]]):
            self.assertEqual(self.transform(cloud(points)).shape, (0,2))

    def test_unfamiliar_layouts_request_original_adapter_fallback(self):
        changes = [lambda m: setattr(m, 'is_bigendian', True),
                   lambda m: setattr(m.fields[0], 'datatype', 8),
                   lambda m: setattr(m.fields[0], 'offset', 4),
                   lambda m: setattr(m.fields[2], 'count', 2),
                   lambda m: setattr(m, 'row_step', m.row_step+8)]
        def forbidden_reader(*args, **kwargs):
            self.fail('Unsupported layout must fall back before reading')
        for change in changes:
            message = cloud([[1,0,.1]])
            change(message)
            self.assertIsNone(tunnel_cloud_xy(message,np.eye(3),np.zeros(3),SETTINGS,forbidden_reader))

    def test_finite_float32_overflow_is_not_silently_dropped(self):
        message = cloud([[3e38,0,.1]])
        self.assertIsNone(self.transform(message, translation=np.array([3e38,0,0])))

    def test_livox_filter_discards_only_nonzero_particle_bits(self):
        tags = [0x00, 0x10, 0x01, 0x30, 0xC0, 0x04, 0x08, 0x0C, 0x14, 0x09]
        points = [[.1 * i, .01 * i, .1] for i in range(len(tags))]
        message = tagged_cloud(points, tags, height=2)
        # Metadata order differs from physical byte order, as allowed by ROS.
        message.fields.reverse()
        settings = dict(SETTINGS, livox_particle_filter=True)
        expected = self.transform(cloud(points[:5]))
        np.testing.assert_array_equal(self.transform(message, settings=settings), expected)

    def test_livox_filter_preserves_coordinate_transform_and_existing_limits(self):
        rng = np.random.RandomState(645)
        points = rng.uniform(-4, 4, (1000, 3))
        tags = rng.randint(0, 256, len(points))
        message = tagged_cloud(points, tags)
        matrix, translation = rotation(.3, -.2, 1.1), np.array([-.033073, 0., .19165])
        settings = dict(SETTINGS, livox_particle_filter=True)
        original = cloud(points[(tags & 0x0C) == 0])
        expected = reference(original, matrix, translation, SETTINGS)
        np.testing.assert_array_equal(self.transform(message, matrix, translation, settings), expected)

    def test_livox_filter_defaults_off_and_leaves_all_tags_unchanged(self):
        points = [[.2,0,.1], [.3,0,.1], [.4,0,.1], [.5,0,.1]]
        message = tagged_cloud(points, [0x04, 0x08, 0x0C, 0xFF])
        expected = self.transform(cloud(points))
        np.testing.assert_array_equal(self.transform(message), expected)
        np.testing.assert_array_equal(self.transform(message, settings=dict(
            SETTINGS, livox_particle_filter=False)), expected)

    def test_livox_filter_empty_or_all_rejected_cloud_has_no_free_rays(self):
        settings = dict(SETTINGS, livox_particle_filter=True)
        for message in (tagged_cloud([], []),
                        tagged_cloud([[1,0,.1], [2,0,.1]], [0x04, 0x08])):
            result = self.transform(message, settings=settings)
            self.assertEqual(result.shape, (0, 2))
            self.assertEqual(result.size, 0)

    def test_livox_tag_rows_stay_aligned_across_nonfinite_coordinates(self):
        settings = dict(SETTINGS, livox_particle_filter=True)
        message = tagged_cloud([[float('nan'), 0, .1], [1,0,.1],
                                [float('inf'), 0, .1], [2,0,.1]], [0, 4, 8, 0])
        np.testing.assert_array_equal(self.transform(message, settings=settings), [[2, 0]])

    def test_livox_filter_rejects_missing_or_malformed_tag_before_reading(self):
        settings = dict(SETTINGS, livox_particle_filter=True)
        def tag(message):
            return next(f for f in message.fields if f.name == 'tag')
        changes = [lambda m: m.fields.remove(tag(m)),
                   lambda m: m.fields.append(tag(m)),
                   lambda m: setattr(tag(m), 'datatype', 7),
                   lambda m: setattr(tag(m), 'count', 2),
                   lambda m: setattr(tag(m), 'offset', 26),
                   lambda m: setattr(tag(m), 'offset', 12),
                   lambda m: setattr(tag(m), 'offset', 16.5),
                   lambda m: setattr(m, 'data', m.data[:-1]),
                   lambda m: setattr(m, 'data', m.data + b'\x00')]
        def forbidden_reader(*args, **kwargs):
            self.fail('Invalid tag metadata must fail before reading')
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                message = tagged_cloud([[1,0,.1]], [0])
                change(message)
                with self.assertRaises(ValueError):
                    tunnel_cloud_xy(message, np.eye(3), np.zeros(3), settings, forbidden_reader)

    def test_livox_filter_never_falls_back_for_unsupported_layout_or_overflow(self):
        settings = dict(SETTINGS, livox_particle_filter=True)
        changes = [lambda m: setattr(m, 'is_bigendian', True),
                   lambda m: setattr(m.fields[0], 'datatype', 8),
                   lambda m: setattr(m, 'row_step', m.row_step + 1),
                   lambda m: setattr(m, 'point_step', None),
                   lambda m: delattr(m, 'point_step')]
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                message = tagged_cloud([[1,0,.1]], [0])
                change(message)
                with self.assertRaises(ValueError):
                    self.transform(message, settings=settings)
        with self.assertRaises(ValueError):
            self.transform(tagged_cloud([[3e38,0,.1]], [0]),
                           translation=np.array([3e38,0.,0.]), settings=settings)

    def test_motion_compensation_static_pose_matches_original_transform_exactly(self):
        rng = np.random.RandomState(920)
        points = rng.uniform(-3, 3, (1000, 3))
        times = 42. + rng.uniform(0., .1, len(points))
        tags = rng.randint(0, 256, len(points))
        message = timed_cloud(points, times, tags=tags, height=2)
        message.fields.reverse()
        settings = dict(SETTINGS, livox_particle_filter=True)
        matrix, translation = rotation(.2, -.1, .6), np.array([-.033073, 0., .19165])
        expected = self.transform(message, matrix, translation, settings)
        actual = self.transform(message, matrix, translation,
                                dict(settings, livox_motion_compensation=True),
                                lambda stamps: np.tile([123., -41., 2.3], (len(stamps), 1)))
        np.testing.assert_array_equal(actual, expected)

    def test_motion_compensation_corrects_wall_during_curved_motion_and_yaw_wrap(self):
        header = 42.
        # Unordered packets include a small timestamp reversal before the
        # header, as seen in actual Livox input. Motion crosses +pi/-pi.
        times = np.array([42.08, 42.001, 42.1, 41.999533, 42.04, 42.099])
        mount = np.array([-.033073, .012, .19165])
        sensor_rotation = rotation(0., 0., .2)
        world_point = np.array([1.2, .9])
        def absolute_pose(stamps):
            delta = np.asarray(stamps) - header
            yaw = 3.12 + delta * .9
            return np.column_stack((.3 * np.sin(delta * .9),
                                    .3 * (1. - np.cos(delta * .9)),
                                    (yaw + np.pi) % (2 * np.pi) - np.pi))
        points = []
        for x, y, yaw in absolute_pose(times):
            dx, dy = world_point - [x, y]
            base = np.array([np.cos(yaw)*dx + np.sin(yaw)*dy,
                             -np.sin(yaw)*dx + np.cos(yaw)*dy, .15])
            points.append(sensor_rotation.T.dot(base - mount))
        message = timed_cloud(points, times, tags=[0, 0x10, 0x04, 0x01, 0, 0])
        queried = []
        def motion(stamps):
            queried.append(stamps.copy())
            return absolute_pose(stamps)
        actual = self.transform(message, sensor_rotation, mount,
                                dict(SETTINGS, livox_particle_filter=True,
                                     livox_motion_compensation=True), motion)
        yaw = absolute_pose([header])[0, 2]
        expected = [np.cos(yaw)*world_point[0] + np.sin(yaw)*world_point[1],
                    -np.sin(yaw)*world_point[0] + np.cos(yaw)*world_point[1]]
        np.testing.assert_allclose(actual, np.tile(expected, (5, 1)), atol=5e-6)
        self.assertEqual(len(queried), 1)
        self.assertTrue(np.all(np.diff(queried[0]) > 0.))
        self.assertLessEqual(np.max(np.diff(queried[0])), .010001)
        self.assertIn(header, queried[0])
        self.assertLessEqual(len(queried[0]), 13)

    def test_motion_compensation_uses_header_frame_range_and_leaves_height_unchanged(self):
        message = timed_cloud([[2.95, 0., .1], [2.7, 0., .3]], [42.1, 42.1])
        def motion(stamps):
            return np.column_stack((2. * (stamps - 42.), np.zeros(len(stamps)), np.zeros(len(stamps))))
        # First point is 3.15 m from the header base after translation; second
        # is over the strict height limit. Neither becomes a free-space ray.
        actual = self.transform(message, settings=dict(SETTINGS, livox_motion_compensation=True),
                                motion_poses=motion)
        self.assertEqual(actual.shape, (0, 2))

    def test_motion_compensation_defaults_off_without_timestamp_metadata(self):
        message = cloud([[.4, .1, .1]])
        def forbidden(stamps):
            self.fail('default-off compensation must not request TF')
        np.testing.assert_array_equal(self.transform(message, motion_poses=forbidden),
                                      self.transform(message))
        for value in ('true', 1, None):
            with self.assertRaises(ValueError):
                self.transform(message, settings=dict(SETTINGS, livox_motion_compensation=value))

    def test_motion_compensation_rejects_bad_timestamp_metadata_before_point_read(self):
        settings = dict(SETTINGS, livox_motion_compensation=True)
        def timestamp(message):
            return next(f for f in message.fields if f.name == 'timestamp')
        changes = [lambda m: m.fields.remove(timestamp(m)),
                   lambda m: m.fields.append(timestamp(m)),
                   lambda m: setattr(timestamp(m), 'datatype', 7),
                   lambda m: setattr(timestamp(m), 'count', 2),
                   lambda m: setattr(timestamp(m), 'offset', 16),
                   lambda m: setattr(timestamp(m), 'offset', 19),
                   lambda m: setattr(m.fields[3], 'offset', 11),
                   lambda m: setattr(m, 'data', m.data[:-1]),
                   lambda m: setattr(m, 'data', m.data + b'\0'),
                   lambda m: setattr(m, 'is_bigendian', True),
                   lambda m: setattr(m, 'row_step', m.row_step + 1)]
        def forbidden(*args, **kwargs):
            self.fail('invalid timestamps must fail before reading XYZ')
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                message = timed_cloud([[.5, 0., .1]], [42.05])
                change(message)
                with self.assertRaises(ValueError):
                    tunnel_cloud_xy(message, np.eye(3), np.zeros(3), settings, forbidden)
        for bad_time in (np.nan, np.inf, 42.3, 41.998, 42. / 1e9):
            with self.subTest(time=bad_time), self.assertRaises(ValueError):
                tunnel_cloud_xy(timed_cloud([[.5, 0., .1]], [bad_time]),
                                np.eye(3), np.zeros(3), settings, forbidden)

    def test_motion_compensation_rejects_missing_or_nonfinite_pose_without_fallback(self):
        message = timed_cloud([[.5, 0., .1]], [42.05])
        settings = dict(SETTINGS, livox_motion_compensation=True)
        for motion in (None, lambda stamps: [[0., 0., 0.]],
                       lambda stamps: np.full((len(stamps), 3), np.nan)):
            with self.assertRaises(ValueError):
                self.transform(message, settings=settings, motion_poses=motion)
        def missing(stamps):
            raise RuntimeError('missing acquisition TF')
        with self.assertRaisesRegex(RuntimeError, 'missing acquisition TF'):
            self.transform(message, settings=settings, motion_poses=missing)

    def test_horizontal_slice_includes_angle_edges_and_rejects_both_sides(self):
        outside = float(np.nextafter(np.float32(1.), np.float32(2.)))
        message = cloud([[1., 0., 1.], [1., 0., -1.], [1., 0., outside],
                         [1., 0., -outside], [0., 1., 0.], [-1., 0., 0.]])
        # No height-band keys are needed when a horizontal slice is selected.
        result = self.transform(message, settings={'horizontal_half_angle_deg': 45.,
                                                   'range_max': 3.})
        np.testing.assert_array_equal(result, [[1., 0.], [1., 0.], [0., 1.], [-1., 0.]])

    def test_horizontal_slice_selects_raw_lidar_plane_instead_of_body_height_band(self):
        message = cloud([[.15, 0., -.06],  # Nearby camera-height return.
                         [.15, 0., .05],   # Nearby roof return.
                         [.7, 0., 0.],     # Bottle crossing LiDAR height.
                         [3., 0., .04]])   # Within 1 degree, above old 22 cm band.
        mount = np.array([-.033073, 0., .19165])
        settings = dict(SETTINGS, z_max=.22, horizontal_half_angle_deg=1.)
        result = self.transform(message, translation=mount, settings=settings)
        expected = np.asarray([[.7 + mount[0], 0.], [3. + mount[0], 0.]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-7)
        # Mount pitch moves an accepted horizontal ray below z_min, but does
        # not change the raw LiDAR elevation used by this explicit mode.
        tilted = self.transform(cloud([[.4, 0., 0.]]), rotation(0., .5, 0.), mount, settings)
        self.assertEqual(len(tilted), 1)
        self.assertEqual(len(self.transform(cloud([[.4, 0., 0.]]), rotation(0., .5, 0.), mount,
                                            dict(SETTINGS, z_max=.22))), 0)

    def test_horizontal_slice_preserves_tag_and_unordered_motion_timestamp_alignment(self):
        message = timed_cloud([[1., 0., 0.], [2., 0., 0.], [2., 0., .1], [1.5, .1, .001]],
                              [42.08, 42.01, 42.1, 42.03], tags=[0, 4, 0, 0])
        settings = dict(SETTINGS, horizontal_half_angle_deg=1., livox_particle_filter=True,
                        livox_motion_compensation=True)
        def motion(stamps):
            return np.column_stack((stamps - 42., np.zeros(len(stamps)), np.zeros(len(stamps))))
        result = self.transform(message, settings=settings, motion_poses=motion)
        np.testing.assert_allclose(result, [[1.08, 0.], [1.53, .1]], atol=1e-7)

    def test_horizontal_slice_keeps_existing_range_and_finite_point_checks(self):
        message = cloud([[3., 0., 0.], [3.01, 0., 0.], [float('nan'), 0., 0.],
                         [float('inf'), 0., 0.], [1., 0., float('nan')]])
        result = self.transform(message, settings=dict(SETTINGS, horizontal_half_angle_deg=1.))
        np.testing.assert_array_equal(result, [[3., 0.]])

    def test_zero_degree_selects_only_raw_zero_height_before_mounting_transform(self):
        nonzero = float(np.nextafter(np.float32(0.), np.float32(1.)))
        message = cloud([[.5, 0., 0.], [.6, 0., -0.], [.7, 0., nonzero],
                         [.8, 0., -nonzero], [.9, 0., .001], [1., 0., -.001]])
        matrix, mount = rotation(0., .1, 0.), np.array([-.033073, 0., .19165])
        settings = dict(SETTINGS, horizontal_half_angle_deg=0.)
        expected = self.transform(cloud([[.5, 0., 0.], [.6, 0., -0.]]), matrix, mount)
        np.testing.assert_array_equal(self.transform(message, matrix, mount, settings), expected)

    def test_zero_degree_preserves_tag_motion_alignment_and_finite_range_limits(self):
        message = timed_cloud([[.5, 0., 0.], [.6, 0., 0.], [.7, 0., .001],
                               [.8, 0., 0.], [float('nan'), 0., 0.], [4., 0., 0.]],
                              [42.08, 42.01, 42.1, 42.03, 42.05, 42.02],
                              tags=[0, 4, 0, 0x10, 0, 0])
        settings = dict(SETTINGS, horizontal_half_angle_deg=0., livox_particle_filter=True,
                        livox_motion_compensation=True)
        def motion(stamps):
            return np.column_stack((stamps - 42., np.zeros(len(stamps)), np.zeros(len(stamps))))
        result = self.transform(message, translation=np.array([-.033073, 0., .19165]),
                                settings=settings, motion_poses=motion)
        np.testing.assert_allclose(result, [[.546927, 0.], [.796927, 0.]], atol=1e-7)

    def test_zero_degree_empty_and_sensor_origin_returns_do_not_refresh_mission_scan(self):
        from tunnel.mission import TunnelMission, load_config
        config_path = Path(__file__).resolve().parents[1] / 'codes/stepbystep/tunnel.yaml'
        mission = TunnelMission(load_config(config_path))
        self.addCleanup(mission.close)
        mount = np.array([-.033073, 0., .19165])
        sensor, odom = (-.033073, 0., 0.), (0., 0., 0.)
        self.assertTrue(mission.update_cloud([[.5, 0.]], sensor, odom, 10.))
        settings = dict(SETTINGS, horizontal_half_angle_deg=0., livox_particle_filter=True)
        cases = (tagged_cloud([], []), tagged_cloud([[.5, 0., .001]], [0]),
                 tagged_cloud([[.5, 0., 0.]], [4]), tagged_cloud([[0., 0., 0.]], [0]))
        for index, message in enumerate(cases):
            points = self.transform(message, translation=mount, settings=settings)
            self.assertFalse(mission.update_cloud(points, sensor, odom, 10.1 + index*.1))
            self.assertEqual(mission.scan[-1], 10.)
        mission.update_odometry(odom, 0., 0., 10.5)
        self.assertEqual(mission.step(10.5), (0., 0., False))
        self.assertIn('waiting for fresh', mission.reason)

    def test_horizontal_slice_rejects_invalid_angles_or_unsupported_xyz_without_fallback(self):
        for angle in (True, False, np.bool_(True), None, '1', np.nan, np.inf, -np.inf,
                      -.001, -1., 90., 91., complex(1., 0.)):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                self.transform(cloud([[1., 0., 0.]]),
                               settings=dict(SETTINGS, horizontal_half_angle_deg=angle))
        settings = dict(SETTINGS, horizontal_half_angle_deg=1.)
        changes = [lambda m: setattr(m, 'is_bigendian', True),
                   lambda m: setattr(m.fields[0], 'offset', 4),
                   lambda m: setattr(m.fields[0], 'datatype', 8),
                   lambda m: setattr(m, 'row_step', m.row_step + 1),
                   lambda m: setattr(m, 'data', m.data[:-1])]
        def forbidden(*args, **kwargs):
            self.fail('unsupported horizontal-slice input must fail before point reading')
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                message = cloud([[1., 0., 0.]])
                change(message)
                with self.assertRaises(ValueError):
                    tunnel_cloud_xy(message, np.eye(3), np.zeros(3), settings, forbidden)

    def test_minimum_intensity_defaults_disabled_without_requiring_or_filtering_intensity(self):
        points = [[.2, 0., .1], [.3, 0., .1], [.4, 0., .1], [.5, 0., .1]]
        expected = self.transform(cloud(points))
        for message in (cloud(points), tagged_cloud(points, [0, 4, 0, 0],
                                                   intensities=[0., 1., np.nan, 256.])):
            np.testing.assert_array_equal(self.transform(message), expected)
            np.testing.assert_array_equal(self.transform(message, settings=dict(
                SETTINGS, minimum_intensity=0)), expected)

    def test_minimum_intensity_includes_threshold_and_255_but_discards_invalid_values(self):
        intensities = [0., 1., 1.99, 2., 255., 256., -1., np.nan, np.inf, -np.inf]
        points = [[.1 * (index + 1), 0., .1] for index in range(len(intensities))]
        message = tagged_cloud(points, [0] * len(points), intensities=intensities)
        result = self.transform(message, settings=dict(SETTINGS, minimum_intensity=2))
        np.testing.assert_array_equal(result, self.transform(cloud(points[3:5])))
        result = self.transform(message, settings=dict(SETTINGS, minimum_intensity=255.))
        np.testing.assert_array_equal(result, self.transform(cloud(points[4:5])))

    def test_minimum_intensity_rejects_nonnumeric_nonfinite_and_out_of_range_settings(self):
        for value in (True, False, np.bool_(True), None, '2', np.nan, np.inf, -np.inf,
                      -.001, 255.001, complex(2., 0.)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.transform(cloud([[1., 0., .1]]),
                               settings=dict(SETTINGS, minimum_intensity=value))

    def test_minimum_intensity_rejects_missing_or_malformed_layout_without_fallback(self):
        settings = dict(SETTINGS, minimum_intensity=2)
        def intensity(message):
            return next(field for field in message.fields if field.name == 'intensity')
        changes = [lambda m: m.fields.remove(intensity(m)),
                   lambda m: m.fields.append(intensity(m)),
                   lambda m: setattr(intensity(m), 'datatype', 2),
                   lambda m: setattr(intensity(m), 'count', 2),
                   lambda m: setattr(intensity(m), 'offset', 13),
                   lambda m: setattr(intensity(m), 'offset', 12.5),
                   lambda m: setattr(m.fields[-1], 'offset', 15),
                   lambda m: setattr(m, 'point_step', 15),
                   lambda m: setattr(m, 'row_step', m.row_step + 1),
                   lambda m: setattr(m, 'width', 1.5),
                   lambda m: setattr(m, 'data', m.data[:-1]),
                   lambda m: setattr(m, 'data', m.data + b'\0'),
                   lambda m: setattr(m, 'is_bigendian', True),
                   lambda m: setattr(m.fields[0], 'datatype', 8)]
        def forbidden(*args, **kwargs):
            self.fail('invalid intensity metadata must fail before reading points')
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                message = tagged_cloud([[1., 0., .1]], [0])
                change(message)
                with self.assertRaises(ValueError):
                    tunnel_cloud_xy(message, np.eye(3), np.zeros(3), settings, forbidden)
        with self.assertRaises(ValueError):
            tunnel_cloud_xy(tagged_cloud([[1., 0., .1]], [0]), np.eye(3), np.zeros(3),
                            settings, lambda *args, **kwargs: iter(()))

    def test_minimum_intensity_keeps_tag_elevation_and_unordered_motion_rows_aligned(self):
        message = timed_cloud([[.5, 0., 0.], [.6, 0., 0.], [.7, 0., 0.],
                               [.8, 0., 0.], [np.nan, 0., 0.], [1., 0., .1]],
                              [42.08, 42.01, 42.1, 42.03, 42.05, 42.02],
                              tags=[0, 4, 0, 0x10, 0, 0], height=2,
                              intensities=[2., 2., 1., 3., 4., 255.])
        message.fields.reverse()
        settings = dict(SETTINGS, minimum_intensity=2, horizontal_half_angle_deg=.5,
                        livox_particle_filter=True, livox_motion_compensation=True)
        calls, queried = [], []
        def reader(*args, **kwargs):
            calls.append(kwargs['field_names'])
            return read_points(*args, **kwargs)
        def motion(stamps):
            queried.append(stamps.copy())
            return np.column_stack((stamps - 42., np.zeros(len(stamps)), np.zeros(len(stamps))))
        result = tunnel_cloud_xy(message, np.eye(3), np.zeros(3), settings, reader, motion)
        np.testing.assert_allclose(result, [[.58, 0.], [.83, 0.]], atol=1e-7)
        self.assertEqual(calls, [('x', 'y', 'z', 'intensity', 'tag')])
        self.assertEqual(len(queried), 1)
        # The weakest point at the final timestamp remains in the TF envelope.
        self.assertAlmostEqual(queried[0][-1], 42.1)

    def test_minimum_intensity_empty_or_all_rejected_input_does_not_refresh_mission_scan(self):
        from tunnel.mission import TunnelMission, load_config
        config_path = Path(__file__).resolve().parents[1] / 'codes/stepbystep/tunnel.yaml'
        mission = TunnelMission(load_config(config_path))
        self.addCleanup(mission.close)
        sensor, odom = (-.033073, 0., 0.), (0., 0., 0.)
        self.assertTrue(mission.update_cloud([[.5, 0.]], sensor, odom, 10.))
        settings = dict(SETTINGS, minimum_intensity=2)
        cases = (tagged_cloud([], []),
                 tagged_cloud([[.5, 0., .1], [.6, 0., .1]], [0, 0], intensities=[0., 1.]),
                 tagged_cloud([[.5, 0., .1]], [0], intensities=[np.nan]))
        for index, message in enumerate(cases):
            points = self.transform(message, settings=settings)
            self.assertEqual(points.shape, (0, 2))
            self.assertFalse(mission.update_cloud(points, sensor, odom, 10.1 + index * .1))
            self.assertEqual(mission.scan[-1], 10.)
        mission.update_odometry(odom, 0., 0., 10.5)
        self.assertEqual(mission.step(10.5), (0., 0., False))
        self.assertIn('waiting for fresh', mission.reason)


if __name__ == '__main__':
    unittest.main()
