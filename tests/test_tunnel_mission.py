"""Tunnel sensor, map, entry and handoff checks using synthetic ROS-free input."""

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))

from tunnel.mission import TunnelMission, load_config, relative_pose
from tunnel.planner import GlobalRoute
from tunnel.geometry import OccupancyGrid, Pose2D


CONFIG = Path(__file__).resolve().parents[1] / "codes/stepbystep/tunnel.yaml"


def occupied(grid, column, row):
    return bool(grid._occupied[row, column])


def configuration(goal=(.65, 0., 0.)):
    config = copy.deepcopy(load_config(CONFIG))
    config['goal']['pose'] = list(goal)
    config['grid']['bounds'] = [-.5, -.75, 1.5, 1.25]
    # Synthetic rooms have their own walls, not the source Gazebo course map.
    config['grid'].pop('static_map', None)
    config['grid'].pop('static_hit_exclusion_radius', None)
    config['entry']['distance'] = .30
    # This ideal room is not the source course's legal centre-line regions.
    config['planner']['keep_in_rectangles'] = None
    return config


def transform(pose, origin):
    c, s = math.cos(origin[2]), math.sin(origin[2])
    return (origin[0] + c*pose[0] - s*pose[1],
            origin[1] + s*pose[0] + c*pose[1], origin[2] + pose[2])


def integrate(pose, linear, angular, duration):
    """Independent exact ideal differential-drive motion, in metres/radians."""
    x, y, yaw = pose
    next_yaw = yaw + angular*duration
    if abs(angular) < 1e-10:
        return (x + linear*duration*math.cos(yaw),
                y + linear*duration*math.sin(yaw), next_yaw)
    return (x + linear/angular*(math.sin(next_yaw)-math.sin(yaw)),
            y - linear/angular*(math.cos(next_yaw)-math.cos(yaw)), next_yaw)


def wall_returns(pose, bounds=(-.4, -.65, 1.4, 1.1), bins=720,
                 sensor=(-.033073, 0., 0.)):
    """Raycast a rectangular room; returns are in robot base coordinates."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    origin = np.array((pose[0] + c*sensor[0] - s*sensor[1],
                       pose[1] + s*sensor[0] + c*sensor[1]))
    angle = -math.pi + (np.arange(bins) + .5)*(2.*math.pi/bins)
    direction = np.column_stack((np.cos(angle+pose[2]+sensor[2]),
                                 np.sin(angle+pose[2]+sensor[2])))
    x0, y0, x1, y1 = bounds
    distances = []
    for axis, low, high in ((0, x0, x1), (1, y0, y1)):
        boundary = np.where(direction[:, axis] > 0., high, low)
        distances.append((boundary-origin[axis])/direction[:, axis])
    ranges = np.minimum(distances[0], distances[1])
    return np.column_stack((sensor[0] + ranges*np.cos(angle+sensor[2]),
                            sensor[1] + ranges*np.sin(angle+sensor[2])))


class TunnelInputAndEntryTest(unittest.TestCase):
    def make_mission(self, goal=(.65, 0., 0.), sensors=None):
        config = configuration(goal)
        if sensors is not None:
            config['sensors'].update(sensors)
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        return mission

    def feed(self, mission, now, pose=(0., 0., 0.), velocity=(0., 0.),
             origin=(0., 0., 0.), bounds=(-.4, -.65, 1.4, 1.1)):
        odom = transform(pose, origin)
        mission.update_odometry(odom, *velocity, now)
        mission.update_cloud(wall_returns(pose, bounds), (-.033073, 0., 0.), odom, now)

    def prepare_following(self, mission, now=10.):
        """Unit-only setup; real phase transitions are covered by closed loops."""
        self.feed(mission, now)
        mission.step(now)
        mission.global_path = GlobalRoute(np.array([[0., 0.], [.65, 0.]]))
        mission.state = 'FOLLOWING'
        mission.last_linear = mission.last_angular = 0.
        mission.worker = SimpleNamespace(is_alive=lambda: True)

    def measured_surface(self, isolated=False):
        """Adjacent real hits support a face near a 2 cm cell's far edge."""
        mission = self.make_mission()
        angle = .015
        distance = .3179 / math.cos(angle)
        ranges = [distance] if isolated else [distance, distance]
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        for i in range(mission.config['grid']['mark_observations']):
            mission.update_scan(ranges, -angle, 2*angle, .10, 4.,
                                (0., 0., 0.), (0., 0., 0.), 10.+i*.05)
            if mission.anchor is None:
                mission.step(10.)
        return mission

    def production_static_mission(self):
        """Initialize the actual default map without any dynamic endpoints."""
        mission = TunnelMission(load_config(CONFIG))
        self.addCleanup(mission.close)
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_scan([math.inf], 0., 1., .10, 4.,
                            (0., 0., 0.), (0., 0., 0.), 10.)
        mission.step(10.)
        return mission

    def confirm_point(self, mission, point):
        distance, angle = math.hypot(*point), math.atan2(point[1], point[0])
        stamp = mission.scan[-1]
        for _ in range(mission.config['grid']['mark_observations']):
            stamp += .05
            mission.update_scan([distance], angle, 1., .10, 4.,
                                (0., 0., 0.), (0., 0., 0.), stamp)

    def test_waits_for_both_sensors_and_does_not_initialize_from_missing_input(self):
        mission = self.make_mission()
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)
        mission.update_odometry((1., 2., .3), 0., 0., 10.)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)

    def test_default_goal_uses_source_gazebo_staging_to_exit_transform(self):
        config = load_config(CONFIG)
        staging = (-1.7475895, .140, -math.pi/2.)
        exit_pose = (.200, -1.748373, 0.)
        expected = relative_pose(exit_pose, staging)
        configured = config['goal']['pose']
        np.testing.assert_allclose(
            [configured[0], configured[1], math.radians(configured[2])],
            [expected.x, expected.y, expected.yaw], atol=1e-10)


    def test_default_static_map_aligns_four_walls_and_leaves_entry_and_exit_open(self):
        mission = self.production_static_mission()
        image_path = Path(mission.config['grid']['static_map']['image'])
        self.assertTrue(image_path.is_absolute())
        self.assertTrue(image_path.is_file())
        grid = mission._grid()
        # SDF state/link world centres transformed by the surveyed staging pose.
        walls = ((1.145857, -.1678905), (2.06118, .7528905),
                 (.972071, 1.7262865), (.17074, .9328905))
        for point in walls:
            with self.subTest(wall=point):
                self.assertTrue(occupied(grid, *grid.world_to_grid(*point)))
        for point in ((0., 0.), (.30, 0.), (mission.goal.x, mission.goal.y)):
            with self.subTest(open_space=point):
                self.assertFalse(occupied(grid, *grid.world_to_grid(*point)))

    def test_default_static_map_does_not_bake_in_lidar_cylinders(self):
        mission = self.production_static_mission()
        grid = mission._grid()
        centers = ((1.47874, 1.1293595), (1.55427, .3824095), (.767116, .6066495))
        for center in centers:
            points = [center] + [(center[0]+.1*math.cos(a), center[1]+.1*math.sin(a))
                                 for a in np.linspace(0., 2.*math.pi, 32, endpoint=False)]
            for point in points:
                with self.subTest(cylinder=center, point=point):
                    self.assertFalse(occupied(grid, *grid.world_to_grid(*point)))

    def test_repeated_near_wall_lidar_hits_do_not_duplicate_static_wall(self):
        mission = self.production_static_mission()
        # The previous Gazebo noise tail lay inside this free cell, near Wall_1.
        ghost = (.9, -.115)
        cell = mission.costmap.world_to_cell(*ghost)
        self.assertIsNotNone(cell)
        self.assertEqual(mission.costmap._static_data[cell[1]*mission.costmap.width+cell[0]], 0)
        self.confirm_point(mission, ghost)
        self.assertFalse(mission.costmap._dynamic_occupied[cell[1]*mission.costmap.width+cell[0]])
        wall_cell = mission.costmap.world_to_cell(.9, -.1678905)
        self.assertEqual(mission.costmap._static_data[wall_cell[1]*mission.costmap.width+wall_cell[0]], 100)
        self.assertTrue(occupied(mission._grid(), *wall_cell))

    def test_lidar_cylinder_surface_far_from_static_walls_is_still_occupied(self):
        mission = self.production_static_mission()
        center = np.array((.767116, .6066495))
        point = center * (1. - .1/np.linalg.norm(center))
        cell = mission.costmap.world_to_cell(*point)
        self.assertEqual(mission.costmap._static_data[cell[1]*mission.costmap.width+cell[0]], 0)
        self.confirm_point(mission, point)
        self.assertTrue(mission.costmap._dynamic_occupied[cell[1]*mission.costmap.width+cell[0]])
        self.assertTrue(occupied(mission._grid(), *cell))

    def test_default_clearance_cost_configuration_matches_source_and_reaches_core(self):
        mission = self.production_static_mission()
        grid_values = {'static_occupied_threshold': 65,
                       'dynamic_inflation_radius': .06, 'dynamic_inflation_value': 64}
        for name, expected in grid_values.items():
            with self.subTest(grid_setting=name):
                self.assertEqual(mission.config['grid'][name], expected)
                self.assertEqual(getattr(mission.costmap, name), expected)
        self.assertEqual(mission._grid().occupied_threshold, mission.costmap.static_occupied_threshold)


    def test_anchor_coordinate_conversion_is_rigid_transform_invariant(self):
        origin = (4.3, -8.2, 1.2)
        relative = (.615, .53, math.radians(85.))
        result = relative_pose(transform(relative, origin), origin)
        np.testing.assert_allclose((result.x, result.y, result.yaw), relative, atol=1e-12)
        mission = self.make_mission()
        self.feed(mission, 10., origin=origin)
        self.assertGreater(mission.step(10.)[0], 0.)
        self.assertEqual(mission.anchor, origin)

    def test_out_of_order_or_nonfinite_odometry_and_cloud_are_rejected(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        initial = mission.odom
        self.assertFalse(mission.update_odometry((100., 0., 0.), 0., 0., 9.))
        self.assertFalse(mission.update_odometry((math.nan, 0., 0.), 0., 0., 11.))
        self.assertEqual(mission.odom, initial)
        self.assertFalse(mission.update_cloud([[1., 0.]], (0., 0., 0.), (0., 0., 0.), 9.))
        self.assertFalse(mission.update_cloud([], (0., 0., 0.), (0., 0., 0.), 11.))
        self.assertEqual(mission.scan[-1], 10.)

    def test_sparse_valid_cloud_does_not_block_entry_as_unknown_space(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_cloud([[1., 0.]], (0., 0., 0.), (0., 0., 0.), 10.)
        linear, angular, finished = mission.step(10.)
        self.assertGreater(linear, 0.)
        self.assertAlmostEqual(angular, 0.)
        self.assertFalse(finished)
        self.assertEqual(mission.state, 'ENTRY')
        self.assertIsNone(mission.global_path)
        self.assertEqual(mission.plans, 0)

    def test_native_scan_no_return_beams_and_real_minimum_range_allow_entry(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        self.assertTrue(mission.update_scan(
            np.full(360, np.inf), -math.pi, 2.*math.pi/359., .10, 40.,
            (-.033073, 0., 0.), (0., 0., 0.), 10.))
        self.assertTrue(math.isinf(mission.front_clearance))
        self.assertGreater(mission.step(10.)[0], 0.)
        self.assertEqual(mission.state, 'ENTRY')

    def test_native_scan_front_clearance_accounts_for_lidar_offset(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        values = np.full(720, np.inf)
        values[360] = mission.footprint.front + mission.config['entry']['obstacle_clearance'] + .033073
        self.assertTrue(mission.update_scan(
            values, -math.pi, 2.*math.pi/720., .10, 40.,
            (-.033073, 0., 0.), (0., 0., 0.), 10.))
        self.assertAlmostEqual(mission.front_clearance, mission.config['entry']['obstacle_clearance'])
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')

    def test_pointcloud_missing_directions_remain_nan_not_free_rays(self):
        mission = self.make_mission()
        self.assertTrue(mission.update_cloud([[1., 0.]], (0., 0., 0.), (0., 0., 0.), 10.))
        values = mission.scan[0]
        self.assertEqual(np.count_nonzero(np.isfinite(values)), 1)
        self.assertEqual(np.count_nonzero(np.isnan(values)), len(values)-1)
        self.assertFalse(np.any(np.isposinf(values)))

    def test_native_no_return_ray_clears_previous_dynamic_obstacle(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_scan([.7], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), 10.)
        mission.step(10.)
        cell = mission.costmap.world_to_cell(.7, 0.)
        stamp = 10.
        for _ in range(mission.config['grid']['mark_observations'] - 1):
            self.assertFalse(occupied(mission._grid(), *cell))
            stamp += .05
            mission.update_scan([.7], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp)
        self.assertTrue(occupied(mission._grid(), *cell))
        for _ in range(mission.config['grid']['clear_observations']):
            self.assertTrue(occupied(mission._grid(), *cell))
            stamp += .05
            mission.update_scan([math.inf], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp)
        self.assertFalse(occupied(mission._grid(), *cell))

    def test_single_hit_followed_by_free_ray_leaves_no_ghost_obstacle(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_scan([.7], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), 10.)
        mission.step(10.)
        cell = mission.costmap.world_to_cell(.7, 0.)
        if mission.config['grid']['mark_observations'] > 1:
            self.assertFalse(occupied(mission._grid(), *cell))
        for i in range(mission.config['grid']['clear_observations']):
            stamp = 10.05 + i*.05
            mission.update_scan([math.inf], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp)
        self.assertFalse(occupied(mission._grid(), *cell))
        if mission.config['grid']['mark_observations'] > 1:
            mission.update_scan([.7], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp+.05)
            self.assertFalse(occupied(mission._grid(), *cell))

    def test_supported_lidar_surface_geometry_reaches_mission_grid(self):
        mission = self.measured_surface()
        expected = mission.costmap.refined_dynamic_cells()
        self.assertTrue(expected)
        self.assertAlmostEqual(mission.costmap.dynamic_cell_size, .005)
        with patch('tunnel.mission.OccupancyGrid', wraps=OccupancyGrid) as constructor:
            grid = mission._grid()
        self.assertEqual(constructor.call_args.kwargs['dynamic_cell_rectangles'], expected)
        self.assertAlmostEqual(grid.collision_resolution, .005)
        self.assertEqual(grid.resolution, .02)

    def test_refined_surface_permits_unused_cell_area_but_blocks_actual_face(self):
        mission = self.measured_surface()
        grid = mission._grid()
        coarse = OccupancyGrid(grid.data, grid.resolution, grid.origin_x, grid.origin_y,
                               occupied_threshold=grid.occupied_threshold)
        # Body front + padding reaches x=.308645, below the measured .3179 face.
        empty_side = (.231, 0., 0.)
        touching_face = (.240, 0., 0.)
        self.assertFalse(mission.geometry.pose_is_collision_free(coarse, empty_side))
        self.assertTrue(mission.geometry.pose_is_collision_free(grid, empty_side))
        self.assertFalse(mission.geometry.pose_is_collision_free(grid, touching_face))
        self.assertTrue(mission.geometry.primitive_is_collision_free(grid, (0., 0., 0.), 0., empty_side[0]))
        self.assertFalse(mission.geometry.primitive_is_collision_free(grid, (0., 0., 0.), 0., touching_face[0]))

    def test_isolated_lidar_hit_keeps_full_coarse_collision_cell(self):
        mission = self.measured_surface(isolated=True)
        grid = mission._grid()
        self.assertEqual(mission.costmap.refined_dynamic_cells(), {})
        self.assertAlmostEqual(grid.collision_resolution, grid.resolution)
        self.assertFalse(mission.geometry.pose_is_collision_free(grid, (.231, 0., 0.)))
        self.assertFalse(mission.geometry.primitive_is_collision_free(grid, (0., 0., 0.), 0., .231))

    def test_clearing_refined_surface_does_not_mutate_previous_planning_grid(self):
        mission = self.measured_surface()
        previous = mission._grid()
        previous_data = previous.data.copy()
        touching_face = (.240, 0., 0.)
        stamp = mission.scan[-1]
        for _ in range(mission.config['grid']['clear_observations']):
            stamp += .05
            mission.update_scan([math.inf, math.inf], -.015, .03, .10, 4.,
                                (0., 0., 0.), (0., 0., 0.), stamp)
        updated = mission._grid()
        self.assertIsNot(previous, updated)
        np.testing.assert_array_equal(previous.data, previous_data)
        self.assertFalse(mission.geometry.pose_is_collision_free(previous, touching_face))
        self.assertTrue(mission.geometry.pose_is_collision_free(updated, touching_face))

    def test_native_range_beyond_local_crop_does_not_create_a_fake_endpoint(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_scan([6.], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), 10.)
        mission.step(10.)
        self.assertEqual(mission.scan[0][0], 6.)
        self.assertEqual(mission.scan[3], .10)
        self.assertEqual(mission.scan[4], mission.sensors['range_max'])
        self.assertFalse(np.any(mission._grid()._occupied))

    def test_invalid_or_out_of_order_native_scan_does_not_refresh_input(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        clearance = mission.front_clearance
        arguments = (0., .01, .10, 40., (0., 0., 0.), (0., 0., 0.))
        self.assertFalse(mission.update_scan([math.inf], *arguments, 9.))
        self.assertFalse(mission.update_scan([math.nan, -math.inf, .01, 50.], *arguments, 11.))
        self.assertFalse(mission.update_scan([], *arguments, 11.))
        self.assertEqual(mission.scan[-1], 10.)
        self.assertEqual(mission.front_clearance, clearance)

    def test_future_sensor_timestamp_cannot_authorize_motion(self):
        mission = self.make_mission()
        self.feed(mission, 10.5)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)

    def test_lidar_age_limit_defaults_and_explicit_values(self):
        for settings, expected in (({}, .3), ({'lidar_maximum_age': .4}, .4),
                                   ({'lidar_maximum_age': None}, math.inf)):
            with self.subTest(settings=settings):
                mission = self.make_mission(sensors=settings)
                self.assertEqual(mission.lidar_maximum_age, expected)
                self.assertEqual(mission.sensors['maximum_age'], .3)
        for value in (True, False, '0.3', 0, -.3, math.inf, -math.inf, math.nan):
            with self.subTest(invalid=value):
                config = configuration()
                config['sensors']['lidar_maximum_age'] = value
                with self.assertRaises(ValueError):
                    TunnelMission(config)

    def test_entry_lidar_expiry_can_be_disabled_without_changing_odometry_age(self):
        for age in (.305, .335):
            for settings in ({}, {'lidar_maximum_age': None}):
                with self.subTest(age=age, settings=settings):
                    mission = self.make_mission(sensors=settings)
                    self.feed(mission, 10.)
                    mission.update_odometry((0., 0., 0.), 0., 0., 10.+age)
                    linear, angular, complete = mission.step(10.+age)
                    if settings:
                        self.assertGreater(linear, 0., mission.reason)
                        self.assertEqual(mission.state, 'ENTRY')
                    else:
                        self.assertEqual((linear, angular), (0., 0.))
                        self.assertIn('waiting for fresh', mission.reason)
                    self.assertFalse(complete)

    def test_finite_lidar_override_keeps_its_own_expiry_boundary(self):
        for age in (.335, .405):
            with self.subTest(age=age):
                mission = self.make_mission(sensors={'lidar_maximum_age': .4})
                self.feed(mission, 10.)
                mission.update_odometry((0., 0., 0.), 0., 0., 10.+age)
                linear, _, complete = mission.step(10.+age)
                self.assertEqual(linear > 0., age < .4, mission.reason)
                self.assertFalse(complete)

    def test_disabled_lidar_expiry_does_not_extend_wall_alignment_result_lifetime(self):
        mission = self.make_mission(sensors={'lidar_maximum_age': None})
        self.feed(mission, 10.)
        mission.update_odometry((0., 0., 0.), 0., 0., 10.335)
        # The fit has already completed; only its independent acceptance
        # lifetime is under test, not the geometric wall-fitting algorithm.
        mission.localizer = SimpleNamespace()
        mission.localization_result = (SimpleNamespace(accepted=True),
                                       (0., 0., 0.), 10.)
        self.assertEqual(mission.step(10.335), (0., 0., False))
        self.assertIsNone(mission.anchor)
        self.assertIn('wall alignment input expired', mission.reason)

    def test_clock_refresh_after_waiting_for_lock_rejects_expired_inputs(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.clock = lambda: 10.5
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)

    def test_odometry_expiring_during_calculation_is_not_used_to_drive(self):
        for settings in ({}, {'lidar_maximum_age': None}):
            with self.subTest(settings=settings):
                mission = self.make_mission(sensors=settings)
                self.feed(mission, 10.)
                readings = iter((10., 10.2))
                mission.clock = lambda: next(readings)
                self.assertEqual(mission.step(10.), (0., 0., False))
                self.assertIn('expired during', mission.reason)

    def test_scan_expiring_during_calculation_respects_lidar_age_limit(self):
        for settings in ({}, {'lidar_maximum_age': None}):
            with self.subTest(settings=settings):
                mission = self.make_mission(sensors=settings)
                self.feed(mission, 10.)
                mission.update_odometry((0., 0., 0.), 0., 0., 10.2)
                mission.control['safety_reaction_time'] = .5
                readings = iter((10.2, 10.35))
                mission.clock = lambda: next(readings)
                result = mission.step(10.2)
                if settings:
                    self.assertGreater(result[0], 0., mission.reason)
                    self.assertFalse(result[2])
                else:
                    self.assertEqual(result, (0., 0., False))
                    self.assertIn('expired during', mission.reason)


    def test_entry_does_not_require_grid_based_turning_room_selection(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        with patch.object(mission, '_grid', side_effect=AssertionError('entry grid check')):
            linear, angular, finished = mission.step(10.)
        self.assertGreater(linear, 0.)
        self.assertEqual(angular, 0.)
        self.assertFalse(finished)
        self.assertIsNone(mission.global_path)

    def test_entry_stops_at_configured_forward_distance_then_requests_guidance(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        self.assertGreater(mission.step(10.)[0], 0.)
        self.feed(mission, 10.05, pose=(mission.config['entry']['distance'], 0., 0.), velocity=(.04, 0.))
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')
        with patch.object(mission, '_request_global_plan') as search:
            self.assertEqual(mission.step(10.1), (0., 0., False))
            search.assert_called_once()

    def test_lateral_displacement_alone_does_not_finish_entry_distance(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        self.feed(mission, 10.05, pose=(0., .31, 0.))
        self.assertGreater(mission.step(10.05)[0], 0.)
        self.assertEqual(mission.state, 'ENTRY')

    def test_front_clearance_is_measured_from_body_front_not_base_origin(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        x = mission.footprint.front + mission.config['entry']['obstacle_clearance']
        mission.update_cloud([[x, 0.]], (0., 0., 0.), (0., 0., 0.), 10.)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')
        self.assertEqual(mission.plans, 0)

    def test_side_wall_outside_body_width_does_not_finish_entry(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_cloud([[.2, .3], [.2, -.3]], (0., 0., 0.), (0., 0., 0.), 10.)
        self.assertGreater(mission.step(10.)[0], 0.)
        self.assertEqual(mission.state, 'ENTRY')

    def test_entry_obstacle_width_defaults_to_the_full_padded_body(self):
        mission = self.make_mission()
        expected = mission.footprint.half_width + mission.footprint.padding
        self.assertEqual(mission.entry_obstacle_half_width, expected)
        self.assertEqual(mission.entry_obstacle_right_width, expected)
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        point = [.2, mission.footprint.half_width + .5 * mission.footprint.padding]
        mission.update_cloud([point], (0., 0., 0.), (0., 0., 0.), 10.)
        self.assertLess(mission.front_clearance, mission.config['entry']['obstacle_clearance'])
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')

    def test_entry_narrower_corridor_keeps_both_margin_returns_in_scan_and_costmap(self):
        original = self.make_mission()
        config = configuration()
        config['entry']['obstacle_half_width'] = config['footprint']['half_width']
        narrower = TunnelMission(config)
        self.addCleanup(narrower.close)
        lateral = original.footprint.half_width + .5 * original.footprint.padding
        points = [[.2, lateral], [.2, -lateral]]
        for mission in (original, narrower):
            for i in range(mission.config['grid']['mark_observations']):
                stamp = 10. + i * .05
                mission.update_odometry((0., 0., 0.), 0., 0., stamp)
                mission.update_cloud(points, (0., 0., 0.), (0., 0., 0.), stamp)
                if mission.anchor is None:
                    mission.step(stamp)
        self.assertEqual(original.state, 'PLANNING')
        self.assertEqual(narrower.state, 'ENTRY')
        self.assertTrue(math.isinf(narrower.front_clearance))
        self.assertGreater(narrower.last_linear, 0.)
        np.testing.assert_array_equal(narrower.scan[0], original.scan[0])
        self.assertEqual(np.count_nonzero(np.isfinite(narrower.scan[0])), 2)
        self.assertTrue(any(narrower.costmap._dynamic_occupied))
        np.testing.assert_array_equal(narrower.costmap._dynamic_occupied,
                                      original.costmap._dynamic_occupied)
        np.testing.assert_array_equal(narrower.costmap.to_occupancy_data(),
                                      original.costmap.to_occupancy_data())
        self.assertEqual(narrower.footprint, original.footprint)
        self.assertEqual(narrower.geometry.footprint, original.geometry.footprint)

    def test_entry_narrower_corridor_still_stops_for_body_width_and_front_gap(self):
        config = configuration()
        config['entry']['obstacle_half_width'] = config['footprint']['half_width']
        half_width = config['footprint']['half_width']
        front = config['footprint']['front']
        points = [[.055, .01], [.2, half_width - .003], [.2, -half_width + .003],
                  [front + config['entry']['obstacle_clearance'], 0.]]
        for point in points:
            with self.subTest(point=point):
                mission = TunnelMission(copy.deepcopy(config))
                self.addCleanup(mission.close)
                mission.update_odometry((0., 0., 0.), 0., 0., 10.)
                mission.update_cloud([point], (-.033073, 0., 0.), (0., 0., 0.), 10.)
                self.assertEqual(mission.step(10.), (0., 0., False))
                self.assertEqual(mission.state, 'PLANNING')
                self.assertIn('front clearance', mission.reason)

    def test_entry_obstacle_width_rejects_invalid_or_outside_body_margin_values(self):
        config = configuration()
        lower = config['footprint']['half_width']
        upper = lower + config['footprint']['padding']
        for value in (True, False, np.bool_(True), None, '0.09', float('nan'),
                      float('inf'), -float('inf'), lower - .0001, upper + .0001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidate = copy.deepcopy(config)
                candidate['entry']['obstacle_half_width'] = value
                TunnelMission(candidate)
        for value in (lower, .5 * (lower + upper), upper):
            candidate = copy.deepcopy(config)
            candidate['entry']['obstacle_half_width'] = value
            mission = TunnelMission(candidate)
            self.addCleanup(mission.close)
            self.assertEqual(mission.entry_obstacle_half_width, value)
            self.assertEqual(mission.entry_obstacle_right_width, value)

    def test_entry_right_width_changes_only_right_strip_and_keeps_its_scan_and_map(self):
        config = configuration()
        config['entry']['obstacle_half_width'] = .0903
        for lateral in (-.08755, .08755):
            with self.subTest(lateral=lateral):
                original = TunnelMission(copy.deepcopy(config))
                changed = copy.deepcopy(config)
                changed['entry']['obstacle_right_width'] = .085
                narrower = TunnelMission(changed)
                self.addCleanup(original.close)
                self.addCleanup(narrower.close)
                for mission in (original, narrower):
                    for i in range(mission.config['grid']['mark_observations']):
                        stamp = 10. + i * .05
                        mission.update_odometry((0., 0., 0.), 0., 0., stamp)
                        mission.update_cloud([[.16, lateral]], (-.033073, 0., 0.),
                                             (0., 0., 0.), stamp)
                        if mission.anchor is None:
                            mission.step(stamp)
                self.assertEqual(original.state, 'PLANNING')
                self.assertEqual(narrower.state, 'ENTRY' if lateral < 0. else 'PLANNING')
                if lateral < 0.:
                    self.assertTrue(math.isinf(narrower.front_clearance))
                    self.assertGreater(narrower.last_linear, 0.)
                else:
                    self.assertEqual(narrower.front_clearance, original.front_clearance)
                self.assertEqual(narrower.entry_obstacle_half_width, .0903)
                np.testing.assert_array_equal(narrower.scan[0], original.scan[0])
                self.assertEqual(np.count_nonzero(np.isfinite(narrower.scan[0])), 1)
                self.assertTrue(any(narrower.costmap._dynamic_occupied))
                np.testing.assert_array_equal(narrower.costmap.to_occupancy_data(),
                                              original.costmap.to_occupancy_data())
                self.assertEqual(narrower.footprint, original.footprint)
                self.assertEqual(narrower.geometry.footprint, original.geometry.footprint)

    def test_entry_right_width_keeps_inner_right_and_straight_ahead_stopping(self):
        config = configuration()
        config['entry']['obstacle_half_width'] = .0903
        config['entry']['obstacle_right_width'] = .085
        front = config['footprint']['front']
        for point in ([.16, -.084], [.055, 0.],
                      [front + config['entry']['obstacle_clearance'], 0.]):
            with self.subTest(point=point):
                mission = TunnelMission(copy.deepcopy(config))
                self.addCleanup(mission.close)
                mission.update_odometry((0., 0., 0.), 0., 0., 10.)
                mission.update_cloud([point], (-.033073, 0., 0.), (0., 0., 0.), 10.)
                self.assertEqual(mission.step(10.), (0., 0., False))
                self.assertEqual(mission.state, 'PLANNING')
                self.assertIn('front clearance', mission.reason)

    def test_physical_entry_previews_ahead_left_without_widening_body_or_right_side(self):
        root = CONFIG.parents[2]
        for folder in ('stepbystep', 'final_codes'):
            for lateral in (-.095, -.088, .088, .095):
                with self.subTest(folder=folder, lateral=lateral):
                    config = load_config(root / 'codes' / folder / 'tunnel.real.yaml')
                    # Isolate the straight leg after localization/heading turn.
                    # The ahead-left preview is separate from the unpadded
                    # body model and the unchanged physical right-side width.
                    config.pop('localization', None)
                    config['grid'].pop('static_map', None)
                    mission = TunnelMission(config)
                    mission.entry_phase, mission.entry_drive_yaw = 'DRIVE', 0.
                    self.addCleanup(mission.close)
                    expected = mission.footprint.half_width + mission.footprint.padding
                    self.assertEqual(mission.footprint.padding, 0.)
                    self.assertAlmostEqual(mission.entry_obstacle_half_width, expected)
                    self.assertAlmostEqual(mission.entry_obstacle_right_width, expected)
                    self.assertEqual(mission.entry_obstacle_left_far_width, .30)
                    self.assertEqual(mission.entry_obstacle_check_distance, .80)
                    mission.update_odometry((0., 0., 0.), 0., 0., 10.)
                    mission.update_cloud([[.16, lateral]], (-.033073, 0., 0.),
                                         (0., 0., 0.), 10.)
                    if lateral >= -mission.footprint.half_width:
                        self.assertLess(mission.front_clearance, config['entry']['obstacle_clearance'])
                        self.assertEqual(mission.step(10.), (0., 0., False))
                        self.assertEqual(mission.state, 'PLANNING')
                        self.assertIn('front clearance', mission.reason)
                    else:
                        self.assertEqual(mission.front_clearance, math.inf)
                        self.assertGreater(mission.step(10.)[0], 0.)
                        self.assertEqual(mission.state, 'ENTRY')

    def test_entry_right_width_rejects_non_numeric_nonfinite_or_nonpositive_values(self):
        config = configuration()
        for value in (True, False, np.bool_(True), None, '0.085', 0., -.001,
                      float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidate = copy.deepcopy(config)
                candidate['entry']['obstacle_right_width'] = value
                TunnelMission(candidate)

    def test_return_inside_front_of_body_is_not_ignored(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_cloud([[.055, .01]], (-.033073, 0., 0.), (0., 0., 0.), 10.)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')

    def test_entry_heading_feedback_returns_toward_frozen_odom_direction(self):
        for yaw in (.12, -.12):
            with self.subTest(yaw=yaw):
                mission = self.make_mission()
                self.feed(mission, 10.)
                mission.step(10.)
                self.feed(mission, 10.05, pose=(0., 0., yaw))
                linear, angular, finished = mission.step(10.05)
                self.assertGreater(linear, 0.)
                self.assertLess(yaw*angular, 0.)
                self.assertFalse(finished)


    def test_exit_handoff_waits_for_new_measured_stop_without_lane_frames(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        goal_pose = (.65, 0., 0.)
        self.feed(mission, 10.05, pose=goal_pose)
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_STOP')
        # The rest sample used to request zero is not proof of a later stop.
        self.assertEqual(mission.step(10.06), (0., 0., False))
        for now, velocity in ((10.10, (.02, 0.)), (10.15, (0., -.10))):
            self.feed(mission, now, pose=goal_pose, velocity=velocity)
            self.assertEqual(mission.step(now), (0., 0., False))
            self.assertEqual(mission.state, 'WAIT_STOP')
        self.feed(mission, 10.20, pose=goal_pose)
        self.assertEqual(mission.step(10.20), (0., 0., True))
        self.assertEqual(mission.state, 'COMPLETE')
        self.assertIsNone(mission.lane_stamp)
        self.assertIn('measured stop confirmed', mission.reason)

    def test_exit_stop_does_not_reuse_future_dated_sample(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        self.feed(mission, 10.09, pose=(.65, 0., 0.))
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_STOP')
        self.assertEqual(mission.completion_started, 10.09)
        self.assertEqual(mission.step(10.10), (0., 0., False))
        self.feed(mission, 10.15, pose=(.65, 0., 0.))
        self.assertEqual(mission.step(10.15), (0., 0., True))

    def test_exit_handoff_rejects_stale_sensors_even_with_visible_lanes(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        self.feed(mission, 10.05, pose=(.65, 0., 0.))
        self.assertEqual(mission.step(10.05), (0., 0., False))
        for stamp in (10.10, 10.15, 10.20):
            mission.update_lane(True, stamp)
        mission.update_odometry((.65, 0., 0.), 0., 0., 10.50)
        self.assertEqual(mission.step(10.50), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_STOP')
        self.assertIn('fresh', mission.reason)
        self.feed(mission, 10.55, pose=(.65, 0., 0.))
        self.assertEqual(mission.step(10.55), (0., 0., True))

    def test_stale_input_stops_without_discarding_valid_route(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        self.feed(mission, 10.05)
        self.assertGreater(mission.step(10.05)[0], 0.)
        path = mission.global_path
        mission.update_odometry((0., 0., 0.), 0., 0., 10.5)
        self.assertEqual(mission.step(10.5), (0., 0., False))
        self.assertIs(mission.global_path, path)
        self.assertEqual(mission.plans, 0)
        self.feed(mission, 10.55)
        self.assertGreater(mission.step(10.55)[0], 0.)
        self.assertIs(mission.global_path, path)


    def test_front_obstacle_finishes_entry_for_stopped_planning(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        self.assertGreater(mission.step(10.)[0], 0.)
        self.feed(mission, 10.05, bounds=(-.4, -.65, .20, 1.1))
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')
        self.assertEqual(mission.plans, 0)

    def test_reset_keeps_sensor_input_but_reanchors_and_drops_old_route(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        old_path = mission.global_path
        self.feed(mission, 10.05, pose=(.02, 0., 0.))
        mission.reset()
        self.assertIsNone(mission.global_path)
        self.assertIsNone(mission.anchor)
        self.assertIsNotNone(mission.odom)
        self.assertIsNotNone(mission.scan)
        mission.step(10.05)
        self.assertEqual(mission.anchor, (.02, 0., 0.))
        self.assertIsNone(mission.global_path)
        self.assertIsNotNone(old_path)
        self.assertEqual(mission.state, 'ENTRY')


if __name__ == '__main__':
    unittest.main()
