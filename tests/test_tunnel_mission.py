"""ROS-free port integration tests, not Gazebo or physical driving evidence.

The closed-loop fixture supplies ideal, finite LiDAR wall returns and integrates
the commanded differential-drive twist. It does not model wheel slip, sensor
noise, TF delay, camera recognition, or the actual TB course geometry.
"""

import copy
import math
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes/stepbystep"))

from tunnel.mission import TunnelMission, load_config, relative_pose
from tunnel.planner import OccupancyGrid, Pose2D


CONFIG = Path(__file__).resolve().parents[1] / "codes/stepbystep/tunnel.yaml"


def configuration(goal=(.65, 0., 0.)):
    config = copy.deepcopy(load_config(CONFIG))
    config['goal']['pose'] = list(goal)
    # This ideal-room goal is the planner endpoint itself. Connector tests
    # below supply both an explicit internal endpoint and an outside endpoint.
    config['goal'].pop('planning_pose', None)
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


class TunnelMissionPortTest(unittest.TestCase):
    def make_mission(self, goal=(.65, 0., 0.)):
        mission = TunnelMission(configuration(goal))
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
        x = np.linspace(0., .65, 34)
        mission.path = mission._tracking_path(x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x))
        mission.index, mission.path_version = 0, -1
        mission.state = 'FOLLOWING'

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
        internal = relative_pose((-.180, -1.748373, 0.), staging)
        configured_internal = config['goal']['planning_pose']
        np.testing.assert_allclose(
            [configured_internal[0], configured_internal[1], math.radians(configured_internal[2])],
            [internal.x, internal.y, internal.yaw], atol=1e-10)

    def connector_mission(self):
        config = configuration()
        config['goal']['planning_pose'] = [.30, 0., 0.]
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        return mission

    @staticmethod
    def straight_plan(end=.30, y=0., yaw=0.):
        x = np.linspace(0., end, 16)
        return SimpleNamespace(x=x, y=np.full_like(x, y), yaw=np.full_like(x, yaw),
                               curvature=np.zeros_like(x))

    def test_inside_planner_goal_and_exit_connector_form_one_moving_path(self):
        mission = self.connector_mission()
        plan = self.straight_plan()
        with patch.object(mission.planner, 'plan', return_value=plan) as search:
            self.assertEqual(mission.step(10.), (0., 0., False))
            mission.worker.join(timeout=2.)
            self.assertEqual(search.call_args.args[2], mission.planning_goal)
            self.assertAlmostEqual(mission.planning_goal.x, .30)
            self.assertGreater(mission.step(10.05)[0], 0.)
        path = mission.path
        self.assertEqual(mission.state, 'FOLLOWING')
        np.testing.assert_allclose(path.x[:len(plan.x)], plan.x, atol=1e-14)
        np.testing.assert_allclose(path.curvature[:len(plan.x)], plan.curvature, atol=1e-14)
        self.assertAlmostEqual(path.x[-1], mission.goal.x)
        self.assertAlmostEqual(path.y[-1], mission.goal.y)
        self.assertGreater(len(path.x), len(plan.x))
        self.assertTrue(np.all(np.diff(path.station) > 0.))
        self.assertGreater(path.speed[len(plan.x)-1], mission.control['exit_velocity'])

        # Unit fixture advances odom to the seam: it must not declare the
        # internal planning target finished or insert a stopped exit phase.
        self.feed(mission, 10.10, pose=(.30, 0., 0.))
        self.assertGreater(mission.step(10.10)[0], 0.)
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertIs(mission.path, path)
        self.feed(mission, 10.15, pose=(.65, 0., 0.))
        self.assertEqual(mission.step(10.15), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_LANE')
        self.assertEqual(mission.plans, 1)

    def test_blocked_exit_connector_is_rejected_before_committing_the_plan(self):
        mission = self.connector_mission()
        self.confirm_point(mission, (.50, 0.))
        stamp = mission.scan[-1]
        mission.update_odometry((0., 0., 0.), 0., 0., stamp)
        grid = mission._grid()
        plan = self.straight_plan()
        self.assertTrue(mission._route_safe(
            grid, 0, mission._tracking_path(plan.x, plan.y, plan.yaw, plan.curvature)))
        self.assertTrue(mission.planner.pose_is_collision_free(grid, mission.goal))
        with patch.object(mission.planner, 'plan', return_value=plan):
            self.assertEqual(mission.step(stamp), (0., 0., False))
            mission.worker.join(timeout=2.)
            self.assertEqual(mission.step(stamp+.05), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')
        self.assertIsNone(mission.path)
        self.assertIn('connector is not collision-free', mission.reason)

    def test_exit_connector_keeps_source_forward_and_curvature_limits(self):
        mission = self.connector_mission()
        grid = mission._grid()
        for plan, reason in ((self.straight_plan(.70), 'outside pose is not ahead'),
                             (self.straight_plan(-.30, yaw=math.pi), 'not strictly forward'),
                             (self.straight_plan(y=.15), 'minimum turning radius')):
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                mission._planned_path(plan, grid)

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
                self.assertTrue(grid.is_occupied_cell(*grid.world_to_grid(*point)))
        for point in ((0., 0.), (.30, 0.), (mission.goal.x, mission.goal.y)):
            with self.subTest(open_space=point):
                self.assertFalse(grid.is_occupied_cell(*grid.world_to_grid(*point)))

    def test_default_static_map_does_not_bake_in_lidar_cylinders(self):
        mission = self.production_static_mission()
        grid = mission._grid()
        centers = ((1.47874, 1.1293595), (1.55427, .3824095), (.767116, .6066495))
        for center in centers:
            points = [center] + [(center[0]+.1*math.cos(a), center[1]+.1*math.sin(a))
                                 for a in np.linspace(0., 2.*math.pi, 32, endpoint=False)]
            for point in points:
                with self.subTest(cylinder=center, point=point):
                    self.assertFalse(grid.is_occupied_cell(*grid.world_to_grid(*point)))

    def test_repeated_near_wall_lidar_hits_do_not_duplicate_static_wall(self):
        mission = self.production_static_mission()
        # The previous Gazebo noise tail lay inside this free cell, near Wall_1.
        ghost = (.9, -.115)
        cell = mission.costmap.world_to_cell(*ghost)
        self.assertIsNotNone(cell)
        self.assertEqual(mission.costmap.static_data[cell[1]*mission.costmap.width+cell[0]], 0)
        self.confirm_point(mission, ghost)
        self.assertFalse(mission.costmap.is_dynamic_occupied(*cell))
        wall_cell = mission.costmap.world_to_cell(.9, -.1678905)
        self.assertEqual(mission.costmap.static_data[wall_cell[1]*mission.costmap.width+wall_cell[0]], 100)
        self.assertTrue(mission._grid().is_occupied_cell(*wall_cell))

    def test_lidar_cylinder_surface_far_from_static_walls_is_still_occupied(self):
        mission = self.production_static_mission()
        center = np.array((.767116, .6066495))
        point = center * (1. - .1/np.linalg.norm(center))
        cell = mission.costmap.world_to_cell(*point)
        self.assertEqual(mission.costmap.static_data[cell[1]*mission.costmap.width+cell[0]], 0)
        self.confirm_point(mission, point)
        self.assertTrue(mission.costmap.is_dynamic_occupied(*cell))
        self.assertTrue(mission._grid().is_occupied_cell(*cell))

    def test_default_clearance_cost_configuration_matches_source_and_reaches_core(self):
        mission = self.production_static_mission()
        grid_values = {'static_occupied_threshold': 65,
                       'dynamic_inflation_radius': .06, 'dynamic_inflation_value': 64}
        planner_values = {'obstacle_cost_weight': .12, 'obstacle_cost_distance': .30,
                          'soft_obstacle_cost_weight': 1., 'soft_cost_check_step': .020}
        for name, expected in grid_values.items():
            with self.subTest(grid_setting=name):
                self.assertEqual(mission.config['grid'][name], expected)
                self.assertEqual(getattr(mission.costmap, name), expected)
        for name, expected in planner_values.items():
            with self.subTest(planner_setting=name):
                self.assertEqual(mission.config['planner'][name], expected)
                self.assertEqual(getattr(mission.planner, name), expected)
        self.assertEqual(mission._grid().occupied_threshold, mission.costmap.static_occupied_threshold)

    def test_soft_clearance_band_adds_primitive_cost_without_becoming_hard_obstacle(self):
        mission = self.measured_surface()
        grid, planner = mission._grid(), mission.planner
        bands, endpoints = np.argwhere(grid.data == 64), np.argwhere(grid.data == 100)
        self.assertGreater(len(bands), 0)
        self.assertGreater(len(endpoints), 0)
        for row, column in bands:
            self.assertFalse(grid.is_occupied_cell(int(column), int(row)))
        for row, column in endpoints:
            self.assertTrue(grid.is_occupied_cell(int(column), int(row)))
        start, distance = Pose2D(.10, 0., 0.), .131
        self.assertTrue(planner.primitive_is_collision_free(grid, start, 0., distance))
        exposure = planner._primitive_soft_cost_exposure(grid, start, 0., distance)
        self.assertGreater(exposure, 0.)
        end = planner._propagate(start, 0., distance)
        cost = planner._transition_cost(grid, end, 0., 0., distance, exposure)
        without_exposure = planner._transition_cost(grid, end, 0., 0., distance, 0.)
        self.assertGreater(cost, without_exposure)
        self.assertAlmostEqual(cost-without_exposure, exposure*planner.soft_obstacle_cost_weight)
        self.assertFalse(planner.pose_is_collision_free(grid, (.240, 0., 0.)))

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
        self.assertIsNone(mission.path)
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
            self.assertFalse(mission._grid().is_occupied_cell(*cell))
            stamp += .05
            mission.update_scan([.7], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp)
        self.assertTrue(mission._grid().is_occupied_cell(*cell))
        for _ in range(mission.config['grid']['clear_observations']):
            self.assertTrue(mission._grid().is_occupied_cell(*cell))
            stamp += .05
            mission.update_scan([math.inf], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp)
        self.assertFalse(mission._grid().is_occupied_cell(*cell))

    def test_single_hit_followed_by_free_ray_leaves_no_ghost_obstacle(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_scan([.7], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), 10.)
        mission.step(10.)
        cell = mission.costmap.world_to_cell(.7, 0.)
        if mission.config['grid']['mark_observations'] > 1:
            self.assertFalse(mission._grid().is_occupied_cell(*cell))
        for i in range(mission.config['grid']['clear_observations']):
            stamp = 10.05 + i*.05
            mission.update_scan([math.inf], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp)
        self.assertFalse(mission._grid().is_occupied_cell(*cell))
        if mission.config['grid']['mark_observations'] > 1:
            mission.update_scan([.7], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), stamp+.05)
            self.assertFalse(mission._grid().is_occupied_cell(*cell))

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
        self.assertFalse(mission.planner.pose_is_collision_free(coarse, empty_side))
        self.assertTrue(mission.planner.pose_is_collision_free(grid, empty_side))
        self.assertFalse(mission.planner.pose_is_collision_free(grid, touching_face))
        self.assertTrue(mission.planner.primitive_is_collision_free(grid, (0., 0., 0.), 0., empty_side[0]))
        self.assertFalse(mission.planner.primitive_is_collision_free(grid, (0., 0., 0.), 0., touching_face[0]))

    def test_isolated_lidar_hit_keeps_full_coarse_collision_cell(self):
        mission = self.measured_surface(isolated=True)
        grid = mission._grid()
        self.assertEqual(mission.costmap.refined_dynamic_cells(), {})
        self.assertAlmostEqual(grid.collision_resolution, grid.resolution)
        self.assertFalse(mission.planner.pose_is_collision_free(grid, (.231, 0., 0.)))
        self.assertFalse(mission.planner.primitive_is_collision_free(grid, (0., 0., 0.), 0., .231))

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
        self.assertFalse(mission.planner.pose_is_collision_free(previous, touching_face))
        self.assertTrue(mission.planner.pose_is_collision_free(updated, touching_face))

    def test_native_range_beyond_local_crop_does_not_create_a_fake_endpoint(self):
        mission = self.make_mission()
        mission.update_odometry((0., 0., 0.), 0., 0., 10.)
        mission.update_scan([6.], 0., 1., .10, 40., (0., 0., 0.), (0., 0., 0.), 10.)
        mission.step(10.)
        self.assertEqual(mission.scan[0][0], 6.)
        self.assertEqual(mission.scan[3], .10)
        self.assertEqual(mission.scan[4], mission.sensors['range_max'])
        self.assertFalse(np.any(np.asarray(mission.costmap.to_raw_occupancy_data()) >=
                               mission.costmap.static_occupied_threshold))
        self.assertTrue(any(mission.costmap.observed_free_mask))

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

    def test_clock_refresh_after_waiting_for_lock_rejects_expired_inputs(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.clock = lambda: 10.5
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIsNone(mission.anchor)

    def test_odometry_expiring_during_calculation_is_not_used_to_drive(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        readings = iter((10., 10.2))
        mission.clock = lambda: next(readings)
        self.assertEqual(mission.step(10.), (0., 0., False))
        self.assertIn('expired during', mission.reason)

    def test_scan_expiring_during_calculation_is_not_used_to_drive(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.update_odometry((0., 0., 0.), 0., 0., 10.2)
        mission.control['safety_reaction_time'] = .5
        readings = iter((10.2, 10.35))
        mission.clock = lambda: next(readings)
        self.assertEqual(mission.step(10.2), (0., 0., False))
        self.assertIn('expired during', mission.reason)

    def test_current_body_overlap_holds_existing_path_then_resumes_after_measured_clearance(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        path = mission.path
        cloud = np.concatenate((wall_returns((0., 0., 0.)), [[.055, .035]]))
        for i in range(mission.config['grid']['mark_observations']):
            stamp = 10.05 + i*.05
            mission.update_cloud(cloud, (-.033073, 0., 0.), (0., 0., 0.), stamp)
            mission.update_odometry((0., 0., 0.), 0., 0., stamp)
        # A current-body conflict is not evidence that the stored route must be
        # destroyed. It still blocks motion; no occupied cell is suppressed.
        with patch.object(mission, '_route_safe', wraps=mission._route_safe) as route_check:
            self.assertEqual(mission.step(stamp), (0., 0., False))
            route_check.assert_not_called()
        self.assertFalse(mission.planner.pose_is_collision_free(mission.grid, (0., 0., 0.)))
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertIs(mission.path, path)
        self.assertEqual(mission.plans, 0)
        self.assertIn('footprint', mission.reason)

        for _ in range(mission.config['grid']['clear_observations']):
            stamp += .05
            mission.update_scan(np.full(720, math.inf), -math.pi, 2.*math.pi/720.,
                                .05, 4., (-.033073, 0., 0.), (0., 0., 0.), stamp)
            mission.update_odometry((0., 0., 0.), 0., 0., stamp)
        self.assertTrue(mission.planner.pose_is_collision_free(mission._grid(), (0., 0., 0.)))
        self.assertGreater(mission.step(stamp)[0], 0.)
        self.assertEqual(mission.state, 'FOLLOWING')
        self.assertIs(mission.path, path)
        self.assertEqual(mission.plans, 0)

    def test_real_future_route_obstruction_still_discards_route_for_stopped_replanning(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        cloud = np.concatenate((wall_returns((0., 0., 0.)), [[.35, 0.]]))
        for i in range(mission.config['grid']['mark_observations']):
            stamp = 10.05 + i*.05
            mission.update_cloud(cloud, (-.033073, 0., 0.), (0., 0., 0.), stamp)
            mission.update_odometry((0., 0., 0.), 0., 0., stamp)
        self.assertTrue(mission.planner.pose_is_collision_free(mission._grid(), (0., 0., 0.)))
        self.assertFalse(mission._route_safe(mission.grid, 0))
        self.assertEqual(mission.step(stamp), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')
        self.assertIsNone(mission.path)
        self.assertEqual(mission.plans, 0)
        self.assertIn('route intersects', mission.reason)

    def test_entry_does_not_require_grid_based_turning_room_selection(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        with patch.object(mission, '_grid', side_effect=AssertionError('entry grid check')):
            linear, angular, finished = mission.step(10.)
        self.assertGreater(linear, 0.)
        self.assertEqual(angular, 0.)
        self.assertFalse(finished)
        self.assertIsNone(mission.path)

    def test_entry_stops_at_configured_forward_distance_then_waits_for_measured_rest(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        self.assertGreater(mission.step(10.)[0], 0.)
        self.feed(mission, 10.05, pose=(mission.config['entry']['distance'], 0., 0.), velocity=(.04, 0.))
        self.assertEqual(mission.step(10.05), (0., 0., False))
        self.assertEqual(mission.state, 'PLANNING')
        with patch.object(mission.planner, 'plan') as search:
            self.assertEqual(mission.step(10.1), (0., 0., False))
            search.assert_not_called()

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

    def test_transient_planner_failure_retries_latest_grid_then_follows_valid_path(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        x = np.linspace(0., .65, 34)
        result = SimpleNamespace(x=x, y=np.zeros_like(x), yaw=np.zeros_like(x),
                                 curvature=np.zeros_like(x))
        period = mission.control['plan_retry_period']
        with patch.object(mission.planner, 'plan', side_effect=(None, result)) as search:
            self.assertEqual(mission.step(10.), (0., 0., False))
            mission.worker.join(timeout=2.)
            self.assertEqual(mission.step(10.05), (0., 0., False))
            self.assertEqual(mission.state, 'PLANNING')
            self.assertIsNone(mission.path)

            self.feed(mission, 10. + period*.5)
            self.assertEqual(mission.step(10. + period*.5), (0., 0., False))
            self.assertEqual(search.call_count, 1)
            # A stale scan still cannot authorize a retry merely because the
            # timer elapsed. The actual retry consumes the new immutable grid.
            retry_at = 10. + period + .1
            self.assertEqual(mission.step(retry_at), (0., 0., False))
            self.assertEqual(search.call_count, 1)
            self.feed(mission, retry_at)
            self.assertEqual(mission.step(retry_at), (0., 0., False))
            mission.worker.join(timeout=2.)
            self.assertEqual(search.call_count, 2)
            self.assertIs(search.call_args_list[1].args[0], mission._grid())
            self.assertGreater(mission.step(retry_at+.05)[0], 0.)
            self.assertEqual(mission.state, 'FOLLOWING')
            self.assertIsNotNone(mission.path)

    def test_repeated_planning_failure_is_rate_limited_without_permanent_failure(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        period = mission.control['plan_retry_period']
        with patch.object(mission.planner, 'plan', return_value=None) as search:
            for attempt in range(3):
                now = 10. + attempt*(period+.1)
                self.feed(mission, now)
                self.assertEqual(mission.step(now), (0., 0., False))
                mission.worker.join(timeout=2.)
                self.assertEqual(search.call_count, attempt+1)
                self.assertEqual(mission.step(now+.01), (0., 0., False))
                for offset in (.02, .04, .08):
                    self.assertEqual(mission.step(now+offset), (0., 0., False))
                self.assertEqual(search.call_count, attempt+1)
                self.assertEqual(mission.state, 'PLANNING')
                self.assertIsNone(mission.path)

    def test_planning_waits_for_measured_rest_not_only_zero_last_command(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        self.feed(mission, 10.05, velocity=(.05, .1))
        with patch.object(mission.planner, 'plan') as search:
            self.assertEqual(mission.step(10.05), (0., 0., False))
            self.assertIn('measured rest', mission.reason)
            search.assert_not_called()

    def test_reset_discards_result_from_superseded_planning_worker(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.path = 'PLANNING', None
        started, release = threading.Event(), threading.Event()

        def slow_search(*args):
            started.set()
            release.wait(timeout=2.)
            return None

        with patch.object(mission.planner, 'plan', side_effect=slow_search):
            mission.step(10.05)
            self.assertTrue(started.wait(timeout=1.))
            try:
                mission.reset()
            finally:
                release.set()
                mission.worker.join(timeout=2.)
            self.assertIsNone(mission.plan_result)
            self.assertEqual(mission.state, 'ENTRY')

    def test_planning_result_from_shifted_start_is_not_accepted(self):
        for shift in ((.02, 0., 0.), (0., 0., math.radians(4.))):
            with self.subTest(shift=shift):
                mission = self.make_mission()
                self.feed(mission, 10.)
                mission.step(10.)
                mission.state, mission.path = 'PLANNING', None
                started, release = threading.Event(), threading.Event()
                x = np.linspace(0., .65, 34)
                result = SimpleNamespace(x=x, y=np.zeros_like(x), yaw=np.zeros_like(x),
                                         curvature=np.zeros_like(x))

                def slow_search(*args):
                    started.set()
                    release.wait(timeout=2.)
                    return result

                with patch.object(mission.planner, 'plan', side_effect=slow_search):
                    mission.step(10.)
                    self.assertTrue(started.wait(timeout=1.))
                    try:
                        self.feed(mission, 10.05, pose=shift)
                    finally:
                        release.set()
                        mission.worker.join(timeout=2.)
                    self.assertEqual(mission.step(10.10), (0., 0., False))
                    self.assertEqual(mission.state, 'PLANNING')
                    self.assertIsNone(mission.path)
                    self.assertEqual(mission.plans, 1)

    def test_lane_handoff_requires_new_consecutive_visible_frames_at_goal(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.completion_started, mission.lane_count = 'WAIT_LANE', 10., 0
        self.assertEqual(mission.step(10.), (0., 0., False))
        mission.update_lane(True, 10.05)
        mission.update_lane(False, 10.10)
        mission.update_lane(True, 10.15)
        mission.update_lane(True, 10.20)
        self.assertEqual(mission.step(10.20), (0., 0., False))
        mission.update_lane(True, 10.25)
        self.assertEqual(mission.step(10.25), (0., 0., True))
        self.assertEqual(mission.state, 'COMPLETE')

    def test_stale_lane_confirmation_does_not_hand_off(self):
        mission = self.make_mission()
        self.feed(mission, 10.)
        mission.step(10.)
        mission.state, mission.completion_started, mission.lane_count = 'WAIT_LANE', 10., 0
        for stamp in (10.05, 10.1, 10.15):
            mission.update_lane(True, stamp)
        self.feed(mission, 10.5)
        self.assertEqual(mission.step(10.5), (0., 0., False))
        self.assertEqual(mission.state, 'WAIT_LANE')

    def test_stale_input_stops_without_discarding_valid_route(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        self.assertGreater(mission.step(10.)[0], 0.)
        path = mission.path
        mission.update_odometry((0., 0., 0.), 0., 0., 10.5)
        self.assertEqual(mission.step(10.5), (0., 0., False))
        self.assertIs(mission.path, path)
        self.assertEqual(mission.plans, 0)
        self.feed(mission, 10.55)
        self.assertGreater(mission.step(10.55)[0], 0.)
        self.assertIs(mission.path, path)

    def test_harmless_new_scan_keeps_route_and_positive_command(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        self.assertGreater(mission.step(10.)[0], 0.)
        path, version = mission.path, mission.version
        for tick in range(1, 4):
            now = 10. + tick*.05
            self.feed(mission, now)
            self.assertGreater(mission.step(now)[0], 0.)
            self.assertIs(mission.path, path)
        self.assertGreaterEqual(mission.version, version)
        self.assertEqual(mission.plans, 0)

    def test_unchanged_scan_reuses_grid_and_does_not_revalidate_the_whole_route(self):
        mission = self.make_mission()
        self.prepare_following(mission)
        # Settle hit confirmation / first-observed cells before measuring reuse.
        for i in range(1, 5):
            self.feed(mission, 10.+i*.05)
            self.assertGreater(mission.step(10.+i*.05)[0], 0.)
        path, grid, version = mission.path, mission._grid(), mission.version
        self.assertEqual(mission.path_version, version)
        self.feed(mission, 10.25)
        self.assertEqual(mission.version, version)
        self.assertIs(mission._grid(), grid)
        with patch.object(mission, '_route_safe', wraps=mission._route_safe) as route_check:
            self.assertGreater(mission.step(10.25)[0], 0.)
            route_check.assert_not_called()
        self.assertIs(mission.path, path)
        self.assertEqual(mission.plans, 0)

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
        old_path = mission.path
        self.feed(mission, 10.05, pose=(.02, 0., 0.))
        mission.reset()
        self.assertIsNone(mission.path)
        self.assertIsNone(mission.anchor)
        self.assertIsNotNone(mission.odom)
        self.assertIsNotNone(mission.scan)
        mission.step(10.05)
        self.assertEqual(mission.anchor, (.02, 0., 0.))
        self.assertIsNone(mission.path)
        self.assertIsNotNone(old_path)
        self.assertEqual(mission.state, 'ENTRY')

    def run_ideal_mission(self, goal, origin=(0., 0., 0.)):
        mission = self.make_mission(goal)
        pose, command, now = (0., 0., 0.), (0., 0.), 10.
        states = []
        lane_frames = 0
        # CPU guard only: graded/refined map rebuilds may run below real time.
        # Keep the 1000-tick mission bound and all completion assertions below.
        deadline = time.monotonic() + 90.
        for _ in range(1000):
            self.assertLess(time.monotonic(), deadline, mission.reason)
            self.feed(mission, now, pose, command, origin)
            if mission.state == 'WAIT_LANE':
                if lane_frames == 0:
                    self.assertFalse(mission.step(now)[2])
                mission.update_lane(True, now)
                lane_frames += 1
            linear, angular, finished = mission.step(now)
            if not states or states[-1] != mission.state:
                states.append(mission.state)
            self.assertLessEqual(abs(linear), mission.control['cruise_velocity'] + 1e-9)
            self.assertLessEqual(abs(angular), mission.control['maximum_angular_velocity'] + 1e-9)
            self.assertNotEqual(mission.state, 'FAILED', mission.reason)
            if mission.worker is not None and mission.worker.is_alive():
                mission.worker.join(timeout=5.)
            if finished:
                self.assertEqual((linear, angular), (0., 0.))
                break
            command = (linear, angular)
            pose = integrate(pose, linear, angular, .05)
            now += .05
        self.assertEqual(mission.state, 'COMPLETE', (mission.state, mission.reason, pose))
        self.assertEqual(states, ['ENTRY', 'PLANNING', 'FOLLOWING', 'WAIT_LANE', 'COMPLETE'])
        self.assertEqual(mission.plans, 1)
        self.assertGreaterEqual(lane_frames, mission.sensors['lane_confirmation_frames'])
        self.assertLessEqual(math.hypot(pose[0]-goal[0], pose[1]-goal[1]),
                             mission.config['goal']['position_tolerance'])

    def test_ideal_lidar_and_differential_drive_complete_straight_mission(self):
        self.run_ideal_mission((.65, 0., 0.), origin=(2.2, -3.1, -.8))

    def test_ideal_lidar_and_differential_drive_complete_turning_mission(self):
        self.run_ideal_mission((.615, .53, 85.))


if __name__ == '__main__':
    unittest.main()
