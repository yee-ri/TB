"""Reject mismatched surveys before refining static collision geometry."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from tunnel.mission import load_config
from tunnel.geometry import Pose2D
from tunnel.static_geometry import refined_static_cells


class SurveyMapTests(unittest.TestCase):
    def setUp(self):
        config = load_config(ROOT / 'codes/stepbystep/tunnel.real.yaml')
        self.static_map = config['grid']['static_map'].copy()
        self.survey = json.loads(Path(self.static_map['geometry']).read_text())
        pixels = cv2.imread(self.static_map['image'], cv2.IMREAD_GRAYSCALE)
        self.data = np.flipud((pixels == 0).astype(np.int8) * 100)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.static_map['geometry'] = str(Path(directory.name) / 'survey.json')

    def refine(self, survey=None, data=None):
        Path(self.static_map['geometry']).write_text(json.dumps(
            self.survey if survey is None else survey))
        return refined_static_cells(self.static_map, .02,
                                    self.data if data is None else data, 65)

    def test_verified_survey_refines_every_wall_without_boundary_slivers(self):
        refined = self.refine()
        self.assertEqual(len(refined), np.count_nonzero(self.data))
        self.assertEqual(len(refined), 334)
        for rectangles in refined.values():
            boxes = np.asarray(rectangles)
            self.assertGreaterEqual(np.min(boxes[:, 2:] - boxes[:, :2]), .002 - 1e-10)

    def test_raster_metadata_and_image_hash_must_match(self):
        for field, bad in [('resolution', .01), ('origin', [-.49, -.4]),
                           ('quarter_turns', 1), ('width', 144),
                           ('height', 129), ('sha256', '0' * 64)]:
            with self.subTest(field=field):
                survey = copy.deepcopy(self.survey)
                survey['raster'][field] = bad
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    self.refine(survey)

    def test_survey_cannot_erase_an_additional_occupied_or_unknown_cell(self):
        for value in (100, -1):
            with self.subTest(value=value):
                data = self.data.copy()
                data[0, 0] = value
                with self.assertRaisesRegex(ValueError, 'all loaded occupied cells'):
                    self.refine(data=data)

    def test_missing_wall_or_mapped_free_wall_is_rejected(self):
        survey = copy.deepcopy(self.survey)
        del survey['wall_rectangles']['right_side_wall']
        with self.assertRaisesRegex(ValueError, 'all loaded occupied cells'):
            self.refine(survey)
        data = self.data.copy()
        data[data != 0] = 0
        with self.assertRaisesRegex(ValueError, 'all loaded occupied cells'):
            self.refine(data=data)

    def test_invalid_or_outside_rectangles_are_rejected(self):
        for rectangle in ([.14, -.15, .14, .15], [.14, float('nan'), .16, .15],
                          [-.6, -.15, .14, .15], [1., 2., 3.]):
            with self.subTest(rectangle=rectangle):
                survey = copy.deepcopy(self.survey)
                survey['wall_rectangles'] = {'invalid': rectangle}
                with self.assertRaises(ValueError):
                    self.refine(survey)

    def test_maps_without_survey_keep_full_cell_collision_geometry(self):
        self.assertEqual(refined_static_cells(None, .02, [0], 65), {})
        self.static_map.pop('geometry')
        self.assertEqual(refined_static_cells(self.static_map, .02, self.data, 65), {})

    def test_planning_crop_reindexes_wall_geometry_without_moving_walls(self):
        from tunnel.mission import TunnelMission
        config = load_config(ROOT / 'codes/stepbystep/tunnel.real.yaml')
        config['grid']['bounds'] = [.2, -.2, 1., .4]
        mission = TunnelMission(config)
        self.addCleanup(mission.close)
        mission._initialize(anchor=(0., 0., 0.))
        grid = mission._grid()
        self.assertLess(grid.width, self.data.shape[1])
        self.assertTrue(mission.static_cell_rectangles)
        self.assertTrue(mission.geometry.pose_is_collision_free(grid, Pose2D(.5, 0., 0.)))
        self.assertFalse(mission.geometry.pose_is_collision_free(grid, Pose2D(.5, -.061, 0.)))
        for rectangles in mission.static_cell_rectangles.values():
            for xmin, ymin, xmax, ymax in rectangles:
                self.assertAlmostEqual(ymin, -.152)
                self.assertAlmostEqual(ymax, -.15)


if __name__ == '__main__':
    unittest.main()
