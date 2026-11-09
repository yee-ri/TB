"""Correct pose from both measured ends of the known exit opening."""
import math

import numpy as np

from .entrance_localization import _connected_groups
from .geometry import Pose2D, normalize_angle
from .localization import AlignmentResult


class ExitLandmarkLocalizer:
    """Match connected wall ends; an unobserved jamb is never synthesized."""

    def __init__(self, landmarks, wall_directions, cluster_distance=.04,
                 sampling_resolution=.002, minimum_points=6,
                 minimum_wall_span=.04, robust_distance=.006,
                 fit_distance=.012, spatial_bin_size=.02,
                 maximum_width_error=.02):
        self.landmarks = np.asarray(landmarks, dtype=float)
        directions = np.asarray(wall_directions, dtype=float)
        if (self.landmarks.shape != (2, 2) or directions.shape != (2, 2)
                or not np.isfinite(self.landmarks).all()
                or not np.isfinite(directions).all()
                or np.any(np.linalg.norm(directions, axis=1) < 1e-9)
                or np.linalg.norm(self.landmarks[1]-self.landmarks[0]) < 1e-9):
            raise ValueError('exit requires two distinct landmarks and wall directions')
        self.directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        for name, value in [('cluster_distance', cluster_distance),
                            ('sampling_resolution', sampling_resolution),
                            ('minimum_wall_span', minimum_wall_span),
                            ('robust_distance', robust_distance),
                            ('fit_distance', fit_distance),
                            ('spatial_bin_size', spatial_bin_size),
                            ('maximum_width_error', maximum_width_error)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive')
            setattr(self, name, float(value))
        if sampling_resolution >= cluster_distance or robust_distance > fit_distance:
            raise ValueError('invalid exit sampling or line fitting distances')
        if isinstance(minimum_points, bool) or int(minimum_points) != minimum_points or minimum_points < 3:
            raise ValueError('minimum_points must be an integer >= 3')
        self.minimum_points = int(minimum_points)
        self.width = float(np.linalg.norm(self.landmarks[1]-self.landmarks[0]))
        self.diagnostics = {}

    @classmethod
    def from_geometry(cls, geometry, **settings):
        opening = np.asarray(geometry['exit_clear_segment'], dtype=float)
        rectangles = np.asarray(list(geometry['wall_rectangles'].values()), dtype=float)
        if (opening.shape != (2, 2) or not np.isfinite(opening).all()
                or rectangles.ndim != 2 or rectangles.shape[1] != 4
                or not np.isfinite(rectangles).all()
                or np.any(rectangles[:, 2:] <= rectangles[:, :2])):
            raise ValueError('exit geometry requires an opening and positive wall rectangles')
        centre = (rectangles[:, :2].min(axis=0)+rectangles[:, 2:].max(axis=0))/2.
        landmarks, directions, selected = [], [], []
        for end in opening:
            delta = np.maximum(np.maximum(rectangles[:, :2]-end, end-rectangles[:, 2:]), 0.)
            index = int(np.argmin(np.linalg.norm(delta, axis=1)))
            rect = rectangles[index]
            axis = int(np.argmax(rect[2:]-rect[:2]))
            normal_axis = 1-axis
            face = min((rect[normal_axis], rect[normal_axis+2]),
                       key=lambda value: abs(value-centre[normal_axis]))
            low, high = np.zeros(2), np.zeros(2)
            low[normal_axis] = high[normal_axis] = face
            low[axis], high[axis] = rect[axis], rect[axis+2]
            point, other = (low, high) if np.linalg.norm(low-end) < np.linalg.norm(high-end) else (high, low)
            landmarks.append(point)
            directions.append(other-point)
            selected.append(index)
        if selected[0] == selected[1]:
            raise ValueError('exit endpoints must adjoin distinct walls')
        return cls(landmarks, directions, **settings)

    def _fit(self, points, forward, origin, yaw_tolerance):
        """Fit one wall orientation, then project measured rays onto its face."""
        ordered = points[np.argsort(points @ forward, kind='stable')]
        anchors = ordered[np.linspace(0, len(ordered)-1, min(24, len(ordered)), dtype=int)]
        first, last = np.triu_indices(len(anchors), 1)
        delta = anchors[last]-anchors[first]
        length = np.linalg.norm(delta, axis=1)
        valid = length >= self.minimum_wall_span
        tangent = delta[valid]/length[valid, None]
        tangent *= np.where(tangent @ forward >= 0., 1., -1.)[:, None]
        valid_direction = tangent @ forward >= math.cos(yaw_tolerance)
        tangent = tangent[valid_direction]
        if not len(tangent):
            return None
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        offsets = np.sum(anchors[first[valid]][valid_direction]*normal, axis=1)
        residual = np.abs(points[:, 0, None]*normal[:, 0] + points[:, 1, None]*normal[:, 1]-offsets)
        support = np.sum(residual <= self.robust_distance, axis=0)
        best = int(np.argmax(support))
        if support[best] < self.minimum_points:
            return None
        tangent, normal, offset = tangent[best], normal[best], offsets[best]
        for _ in range(8):
            residual = points @ normal-offset
            bins = np.floor((points @ tangent)/self.spatial_bin_size).astype(int)
            bins -= bins.min()
            weight = np.maximum(0., 1.-(residual/self.robust_distance)**2)**2 / np.bincount(bins)[bins]
            if np.count_nonzero(weight) < self.minimum_points:
                return None
            centre = np.sum(points*weight[:, None], axis=0)/weight.sum()
            delta = points-centre
            values, vectors = np.linalg.eigh((delta*weight[:, None]).T @ delta/weight.sum())
            if values[-1]-values[0] < 1e-12:
                return None
            tangent = vectors[:, -1]
            if tangent @ forward < 0.:
                tangent = -tangent
            normal = np.array([-tangent[1], tangent[0]])
            offset = float(centre @ normal)
        residual = points @ normal-offset
        selected = np.flatnonzero(np.abs(residual) <= self.fit_distance)
        selected = selected[np.argsort(points[selected] @ tangent, kind='stable')]
        cuts = np.flatnonzero(np.diff(points[selected] @ tangent) > self.cluster_distance)+1
        runs = [run for run in np.split(selected, cuts)
                if len(run) >= self.minimum_points
                and np.ptp(points[run] @ tangent) >= self.minimum_wall_span]
        if not runs or tangent @ forward < math.cos(yaw_tolerance):
            return None
        run = max(runs, key=lambda group: np.ptp(points[group] @ tangent))
        selected = np.zeros(len(points), dtype=bool)
        selected[run] = True
        inliers = points[selected]
        rays = inliers-origin
        denominator = rays @ normal
        valid = np.abs(denominator) > 1e-9
        scale = (offset-origin @ normal)/denominator[valid]
        intersections = origin+scale[:, None]*rays[valid]
        intersections = intersections[scale > 0.]
        if len(intersections) < self.minimum_points:
            return None
        endpoint = intersections[np.argmin(intersections @ tangent)]
        along = inliers @ tangent
        bins = np.floor(along/self.spatial_bin_size).astype(int)
        bins -= bins.min()
        error = residual[selected]
        weight = np.maximum(0., 1.-(error/self.robust_distance)**2)**2 / np.bincount(bins)[bins]
        total_weight = weight.sum()
        if total_weight <= 0.:
            return None
        along_mean = float(np.sum(weight*along)/total_weight)
        normal_mean = float(np.sum(weight*error)/total_weight)
        normal_variance = float(np.sum(weight*(error-normal_mean)**2)/total_weight)
        # Direction confidence follows observed spread and perpendicular noise.
        # A short, noisy jamb face must not rotate the pose as much as a long wall.
        information = float(np.sum(weight*(along-along_mean)**2)
                            / max(normal_variance, self.sampling_resolution**2/12.))
        return dict(endpoint=endpoint, tangent=tangent, count=len(inliers),
                    rms=float(np.sqrt(np.mean(error**2))), selected=selected,
                    information=information)

    def track(self, points, prior_pose, max_translation=.08,
              max_yaw=math.radians(6), sensor_origin=(0., 0.)):
        prior = np.asarray((prior_pose.x, prior_pose.y, prior_pose.yaw)
                           if isinstance(prior_pose, Pose2D) else prior_pose, dtype=float)
        origin = np.asarray(sensor_origin, dtype=float)
        if (prior.shape != (3,) or not np.isfinite(prior).all()
                or origin.shape != (2,) or not np.isfinite(origin).all()
                or not math.isfinite(max_translation) or max_translation <= 0.
                or not math.isfinite(max_yaw) or not 0. < max_yaw < math.pi/2):
            raise ValueError('exit tracking requires finite pose, origin and positive bounds')
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('exit points must be N by 2 base-frame XY')
        points = points[np.isfinite(points).all(axis=1)]
        self.diagnostics = {'input_points': len(points)}
        if len(points) < 2*self.minimum_points:
            return AlignmentResult(False, reason='both exit wall ends missing', total=len(points))
        _, indices = np.unique(np.floor(points/self.sampling_resolution).astype(np.int64),
                               axis=0, return_index=True)
        samples = points[np.sort(indices)]
        groups = _connected_groups(samples, self.cluster_distance, self.sampling_resolution)
        c, s = math.cos(prior[2]), math.sin(prior[2])
        forward = self.directions @ np.array(((c, -s), (s, c)))
        candidates = [[], []]
        expected = (self.landmarks-prior[:2]) @ np.array(((c, -s), (s, c)))
        bound = (max_translation+2.*np.linalg.norm(self.landmarks-prior[:2], axis=1)
                 * math.sin(max_yaw/2.)+self.maximum_width_error/2.)
        rays = samples-origin
        ray_length = np.linalg.norm(rays, axis=1)
        supported = []
        for endpoint, radius in zip(expected, bound):
            direction = endpoint-origin
            distance = float(np.linalg.norm(direction))
            # Any accepted endpoint lies on a measured positive ray inside this
            # bound. Skip impossible hypotheses without cropping a wall segment.
            if distance <= radius:
                supported.append(np.ones(len(samples), dtype=bool))
            else:
                supported.append(rays @ direction >= ray_length
                                 * math.sqrt(distance*distance-radius*radius)-1e-12)
        for group_id, indices in enumerate(groups):
            if len(indices) < self.minimum_points:
                continue
            for side in (0, 1):
                remaining = samples[indices]
                possible_endpoint = supported[side][indices]
                while len(remaining) >= self.minimum_points and np.any(possible_endpoint):
                    fitted = self._fit(remaining, forward[side], origin, max_yaw)
                    if fitted is None:
                        break
                    remaining_indices = ~fitted.pop('selected')
                    remaining = remaining[remaining_indices]
                    possible_endpoint = possible_endpoint[remaining_indices]
                    fitted['group'] = group_id
                    candidates[side].append(fitted)
        self.diagnostics.update(clusters=len(groups), wall_candidates=[len(c) for c in candidates])
        matches = []
        for left in candidates[0]:
            for right in candidates[1]:
                measured = np.array([left['endpoint'], right['endpoint']])
                width = float(np.linalg.norm(measured[1]-measured[0]))
                if abs(width-self.width) > self.maximum_width_error:
                    continue
                angles = [normalize_angle(math.atan2(direction[1], direction[0])
                                          - math.atan2(line['tangent'][1], line['tangent'][0]))
                          for direction, line in zip(self.directions, (left, right))]
                if abs(normalize_angle(angles[1]-angles[0])) > max_yaw:
                    continue
                information = np.array([left['information'], right['information']])
                yaw = math.atan2(float(information @ np.sin(angles)),
                                 float(information @ np.cos(angles)))
                c, s = math.cos(yaw), math.sin(yaw)
                rotated = measured @ np.array(((c, s), (-s, c)))
                translation = np.mean(self.landmarks-rotated, axis=0)
                pair_error = float(np.linalg.norm((rotated[1]-rotated[0])
                                                   -(self.landmarks[1]-self.landmarks[0])))
                if pair_error > self.maximum_width_error:
                    continue
                shift = float(np.linalg.norm(translation-prior[:2]))
                turn = abs(normalize_angle(yaw-prior[2]))
                if shift > max_translation or turn > max_yaw:
                    continue
                score = shift+self.width*turn+pair_error
                matches.append((score, translation, yaw, measured, width, pair_error, left, right))
        self.diagnostics['matched_pairs'] = len(matches)
        if not matches:
            return AlignmentResult(False, reason='observed exit opening not matched', total=len(points))
        matches.sort(key=lambda match: match[0])
        best = matches[0]
        for alternative in matches[1:]:
            separation = np.linalg.norm(alternative[1]-best[1])
            separation += self.width*abs(normalize_angle(alternative[2]-best[2]))
            if (alternative[0]-best[0] <= self.maximum_width_error
                    and separation > self.maximum_width_error):
                return AlignmentResult(False, reason='ambiguous observed exit opening', total=len(points))
        _, translation, yaw, measured, width, error, left, right = best
        pose = Pose2D(float(translation[0]), float(translation[1]), normalize_angle(yaw))
        count = left['count']+right['count']
        rms = math.sqrt((left['rms']**2*left['count']+right['rms']**2*right['count'])/count)
        self.diagnostics.update(observed_width=width, expected_width=self.width,
                                endpoint_pair_error=error, wall_direction_information=[left['information'], right['information']],
                                endpoints_base=measured.tolist(),
                                endpoints_map=self.landmarks.tolist(), pose=[pose.x, pose.y, pose.yaw])
        return AlignmentResult(True, pose, 'both exit wall ends and opening matched',
                               rms, count, len(points))
