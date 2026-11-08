"""Correct an existing pose against measured, fixed tunnel walls.

Inputs are filtered LiDAR hits in the acquisition-time base frame.
"""

from dataclasses import dataclass
import math

import numpy as np

from .geometry import Pose2D, normalize_angle


@dataclass(frozen=True)
class AlignmentResult:
    accepted: bool
    pose: object = None
    reason: str = ''
    rms: float = math.inf
    inliers: int = 0
    total: int = 0


class WallLocalizer:
    """Match nearby poses using extended support from perpendicular walls."""

    def __init__(self, wall_rectangles, minimum_inliers=60,
                 minimum_inlier_fraction=.20, maximum_rms=.012,
                 inlier_distance=.025, minimum_wall_span=.25,
                 minimum_axis_inliers=8):
        rectangles = np.asarray(list(wall_rectangles.values())
                                if isinstance(wall_rectangles, dict)
                                else wall_rectangles, dtype=float)
        if (rectangles.ndim != 2 or rectangles.shape[1] != 4
                or not np.isfinite(rectangles).all()
                or np.any(rectangles[:, 2:] <= rectangles[:, :2])):
            raise ValueError('localization walls must be finite positive rectangles')
        if (not 3 <= minimum_inliers or not 0 < minimum_inlier_fraction <= 1
                or not 0 < maximum_rms < inlier_distance
                or minimum_wall_span <= 0 or minimum_axis_inliers < 3):
            raise ValueError('invalid localization acceptance thresholds')
        self.minimum_inliers = int(minimum_inliers)
        # Inlier count and fraction complement the independent wall-extent checks.
        self.minimum_fraction = float(minimum_inlier_fraction)
        self.maximum_rms = float(maximum_rms)
        self.inlier_distance = float(inlier_distance)
        self.minimum_span = float(minimum_wall_span)
        self.minimum_axis = int(minimum_axis_inliers)
        segments = []
        for x0, y0, x1, y1 in rectangles:
            if y1-y0 >= minimum_wall_span:
                segments += [(0, x0, y0, y1), (0, x1, y0, y1)]
            if x1-x0 >= minimum_wall_span:
                segments += [(1, y0, x0, x1), (1, y1, x0, x1)]
        if {s[0] for s in segments} != {0, 1}:
            raise ValueError('localization needs perpendicular extended wall faces')
        self.segments = np.asarray(segments)
        self.axes = self.segments[:, 0].astype(int)
        self.tangents = 1-self.axes
        self.face_signs = np.tile((-1., 1.), len(self.segments)//2)
        self.cap = max(.045, 1.8*self.inlier_distance)

    def _distances(self, points, nearest=True, origins=None):
        """Squared finite-segment distances and nearest segment for each hit."""
        if origins is not None:
            # A return cannot come from the opposite face through a solid wall.
            # Origins and points share the leading candidate-pose dimensions.
            origin_axes = np.asarray(origins)[..., None, :][..., self.axes]
            visible = self.face_signs*(origin_axes-self.segments[:, 1]) >= 0.
        else:
            visible = None
        if not nearest:
            # Each rectangle contributes consecutive parallel faces with the
            # same finite extent.  Coarse search needs only their minimum:
            # share the endpoint-distance calculation between the two faces
            # instead of allocating every face for every pose and hit.
            first, second = self.segments[::2], self.segments[1::2]
            axes = self.axes[::2]
            normal = points[..., axes]
            normal_first = normal-first[:, 1]
            normal_second = normal-second[:, 1]
            tangent = points[..., 1-axes]
            outside = np.maximum(first[:, 2]-tangent,
                                 np.maximum(tangent-first[:, 3], 0.))
            first_square = normal_first*normal_first
            second_square = normal_second*normal_second
            if visible is not None:
                first_square = np.where(visible[..., ::2], first_square, np.inf)
                second_square = np.where(visible[..., 1::2], second_square, np.inf)
            square = np.minimum(first_square, second_square) + outside*outside
            return np.min(square, axis=-1)
        normal = points[..., self.axes] - self.segments[:, 1]
        tangent = points[..., self.tangents]
        outside = np.maximum(self.segments[:, 2]-tangent,
                             np.maximum(tangent-self.segments[:, 3], 0.))
        square = normal*normal + outside*outside
        if visible is not None:
            square = np.where(visible, square, np.inf)
        indices = np.argmin(square, axis=-1)
        return np.take_along_axis(square, indices[..., None], axis=-1)[..., 0], indices

    @staticmethod
    def _transform(points, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return points @ np.array(((c, s), (-s, c))) + pose[:2]

    def _score(self, points, pose, sensor_origin=(0., 0.)):
        origin = self._transform(np.asarray(sensor_origin), pose)
        distances, _ = self._distances(self._transform(points, pose), origins=origin)
        return float(np.mean(np.minimum(distances, self.cap*self.cap)))

    def _refine(self, points, pose, maximum_iterations=18, sensor_origin=(0., 0.)):
        pose = np.array(pose, dtype=float)
        for _ in range(maximum_iterations):
            transformed = self._transform(points, pose)
            origin = self._transform(np.asarray(sensor_origin), pose)
            squared, indices = self._distances(transformed, origins=origin)
            selected = squared < self.cap*self.cap
            if np.count_nonzero(selected) < self.minimum_inliers:
                break
            active = indices[selected]
            axes = self.axes[active]
            mapped = transformed[selected]
            # Only interior projections constrain a wall normal.  Endpoints
            # must not masquerade as a second perpendicular wall.
            tangent = mapped[np.arange(len(mapped)), 1-axes]
            inside = ((tangent >= self.segments[active, 2])
                      & (tangent <= self.segments[active, 3]))
            mapped, axes, active = mapped[inside], axes[inside], active[inside]
            if len(mapped) < self.minimum_inliers:
                break
            residual = mapped[np.arange(len(mapped)), axes]-self.segments[active, 1]
            rotated = mapped-pose[:2]
            jacobian = np.column_stack((axes == 0, axes == 1,
                                        np.where(axes == 0, -rotated[:, 1], rotated[:, 0])))
            weights = np.sqrt(np.minimum(1., .012/np.maximum(np.abs(residual), 1e-12)))
            delta, _, rank, _ = np.linalg.lstsq(jacobian*weights[:, None],
                                               -residual*weights, rcond=None)
            if rank < 3:
                break
            delta[:2] = np.clip(delta[:2], -.05, .05)
            delta[2] = np.clip(delta[2], -math.radians(3), math.radians(3))
            pose += delta
            if np.linalg.norm(delta[:2]) < 1e-5 and abs(delta[2]) < 1e-5:
                break
        return pose

    def _support(self, points, pose, sensor_origin=(0., 0.)):
        mapped = self._transform(points, pose)
        origin = self._transform(np.asarray(sensor_origin), pose)
        squared, indices = self._distances(mapped, origins=origin)
        inliers = squared < self.inlier_distance**2
        count = int(np.count_nonzero(inliers))
        rms = float(np.sqrt(np.mean(squared[inliers]))) if count else math.inf
        if count < self.minimum_inliers or count < self.minimum_fraction*len(points):
            return False, 'insufficient wall inliers', rms, count
        if rms > self.maximum_rms:
            return False, 'wall residual too large', rms, count
        for axis in (0, 1):
            best = False
            for i in np.flatnonzero(self.axes == axis):
                # Both faces of the same thin wall contribute to its span.
                same_plane = ((self.axes == axis)
                              & (np.abs(self.segments[:, 1]-self.segments[i, 1]) <= .006))
                mask = inliers & same_plane[indices]
                tangent = mapped[mask, 1-axis]
                if len(tangent) < self.minimum_axis:
                    continue
                # Separated bottle arcs can form a convincing straight row.
                # Require a continuously sampled wall run, rather than using
                # the distance between disconnected clusters as its span.
                ordered = np.sort(tangent)
                cuts = np.flatnonzero(np.diff(ordered) > .10)+1
                for run in np.split(ordered, cuts):
                    if len(run) < self.minimum_axis:
                        continue
                    lo, hi = np.quantile(run, [.05, .95])
                    covered = len(np.unique(np.floor(run/.05)))
                    if hi-lo >= self.minimum_span and covered >= 6:
                        best = True
                        break
                if best:
                    break
            if not best:
                return False, 'no extended perpendicular wall support', rms, count
        return True, 'wall alignment accepted', rms, count

    def track(self, points, prior_pose, max_translation=.08,
              max_yaw=math.radians(6), sensor_origin=(0., 0.)):
        """Measure a small correction to an acquisition-time map pose.

        Translation is bounded in metres and ``max_yaw`` in radians. Up to
        eight starts and 500 spatial samples bound the fitting work. All scan
        directions are used; ``sensor_origin`` is LiDAR XY in the base frame.
        The caller decides how to apply the returned absolute map pose.
        """
        prior = np.asarray((prior_pose.x, prior_pose.y, prior_pose.yaw)
                           if isinstance(prior_pose, Pose2D) else prior_pose,
                           dtype=float)
        if (prior.shape != (3,) or not np.isfinite(prior).all()
                or not math.isfinite(max_translation) or max_translation <= 0.
                or not math.isfinite(max_yaw) or not 0. < max_yaw < math.pi/2):
            raise ValueError('tracking requires a finite prior and positive local bounds')
        prior = prior.copy()
        prior[2] = normalize_angle(prior[2])
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('localization points must be N by 2 base-frame XY')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            _, selected = np.unique(np.floor(points/.015).astype(np.int64),
                                    axis=0, return_index=True)
            points = points[np.sort(selected)]
        if len(points) > 500:
            points = points[np.linspace(0, len(points)-1, 500, dtype=int)]
        if len(points) < self.minimum_inliers:
            return AlignmentResult(False, reason='insufficient tracking wall points',
                                   total=len(points))
        coarse = (points[np.linspace(0, len(points)-1, 200, dtype=int)]
                  if len(points) > 200 else points)
        offsets = np.linspace(-max_translation, max_translation, 5)
        translations = np.array([(x, y) for x in offsets for y in offsets
                                 if math.hypot(x, y) <= max_translation+1e-12])
        translations += prior[:2]
        seeds = []
        for yaw in prior[2]+np.linspace(-max_yaw, max_yaw, 7):
            rotated = self._transform(coarse, (0., 0., yaw))
            origin = self._transform(np.asarray(sensor_origin), (0., 0., yaw))
            squared = self._distances(rotated[None, :, :]+translations[:, None, :],
                                      nearest=False, origins=translations+origin)
            scores = np.mean(np.minimum(squared, self.cap*self.cap), axis=1)
            for index in np.argsort(scores)[:6]:
                seeds.append((float(scores[index]), (*translations[index], yaw)))
        candidates = []
        for _, seed in sorted(seeds, key=lambda item: item[0]):
            seed = np.asarray(seed)
            if any(np.linalg.norm(seed[:2]-other[:2]) < .025
                   and abs(seed[2]-other[2]) < math.radians(1.5)
                   for other in candidates):
                continue
            candidates.append(seed)
            if len(candidates) == 8:
                break
        refined = []
        for seed in candidates:
            pose = self._refine(points, seed, maximum_iterations=10,
                                sensor_origin=sensor_origin)
            if any(np.linalg.norm(pose[:2]-other[1][:2]) < .008
                   and abs(normalize_angle(pose[2]-other[1][2])) < math.radians(.5)
                   for other in refined):
                continue
            # Keep out-of-bounds solutions for ranking: choosing a poorer
            # in-bounds hypothesis after the best fit jumped would mask drift.
            refined.append((self._score(points, pose, sensor_origin), pose,
                            self._support(points, pose, sensor_origin)))
        refined.sort(key=lambda item: item[0])
        best_score, pose, (accepted, reason, rms, count) = refined[0]
        if (np.linalg.norm(pose[:2]-prior[:2]) > max_translation+1e-9
                or abs(normalize_angle(pose[2]-prior[2])) > max_yaw+1e-9):
            accepted, reason = False, 'wall correction outside tracking bounds'
        if accepted:
            for score, other, support in refined[1:]:
                if (support[0] and score <= best_score + self.inlier_distance**2*.08
                        and (np.linalg.norm(pose[:2]-other[:2]) > .025
                             or abs(normalize_angle(pose[2]-other[2])) > math.radians(2.))):
                    accepted, reason = False, 'ambiguous tracking wall alignment'
                    break
        measured = (Pose2D(float(pose[0]), float(pose[1]), normalize_angle(pose[2]))
                    if accepted else None)
        return AlignmentResult(accepted, measured, reason, rms, count, len(points))
