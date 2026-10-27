"""Stopped entrance registration against measured, fixed tunnel walls.

This is a bounded initial pose measurement, not SLAM.  A single wall, short
objects or competing poses cannot initialize the map.  Inputs are already
filtered LiDAR hits in the acquisition-time base frame; no ROS is imported.
"""

from dataclasses import dataclass
import math

import numpy as np

from .planner import Pose2D


@dataclass(frozen=True)
class AlignmentResult:
    accepted: bool
    pose: object = None
    reason: str = ''
    rms: float = math.inf
    inliers: int = 0
    total: int = 0


class EntranceLocalizer:
    """Match an entrance-facing scan within an explicitly bounded region.

    The bounds describe the robot axle in the fixed map, in metres; yaw is
    within ``yaw_limit_deg`` of map +X.  Extended support from perpendicular
    walls is required.  Returns outside the forward half-plane are ignored,
    since they normally describe the room outside the tunnel.
    """

    def __init__(self, wall_rectangles, search_bounds=(-.35, -.07, .45, .07),
                 yaw_limit_deg=20., minimum_inliers=60,
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
        self.bounds = np.asarray(search_bounds, dtype=float)
        if (self.bounds.shape != (4,) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[2:] <= self.bounds[:2])
                or not 0 < yaw_limit_deg < 90):
            raise ValueError('localization requires finite entrance pose bounds')
        if (not 3 <= minimum_inliers or not 0 < minimum_inlier_fraction <= 1
                or not 0 < maximum_rms < inlier_distance
                or minimum_wall_span <= 0 or minimum_axis_inliers < 3):
            raise ValueError('invalid localization acceptance thresholds')
        self.yaw_limit = math.radians(yaw_limit_deg)
        self.minimum_inliers = int(minimum_inliers)
        # Most forward hits can belong to bottles or the surrounding room.
        # Their count must not replace the independent two-wall extent,
        # residual and competing-pose checks below.  Require 60 actual spatial
        # samples as well as a 20% fraction; no missing ray becomes evidence.
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
        self.cap = max(.045, 1.8*self.inlier_distance)

    def _distances(self, points, nearest=True):
        """Squared finite-segment distances and nearest segment for each hit."""
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
            square = np.minimum(normal_first*normal_first,
                                normal_second*normal_second) + outside*outside
            return np.min(square, axis=-1)
        normal = points[..., self.axes] - self.segments[:, 1]
        tangent = points[..., self.tangents]
        outside = np.maximum(self.segments[:, 2]-tangent,
                             np.maximum(tangent-self.segments[:, 3], 0.))
        square = normal*normal + outside*outside
        indices = np.argmin(square, axis=-1)
        return np.take_along_axis(square, indices[..., None], axis=-1)[..., 0], indices

    @staticmethod
    def _transform(points, pose):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return points @ np.array(((c, s), (-s, c))) + pose[:2]

    def _score(self, points, pose):
        distances, _ = self._distances(self._transform(points, pose))
        return float(np.mean(np.minimum(distances, self.cap*self.cap)))

    def _refine(self, points, pose):
        pose = np.array(pose, dtype=float)
        for _ in range(18):
            transformed = self._transform(points, pose)
            squared, indices = self._distances(transformed)
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

    def _support(self, points, pose):
        mapped = self._transform(points, pose)
        squared, indices = self._distances(mapped)
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

    def estimate(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('localization points must be N by 2 base-frame XY')
        points = points[np.isfinite(points).all(axis=1)]
        points = points[points[:, 0] >= 0.]
        # Spatial thinning prevents a dense bottle or a single close wall
        # from dominating confidence.  Every retained hit is still measured.
        if len(points):
            _, selected = np.unique(np.floor(points/.015).astype(np.int64),
                                    axis=0, return_index=True)
            points = points[np.sort(selected)]
        if len(points) < self.minimum_inliers:
            return AlignmentResult(False, reason='insufficient forward wall points', total=len(points))
        evidence = points
        # Bound computation without depending on LiDAR angular density.
        if len(points) > 500:
            points = points[np.linspace(0, len(points)-1, 500, dtype=int)]
        coarse = (points[np.linspace(0, len(points)-1, 200, dtype=int)]
                  if len(points) > 200 else points)
        xs = np.linspace(self.bounds[0], self.bounds[2],
                         int(math.ceil((self.bounds[2]-self.bounds[0])/.04))+1)
        ys = np.linspace(self.bounds[1], self.bounds[3],
                         int(math.ceil((self.bounds[3]-self.bounds[1])/.035))+1)
        yaws = np.linspace(-self.yaw_limit, self.yaw_limit,
                           int(math.ceil(2*self.yaw_limit/math.radians(2)))+1)
        translations = np.array([(x, y) for x in xs for y in ys])
        seeds = []
        for yaw in yaws:
            rotated = self._transform(coarse, (0., 0., yaw))
            squared = self._distances(rotated[None, :, :]+translations[:, None, :], nearest=False)
            scores = np.mean(np.minimum(squared, self.cap*self.cap), axis=1)
            for index in np.argsort(scores)[:6]:
                seeds.append((float(scores[index]), (*translations[index], yaw)))
        # Preserve geographically distinct hypotheses before refining them.
        candidates = []
        for _, seed in sorted(seeds, key=lambda item: item[0]):
            if any(np.linalg.norm(np.asarray(seed[:2])-other[:2]) < .035
                   and abs(seed[2]-other[2]) < math.radians(3) for other in candidates):
                continue
            candidates.append(np.asarray(seed))
            if len(candidates) == 12:
                break
        refined = []
        for seed in candidates:
            pose = self._refine(points, seed)
            if (np.any(pose[:2] < self.bounds[:2]) or np.any(pose[:2] > self.bounds[2:])
                    or abs(pose[2]) > self.yaw_limit):
                continue
            if any(np.linalg.norm(pose[:2]-other[1][:2]) < .012
                   and abs(pose[2]-other[1][2]) < math.radians(.5) for other in refined):
                continue
            refined.append((self._score(points, pose), pose, self._support(evidence, pose)))
        if not refined:
            return AlignmentResult(False, reason='wall match outside entrance bounds', total=len(evidence))
        refined.sort(key=lambda item: item[0])
        best_score, pose, (accepted, reason, rms, count) = refined[0]
        if accepted:
            for score, other, support in refined[1:]:
                if (support[0] and score <= best_score + self.inlier_distance**2*.08
                        and (np.linalg.norm(pose[:2]-other[:2]) > .03
                             or abs(pose[2]-other[2]) > math.radians(2.))):
                    accepted, reason = False, 'ambiguous wall alignment'
                    break
        measured = Pose2D(*map(float, pose)) if accepted else None
        return AlignmentResult(accepted, measured, reason, rms, count, len(evidence))


def localize(points, settings):
    """Convenience API; reuse EntranceLocalizer for repeated stopped scans."""
    return EntranceLocalizer(**settings).estimate(points)
