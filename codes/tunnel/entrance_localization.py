"""Initial pose from a connected right-wall segment and its visible entrance end."""
import math
import numpy as np

from .geometry import Pose2D
from .localization import AlignmentResult


def _connected_groups(points, distance, resolution):
    """Cluster measured XY; coarse search cells do not restrict map position."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    cells = np.floor(points[order] / resolution).astype(np.int64)
    _, first, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    samples = points[order[first]]
    keys, buckets = np.unique(np.floor(samples / (distance / math.sqrt(2))).astype(np.int64),
                              axis=0, return_inverse=True)
    boundaries = np.cumsum(np.bincount(buckets))[:-1]
    groups = np.split(samples[np.argsort(buckets, kind='stable')], boundaries)
    lower = np.array([group.min(axis=0) for group in groups])
    upper = np.array([group.max(axis=0) for group in groups])
    first_xy = [tuple(group[0]) for group in groups]
    lower = lower.tolist()
    upper = upper.tolist()
    keys = [tuple(key) for key in keys.tolist()]
    lookup = {key: i for i, key in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    offsets = [(0, 1), (0, 2)] + [(x, y) for x in (1, 2) for y in range(-2, 3)]
    radius2 = distance * distance
    for i, key in enumerate(keys):
        for dx, dy in offsets:
            j = lookup.get((key[0] + dx, key[1] + dy))
            if j is None or find(i) == find(j):
                continue
            dx0 = first_xy[i][0]-first_xy[j][0]
            dy0 = first_xy[i][1]-first_xy[j][1]
            if dx0*dx0 + dy0*dy0 <= radius2:
                parent[find(j)] = find(i)
                continue
            gx = max(0., lower[i][0]-upper[j][0], lower[j][0]-upper[i][0])
            gy = max(0., lower[i][1]-upper[j][1], lower[j][1]-upper[i][1])
            if gx*gx+gy*gy > radius2:
                continue
            a, b = groups[i], groups[j]
            delta = a[:, None, :] - b[None, :, :]
            if np.any(np.einsum('ijk,ijk->ij',delta,delta) <= radius2):
                parent[find(j)] = find(i)
    roots = np.array([find(i) for i in range(len(keys))])
    labels = roots[buckets][inverse]
    ordered = np.argsort(labels, kind='stable')
    boundaries = np.flatnonzero(np.diff(labels[ordered]))+1
    return np.split(order[ordered], boundaries)


class EntranceLandmarkLocalizer:
    """Locate the nearby right-wall end; use the opposite wall for heading.

    The prior ranks alternative interpretations, never substitutes for a
    measurement. The endpoint distance limit does not crop a connected wall.
    """

    def __init__(self, prior_pose=(.213, -.022, 0.), side_start_x=.14,
                 side_y=-.15, cluster_distance=.02, sampling_resolution=.002,
                 maximum_direction_error_deg=45., fit_distance=.025,
                 minimum_points=6, spatial_bin_size=.02, robust_distance=.006,
                 opposite_side_y=None, maximum_corner_distance=.5,
                 minimum_heading_span=.25, maximum_heading_error_deg=2.):
        self.prior = np.asarray(prior_pose, dtype=float)
        if self.prior.shape != (3,) or not np.isfinite(self.prior).all():
            raise ValueError('prior_pose must contain three finite coordinates')
        for name, value in [('side_start_x', side_start_x), ('side_y', side_y)]:
            if not math.isfinite(float(value)):
                raise ValueError(name + ' must be finite')
            setattr(self, name, float(value))
        for name, value in [('cluster_distance', cluster_distance),
                            ('sampling_resolution', sampling_resolution),
                            ('fit_distance', fit_distance),
                            ('spatial_bin_size', spatial_bin_size),
                            ('robust_distance', robust_distance),
                            ('maximum_corner_distance', maximum_corner_distance),
                            ('minimum_heading_span', minimum_heading_span)]:
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(name + ' must be positive')
            setattr(self, name, float(value))
        if sampling_resolution >= cluster_distance:
            raise ValueError('sampling_resolution must be smaller than cluster_distance')
        if not math.isfinite(maximum_direction_error_deg) or not 0 < maximum_direction_error_deg < 90:
            raise ValueError('maximum_direction_error_deg must be between 0 and 90')
        self.minimum_direction_cosine = math.cos(math.radians(maximum_direction_error_deg))
        if (isinstance(minimum_points, bool)
                or int(minimum_points) != minimum_points or minimum_points < 2):
            raise ValueError('minimum_points must be an integer >= 2')
        self.minimum_points = int(minimum_points)
        if (not math.isfinite(maximum_heading_error_deg)
                or not 0 < maximum_heading_error_deg < 45):
            raise ValueError('maximum_heading_error_deg must be between 0 and 45')
        self.maximum_heading_error = math.radians(maximum_heading_error_deg)
        if opposite_side_y is not None:
            if not math.isfinite(opposite_side_y) or opposite_side_y <= side_y:
                raise ValueError('opposite_side_y must be above side_y')
        self.opposite_side_y = opposite_side_y
        self.diagnostics = {}

    @classmethod
    def from_geometry(cls, geometry, **settings):
        wall = geometry['wall_rectangles']['right_side_wall']
        parameters = dict(side_start_x=wall[0], side_y=wall[3])
        opposite = geometry['wall_rectangles'].get('exit_wall')
        if opposite is not None:
            parameters['opposite_side_y'] = opposite[1]
        parameters.update(settings)
        return cls(**parameters)

    def _line_candidates(self, points, forward, origin):
        """Fit the best-supported wall direction in each connected group."""
        _, indices = np.unique(np.floor(points / self.sampling_resolution).astype(np.int64),
                               axis=0, return_index=True)
        samples = points[indices]
        anchors = samples[np.linspace(0, len(samples)-1, min(24, len(samples))).astype(int)]
        a, b = np.triu_indices(len(anchors), 1)
        direction = anchors[b] - anchors[a]
        lengths = np.linalg.norm(direction, axis=1)
        good = lengths >= self.cluster_distance
        direction, anchors_a = direction[good] / lengths[good, None], anchors[a[good]]
        direction *= np.where(direction @ forward >= 0, 1., -1.)[:, None]
        normals = np.column_stack((-direction[:, 1], direction[:, 0]))
        offsets = np.sum(anchors_a * normals, axis=1)
        good = ((direction @ forward >= self.minimum_direction_cosine)
                & (offsets - normals @ origin < 0.))
        direction, normals, offsets = direction[good], normals[good], offsets[good]
        if not len(direction):
            return
        residual = np.abs(samples @ normals.T - offsets)
        support = np.sum(residual <= self.robust_distance, axis=0)
        index = int(np.argmax(support))
        if support[index] >= self.minimum_points:
            yield points, direction[index], offsets[index]

    def _fit_wall(self, local, tangent, offset, forward, origin, robust_distance=None):
        robust_distance = self.robust_distance if robust_distance is None else robust_distance
        normal = np.array([-tangent[1], tangent[0]])
        for iteration in range(12):
            previous_tangent = tangent.copy()
            residual = local @ normal - offset
            bins = np.floor((local @ tangent)/self.spatial_bin_size).astype(int)
            bins -= bins.min()
            counts = np.bincount(bins)
            weight = np.maximum(0., 1.-(residual/robust_distance)**2)**2 / counts[bins]
            if np.count_nonzero(weight) < self.minimum_points:
                return None
            total_weight = weight.sum()
            centre = np.sum(local * weight[:, None], axis=0) / total_weight
            delta = local - centre
            eigenvalues, vectors = np.linalg.eigh((delta * weight[:, None]).T @ delta / total_weight)
            if eigenvalues[-1] <= 1e-12 or eigenvalues[-1]-eigenvalues[0] <= 1e-12:
                return None
            tangent = vectors[:, -1]
            if tangent @ forward < 0:
                tangent = -tangent
            normal = np.array([-tangent[1], tangent[0]])
            new_offset = float(centre @ normal)
            change = abs(new_offset - offset)
            offset = new_offset
            if (iteration >= 2 and change < 1e-7
                    and np.linalg.norm(tangent-previous_tangent) < 1e-7):
                break
        if tangent @ forward < self.minimum_direction_cosine or offset-normal @ origin >= 0:
            return None
        return self._wall_result(local, tangent, offset, origin, iteration+1)

    def _wall_result(self, local, tangent, offset, origin, iterations):
        normal = np.array([-tangent[1], tangent[0]])
        residual = local @ normal - offset
        inliers = local[np.abs(residual) <= self.fit_distance]
        if len(inliers) < self.minimum_points:
            return None
        rays = inliers - origin
        denominator = rays @ normal
        valid = np.abs(denominator) > 1e-12
        distance = (offset-origin @ normal) / denominator[valid]
        intersections = origin + distance[:, None]*rays[valid]
        intersections = intersections[distance > 0]
        if len(intersections) < self.minimum_points:
            return None
        along = intersections @ tangent
        endpoint = intersections[int(np.argmin(along))]
        pose = Pose2D(self.side_start_x-float(endpoint @ tangent),
                      self.side_y-offset, math.atan2(-tangent[1], tangent[0]))
        rms = float(np.sqrt(np.mean(residual[np.abs(residual) <= self.fit_distance]**2)))
        diagnostics = dict(pose=[pose.x, pose.y, pose.yaw], wall_inliers=len(inliers),
                           wall_span=float(np.ptp(inliers @ tangent)), wall_offset=offset,
                           wall_tangent=tangent.tolist(), endpoint_base=endpoint.tolist(),
                           endpoint_input_count=len(inliers), iterations=iterations, rms=rms,
                           endpoint_prior_map_x=float(endpoint @ np.array([
                               math.cos(self.prior[2]), -math.sin(self.prior[2])])+self.prior[0]),
                           raw_endpoint_along=float(np.min(inliers @ tangent)),
                           ray_endpoint_along=float(np.min(along)))
        return pose, rms, len(inliers), diagnostics

    def _opposite_heading(self, points, groups, local, fitted, forward, origin):
        """A long parallel wall must also match the surveyed tunnel width."""
        if self.opposite_side_y is None:
            return None
        diagnostics = fitted[3]
        tangent = np.asarray(diagnostics['wall_tangent'])
        normal = np.array([-tangent[1], tangent[0]])
        uncertainty = math.atan2(2*fitted[1], diagnostics['wall_span'])
        nearby = local[np.abs(local @ normal-diagnostics['wall_offset']) <= self.fit_distance]
        width = self.opposite_side_y-self.side_y
        best = None
        for indices in groups:
            if len(indices) < self.minimum_points:
                continue
            group = points[indices]
            if (np.ptp(group @ forward) < self.minimum_heading_span
                    or np.median(group @ forward) <= origin @ forward):
                continue
            for other, direction, offset in self._line_candidates(group, -forward, origin):
                # Fit across the measured wall thickness, rather than locking
                # onto a narrow 6 mm band inside noisy range measurements.
                measured = self._fit_wall(other, direction, offset, -forward, origin,
                                          robust_distance=self.fit_distance)
                if measured is None:
                    continue
                _, rms, count, wall = measured
                span = wall['wall_span']
                if (span < self.minimum_heading_span or count < .7*len(other)
                        or rms > self.fit_distance/2.
                        or math.atan2(2*rms, span) > self.maximum_heading_error):
                    continue
                direction = -np.asarray(wall['wall_tangent'])
                disagreement = math.acos(float(np.clip(direction @ tangent, -1., 1.)))
                if disagreement > uncertainty+self.maximum_heading_error:
                    continue
                normal = np.array([-direction[1], direction[0]])
                offset = float(np.median(nearby @ normal))
                width_error = -wall['wall_offset']-offset-width
                if abs(width_error) > 2*self.fit_distance:
                    continue
                if best is None or span > best[0]:
                    best = span, direction, offset, rms, width_error
        return best

    def estimate(self, points, sensor_origin=(0., 0.)):
        points = np.asarray(points, dtype=float)
        origin = np.asarray(sensor_origin, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points must be N by 2 XY')
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError('sensor_origin must be finite XY')
        points = points[np.isfinite(points).all(axis=1)]
        self.diagnostics = {'input_points': len(points), 'sensor_origin': origin.tolist()}
        if len(points) < self.minimum_points:
            return AlignmentResult(False, reason='right wall returns missing', total=len(points))
        forward = np.array([math.cos(self.prior[2]), -math.sin(self.prior[2])])
        groups = _connected_groups(points, self.cluster_distance, self.sampling_resolution)
        best = None
        candidate_count = 0
        for indices in groups:
            if len(indices) < self.minimum_points:
                continue
            for local, tangent, offset in self._line_candidates(points[indices], forward, origin):
                fitted = self._fit_wall(local, tangent, offset, forward, origin)
                if fitted is None:
                    continue
                candidate_count += 1
                pose, rms, count, diagnostics = fitted
                if np.linalg.norm(np.asarray(diagnostics['endpoint_base'])-origin) > self.maximum_corner_distance:
                    continue
                yaw_delta = math.atan2(math.sin(pose.yaw-self.prior[2]),
                                       math.cos(pose.yaw-self.prior[2]))
                position_delta = math.hypot(pose.x-self.prior[0], pose.y-self.prior[1])
                normal = np.array([math.sin(pose.yaw), math.cos(pose.yaw)])
                wall_distance = abs(diagnostics['wall_offset']-normal @ origin)
                score = position_delta + wall_distance*abs(yaw_delta)
                if best is None or score < best[0]:
                    best = score, fitted, local
        self.diagnostics.update(clusters=len(groups), candidates=candidate_count)
        if best is None:
            return AlignmentResult(False, reason='connected right wall missing', total=len(points))
        _, fitted, local = best
        heading = self._opposite_heading(points, groups, local, fitted, forward, origin)
        if heading is not None:
            span, tangent, offset, heading_rms, width_error = heading
            fitted = self._wall_result(local, tangent, offset, origin, fitted[3]['iterations'])
            if fitted is None:
                return AlignmentResult(False, reason='right wall missing after heading fit', total=len(points))
            fitted[3].update(heading_source='opposite wall', heading_wall_span=span,
                             heading_wall_rms=heading_rms, tunnel_width_error=width_error)
        else:
            # A short, thick cloud cannot supply a precise heading by itself.
            # Clean short walls remain usable when the opposite wall is hidden.
            _, rms, _, diagnostics = fitted
            uncertainty = math.atan2(2*rms, diagnostics['wall_span'])
            if uncertainty > self.maximum_heading_error:
                self.diagnostics.update(diagnostics, heading_uncertainty=uncertainty)
                return AlignmentResult(False, reason='right wall heading uncertain; opposite wall not confirmed',
                                       total=len(points))
            fitted[3]['heading_source'] = 'right wall'
        pose, rms, count, diagnostics = fitted
        if np.linalg.norm(np.asarray(diagnostics['endpoint_base'])-origin) > self.maximum_corner_distance:
            return AlignmentResult(False, reason='entrance corner is not nearby', total=len(points))
        self.diagnostics.update(diagnostics)
        return AlignmentResult(True, pose, 'connected right wall and entrance corner matched',
                               rms, count, len(points))
