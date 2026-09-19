//! Spatial islands of prepared geometry, used to frame the camera on the
//! bulk of a drawable instead of on the union of far-flung pieces.
//!
//! Some drawables (e.g. `hei_bank_heist_card`, rpf-cli#4) are authored as
//! several small pieces metres apart rather than one compact mesh. Framing
//! the union of their vertices, as `render::bounds_of` used to, shrinks the
//! actual subject to a handful of pixels. This module groups geometry into
//! islands by proximity and picks the one the camera should fit, without
//! touching what gets drawn — stray islands stay in the draw list, they are
//! simply left out of the frame.

use crate::math::Vec3;
use crate::ydd::DrawableBounds;

use super::camera::fit_radius;
use super::mesh::PreparedGeometry;

/// Empty space between two pieces that still counts as one object, as a
/// multiple of the larger piece's own diagonal. 1.0 reads as: two blobs
/// belong together when the gap between them is no wider than the larger
/// blob itself. A normally-jointed multi-geometry prop (lid + body, seat +
/// legs) has a gap far smaller than either part, so this never splits it;
/// `hei_bank_heist_card`'s two cards are ~72x their own diagonal apart.
const LINK_FACTOR: f32 = 1.0;

/// Absolute floor on that gap, in metres, so pieces that are small in
/// absolute terms (decals, screws, a card) are never split merely for
/// being small.
const MIN_LINK: f32 = 0.25;

/// Only reframe when the union's fit radius is at least this many times the
/// radius the kept island alone would need: at the default 40 degree fov
/// that is a subject occupying under a quarter of the frame width (under 6%
/// of its area). Conservative on purpose — a picture that was already
/// readable never changes.
const MIN_SHRINK: f32 = 4.0;

/// Above this many geometries the pairwise pass is skipped and the union is
/// framed as before, so a pathological drawable cannot make framing
/// quadratic (this runs in wasm too).
const MAX_CLUSTERED_GEOMETRIES: usize = 4096;

/// One prepared geometry's own axis-aligned box.
#[derive(Clone, Copy)]
pub(crate) struct GeometryBox {
    pub min: Vec3,
    pub max: Vec3,
    pub triangles: usize,
}

impl GeometryBox {
    fn diagonal(&self) -> f32 {
        (self.max - self.min).length()
    }
}

/// A set of geometries linked to one another, directly or through neighbours.
pub(crate) struct Island {
    pub min: Vec3,
    pub max: Vec3,
    pub triangles: usize,
    pub members: usize,
    /// Lowest index among the geometries that make up this island, into the
    /// slice `geometry_boxes` was built from. Used only to break ties
    /// deterministically.
    pub first_geometry: usize,
}

impl Island {
    pub fn bounds(&self) -> DrawableBounds {
        DrawableBounds {
            center: (self.min + self.max) * 0.5,
            sphere_radius: (self.max - self.min).length() * 0.5,
            box_min: self.min,
            box_max: self.max,
        }
    }

    fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

/// Per-geometry boxes, skipping geometries with no finite vertex.
pub(crate) fn geometry_boxes(geometries: &[PreparedGeometry<'_>]) -> Vec<GeometryBox> {
    let mut boxes = Vec::with_capacity(geometries.len());

    for geometry in geometries {
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut seen = false;

        for vertex in &geometry.verts {
            let p = vertex.position;
            if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() {
                min = min.min(p);
                max = max.max(p);
                seen = true;
            }
        }

        if seen {
            boxes.push(GeometryBox { min, max, triangles: geometry.indices.len() / 3 });
        }
    }

    boxes
}

/// Union of `boxes`, or degenerate zero bounds when empty.
pub(crate) fn union_bounds(boxes: &[GeometryBox]) -> DrawableBounds {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut seen = false;

    for b in boxes {
        min = min.min(b.min);
        max = max.max(b.max);
        seen = true;
    }

    if !seen || min.x > max.x {
        return DrawableBounds { center: Vec3::ZERO, sphere_radius: 0.0, box_min: Vec3::ZERO, box_max: Vec3::ZERO };
    }

    DrawableBounds {
        center: (min + max) * 0.5,
        sphere_radius: (max - min).length() * 0.5,
        box_min: min,
        box_max: max,
    }
}

/// Distance between two boxes; zero when they touch or overlap.
fn box_gap(a: &GeometryBox, b: &GeometryBox) -> f32 {
    let axis_gap = |a_min: f32, a_max: f32, b_min: f32, b_max: f32| {
        (b_min - a_max).max(a_min - b_max).max(0.0)
    };
    let gap = Vec3::new(
        axis_gap(a.min.x, a.max.x, b.min.x, b.max.x),
        axis_gap(a.min.y, a.max.y, b.min.y, b.max.y),
        axis_gap(a.min.z, a.max.z, b.min.z, b.max.z),
    );
    gap.length()
}

/// True when `a` and `b` are close enough to be treated as one object.
fn linked(a: &GeometryBox, b: &GeometryBox) -> bool {
    box_gap(a, b) <= (LINK_FACTOR * a.diagonal().max(b.diagonal())).max(MIN_LINK)
}

/// Single-linkage islands of `boxes`, in no particular order. Thresholds are
/// evaluated between the original geometries, not the growing island, so the
/// link graph is fixed up front and one O(n^2) pass over it is enough —
/// deterministic, and a chain of small adjoining pieces (a fence, a railing)
/// still ends up as a single island via its neighbours.
pub(crate) fn islands(boxes: &[GeometryBox]) -> Vec<Island> {
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    if n <= MAX_CLUSTERED_GEOMETRIES {
        for i in 0..n {
            for j in (i + 1)..n {
                if linked(&boxes[i], &boxes[j]) {
                    union(&mut parent, i, j);
                }
            }
        }
    }

    let mut by_root: std::collections::BTreeMap<usize, Island> = std::collections::BTreeMap::new();
    for (i, b) in boxes.iter().enumerate() {
        let root = find(&mut parent, i);
        by_root
            .entry(root)
            .and_modify(|island| {
                island.min = island.min.min(b.min);
                island.max = island.max.max(b.max);
                island.triangles += b.triangles;
                island.members += 1;
                island.first_geometry = island.first_geometry.min(i);
            })
            .or_insert(Island {
                min: b.min,
                max: b.max,
                triangles: b.triangles,
                members: 1,
                first_geometry: i,
            });
    }

    by_root.into_values().collect()
}

/// The bounds to frame when clustering should take over, and how many
/// islands were found.
///
/// `current` is the bounds the camera would otherwise use (the drawable's
/// stored bounds when they were accepted, the vertex union otherwise).
/// Returns `None` in the override slot when there is one island, or when
/// reframing would not meaningfully change the picture.
pub(crate) fn framing_override(
    boxes: &[GeometryBox],
    current: &DrawableBounds,
) -> (usize, Option<(DrawableBounds, usize)>) {
    let found = islands(boxes);
    if found.len() < 2 {
        return (found.len(), None);
    }

    // Most triangles first; ties broken by distance to the drawable origin
    // (RAGE places entities by the drawable origin, so on a tie the piece
    // sitting at the origin is the one meant to be seen), then by the
    // lowest geometry index for full determinism.
    let mut ranked: Vec<&Island> = found.iter().collect();
    ranked.sort_by(|a, b| {
        b.triangles
            .cmp(&a.triangles)
            .then_with(|| a.center().length().total_cmp(&b.center().length()))
            .then_with(|| a.first_geometry.cmp(&b.first_geometry))
    });
    let primary = ranked[0];

    let primary_bounds = primary.bounds();
    if fit_radius(&primary_bounds) * MIN_SHRINK > fit_radius(current) {
        return (found.len(), None);
    }

    let excluded: usize = boxes.len() - primary.members;
    (found.len(), Some((primary_bounds, excluded)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_at(center: Vec3, half_extent: f32, triangles: usize) -> GeometryBox {
        GeometryBox {
            min: center - Vec3::new(half_extent, half_extent, half_extent),
            max: center + Vec3::new(half_extent, half_extent, half_extent),
            triangles,
        }
    }

    /// The bank card's shape: two 0.08m boxes 5.8m apart. Reproduces the
    /// reported bug and pins the fix.
    #[test]
    fn far_apart_pieces_split_into_two_islands() {
        let a = cube_at(Vec3::ZERO, 0.04, 18);
        let b = cube_at(Vec3::new(5.8, 0.0, 0.0), 0.04, 18);
        let boxes = [a, b];

        assert_eq!(islands(&boxes).len(), 2);

        let current = union_bounds(&boxes);
        let (count, override_) = framing_override(&boxes, &current);
        assert_eq!(count, 2);
        let (framed, excluded) = override_.expect("union should be rejected in favour of one island");
        assert_eq!(excluded, 1);
        assert!(
            fit_radius(&framed) < 0.1,
            "expected a small fitted radius, got {}", fit_radius(&framed)
        );
        assert!(
            fit_radius(&current) > 2.5,
            "sanity: the union radius should still be large, got {}", fit_radius(&current)
        );
    }

    #[test]
    fn touching_pieces_are_one_island() {
        let a = cube_at(Vec3::ZERO, 0.5, 4);
        let b = cube_at(Vec3::new(1.0, 0.0, 0.0), 0.5, 4);
        assert_eq!(islands(&[a, b]).len(), 1);
    }

    #[test]
    fn a_gap_smaller_than_the_pieces_stays_one_island() {
        let a = cube_at(Vec3::ZERO, 0.5, 4);
        let b = cube_at(Vec3::new(1.5, 0.0, 0.0), 0.5, 4); // 0.5m gap, pieces are 1m wide
        assert_eq!(islands(&[a, b]).len(), 1);
    }

    #[test]
    fn a_gap_under_the_absolute_floor_stays_one_island() {
        let a = cube_at(Vec3::ZERO, 0.005, 2);
        let b = cube_at(Vec3::new(0.21, 0.0, 0.0), 0.005, 2); // ~0.2m gap, 1cm pieces
        assert_eq!(islands(&[a, b]).len(), 1);
    }

    #[test]
    fn pieces_chain_through_their_neighbours() {
        let boxes: Vec<GeometryBox> = (0..5)
            .map(|i| cube_at(Vec3::new(i as f32 * 0.8, 0.0, 0.0), 0.5, 1))
            .collect();
        assert_eq!(islands(&boxes).len(), 1);
    }

    #[test]
    fn similar_islands_keep_the_union() {
        let a = cube_at(Vec3::ZERO, 0.5, 4);
        let b = cube_at(Vec3::new(3.0, 0.0, 0.0), 0.5, 4);
        let boxes = [a, b];
        assert_eq!(islands(&boxes).len(), 2);

        let current = union_bounds(&boxes);
        let (_, override_) = framing_override(&boxes, &current);
        assert!(override_.is_none(), "shrink is under the threshold; framing should not change");
    }

    #[test]
    fn the_island_with_more_triangles_wins() {
        let dense = cube_at(Vec3::new(3.0, 0.0, 0.0), 0.02, 100);
        let sparse = cube_at(Vec3::ZERO, 0.02, 4);
        let boxes = [sparse, dense];
        let current = union_bounds(&boxes);
        let (_, override_) = framing_override(&boxes, &current);
        let (framed, _) = override_.unwrap();
        assert!((framed.center - Vec3::new(3.0, 0.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn equal_islands_are_broken_by_the_origin_then_deterministically() {
        let at_origin = cube_at(Vec3::ZERO, 0.02, 10);
        let far = cube_at(Vec3::new(3.0, 0.0, 0.0), 0.02, 10);
        let boxes = [far, at_origin];
        let current = union_bounds(&boxes);
        let (_, override_) = framing_override(&boxes, &current);
        let (framed, _) = override_.unwrap();
        assert!((framed.center - Vec3::ZERO).length() < 1e-3, "expected the origin island to win the tie");

        // Repeated calls agree.
        let (_, override_2) = framing_override(&boxes, &current);
        let (framed_2, _) = override_2.unwrap();
        assert_eq!(framed.center, framed_2.center);
    }

    #[test]
    fn box_gap_is_zero_when_boxes_overlap_and_diagonal_otherwise() {
        let a = cube_at(Vec3::ZERO, 1.0, 1);
        let b = cube_at(Vec3::new(0.5, 0.0, 0.0), 1.0, 1);
        assert_eq!(box_gap(&a, &b), 0.0);

        let c = cube_at(Vec3::new(3.0, 4.0, 0.0), 1.0, 1);
        // a spans [-1,1]^3, c spans [2,4]x[3,5]x[-1,1]; gap on x is 1, on y is 2.
        let gap = box_gap(&a, &c);
        assert!((gap - (1.0f32 * 1.0 + 2.0 * 2.0).sqrt()).abs() < 1e-4, "gap was {gap}");
    }

    #[test]
    fn huge_geometry_counts_skip_clustering() {
        // Build more boxes than the cap, all coincident (would otherwise
        // trivially merge anyway) — this only exercises that the O(n^2)
        // pass is skipped without panicking or hanging.
        let boxes: Vec<GeometryBox> = (0..(MAX_CLUSTERED_GEOMETRIES + 1))
            .map(|i| cube_at(Vec3::new(i as f32 * 100.0, 0.0, 0.0), 0.1, 1))
            .collect();
        let found = islands(&boxes);
        // Above the cap every box is its own island (no linking performed).
        assert_eq!(found.len(), boxes.len());
    }
}
