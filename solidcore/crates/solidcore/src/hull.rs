//! The convex hull of a point set: the least convex polyhedron holding every point. Built
//! incrementally: a first tetrahedron, then each point outside the hull so far replaces the
//! triangles it sees by a fan from it to their horizon. Points within a tolerance of the hull
//! are inside. Adjacent triangles on one plane become one face, its plane fitted to its corners.

use std::collections::{HashMap, HashSet};

use crate::Vec3;

/// Most points one hull takes.
pub const MAX_HULL_POINTS: usize = 200_000;

/// Most faces a hull may have.
pub const MAX_HULL_FACES: usize = 20_000;

/// A face's plane: `n · x = d`.
#[derive(Clone, Copy, Debug)]
pub struct HalfSpace {
    /// The outward unit normal.
    pub n: Vec3,
    /// The plane's offset along `n`.
    pub d: f64,
}

/// A hull face: its plane and its corners (indices into the hull's corners), counter-clockwise
/// about the outward normal.
pub type HullFace = (HalfSpace, Vec<usize>);

/// Why a hull was refused: too few points, not finite, all on a plane, too many faces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HullError(pub String);

impl std::fmt::Display for HullError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for HullError {}

type Result<T> = std::result::Result<T, HullError>;

#[derive(Clone, Copy)]
struct Tri {
    v: [usize; 3],
    n: Vec3,
    d: f64,
    alive: bool,
}

impl Tri {
    fn dist(&self, p: Vec3) -> f64 {
        self.n.dot(p) - self.d
    }
    fn edges(&self) -> [(usize, usize); 3] {
        let [a, b, c] = self.v;
        [(a, b), (b, c), (c, a)]
    }
}

fn invalid(why: &str) -> HullError {
    HullError(why.into())
}

fn unit(v: Vec3) -> Option<Vec3> {
    v.length_and_unit().map(|(_, u)| u)
}

/// The points, read without panicking.
struct Pts<'a>(&'a [Vec3]);

impl Pts<'_> {
    fn at(&self, i: usize) -> Result<Vec3> {
        self.0.get(i).copied().ok_or_else(|| invalid("a hull corner is missing"))
    }
    fn tri(&self, a: usize, b: usize, c: usize) -> Result<Tri> {
        let (pa, pb, pc) = (self.at(a)?, self.at(b)?, self.at(c)?);
        let n = unit((pb - pa).cross(pc - pa)).ok_or_else(|| invalid("a hull face has no area"))?;
        Ok(Tri { v: [a, b, c], n, d: n.dot(pa), alive: true })
    }
}

/// The triangles so far and each directed edge's triangle.
#[derive(Default)]
struct Mesh {
    tris: Vec<Tri>,
    edges: HashMap<(usize, usize), usize>,
}

impl Mesh {
    fn get(&self, k: usize) -> Result<Tri> {
        self.tris.get(k).copied().ok_or_else(|| invalid("a hull triangle is missing"))
    }
    fn push(&mut self, t: Tri) {
        let k = self.tris.len();
        for e in t.edges() {
            self.edges.insert(e, k);
        }
        self.tris.push(t);
    }
    fn kill(&mut self, k: usize) -> Result<()> {
        let t = self.get(k)?;
        for e in t.edges() {
            if self.edges.get(&e) == Some(&k) {
                self.edges.remove(&e);
            }
        }
        if let Some(t) = self.tris.get_mut(k) {
            t.alive = false;
        }
        Ok(())
    }
    /// The live triangle across edge `(u, w)` of a triangle: the one that has `(w, u)`.
    fn across(&self, u: usize, w: usize) -> Option<usize> {
        self.edges.get(&(w, u)).copied()
    }
}

/// The hull of `points`: its corners and its faces.
///
/// # Errors
/// [`HullError`] for points that hold no solid (fewer than 4, not finite, all on one line or
/// plane) or a hull of more than [`MAX_HULL_FACES`] faces.
pub fn hull_faces(points: &[Vec3]) -> Result<(Vec<Vec3>, Vec<HullFace>)> {
    if points.len() < 4 {
        return Err(invalid("a hull needs at least 4 points"));
    }
    if points.len() > MAX_HULL_POINTS {
        return Err(invalid(&format!("a hull takes at most {MAX_HULL_POINTS} points")));
    }
    if points.iter().any(|p| !(p.x.is_finite() && p.y.is_finite() && p.z.is_finite())) {
        return Err(invalid("a hull's points must be finite"));
    }
    let scale = points.iter().map(|p| p.x.abs().max(p.y.abs()).max(p.z.abs())).fold(1.0, f64::max);
    let eps = scale * 1e-10;
    let p = Pts(points);
    // The first tetrahedron: the extreme point in x, the point farthest from it, the one
    // farthest from their line, the one farthest from their plane.
    let far = |f: &dyn Fn(Vec3) -> f64| points.iter().enumerate().max_by(|a, b| f(*a.1).total_cmp(&f(*b.1))).map_or(0, |(i, _)| i);
    let i0 = points.iter().enumerate().min_by(|a, b| a.1.x.total_cmp(&b.1.x)).map_or(0, |(i, _)| i);
    let p0 = p.at(i0)?;
    let i1 = far(&|q| (q - p0).length());
    let axis = p.at(i1)? - p0;
    if axis.length() <= eps * 100.0 {
        return Err(invalid("a hull's points are all at one place"));
    }
    let i2 = far(&|q| axis.cross(q - p0).length());
    let normal = axis.cross(p.at(i2)? - p0);
    if normal.length() <= eps * 100.0 * axis.length() {
        return Err(invalid("a hull's points are all on one line"));
    }
    let i3 = far(&|q| normal.dot(q - p0).abs());
    let normal = unit(normal).ok_or_else(|| invalid("a hull's points are all on one line"))?;
    if normal.dot(p.at(i3)? - p0).abs() <= eps * 100.0 {
        return Err(invalid("a hull's points are all on one plane: they hold no solid"));
    }
    let inside = (p0 + p.at(i1)? + p.at(i2)? + p.at(i3)?) * 0.25;
    let mut m = Mesh::default();
    for [a, b, c] in [[i0, i1, i2], [i0, i1, i3], [i0, i2, i3], [i1, i2, i3]] {
        let t = p.tri(a, b, c)?;
        m.push(if t.dist(inside) > 0.0 { p.tri(a, c, b)? } else { t });
    }
    for (ip, q) in points.iter().enumerate() {
        if [i0, i1, i2, i3].contains(&ip) {
            continue;
        }
        // The triangles the point sees: a connected region grown from the one it sees most.
        let best = m.tris.iter().enumerate().filter(|(_, t)| t.alive).map(|(k, t)| (k, t.dist(*q))).max_by(|a, b| a.1.total_cmp(&b.1));
        let Some((start, d)) = best else { break };
        if d <= eps {
            continue;
        }
        let mut visible = vec![start];
        let mut seen = HashSet::from([start]);
        let mut i = 0;
        while let Some(&k) = visible.get(i) {
            i += 1;
            for (u, w) in m.get(k)?.edges() {
                let Some(o) = m.across(u, w) else { continue };
                let t = m.get(o)?;
                if !seen.contains(&o) && t.alive && t.dist(*q) > eps {
                    seen.insert(o);
                    visible.push(o);
                }
            }
        }
        let mut horizon = Vec::new();
        for &k in &visible {
            for (u, w) in m.get(k)?.edges() {
                if m.across(u, w).is_none_or(|o| !seen.contains(&o)) {
                    horizon.push((u, w));
                }
            }
        }
        for &k in &visible {
            m.kill(k)?;
        }
        for (u, w) in horizon {
            m.push(p.tri(u, w, ip)?);
        }
        if m.tris.iter().filter(|t| t.alive).count() > 2 * MAX_HULL_FACES {
            return Err(invalid(&format!("the hull has more than {MAX_HULL_FACES} faces")));
        }
    }
    merge(&p, &m, scale)
}

/// The hull's faces: adjacent triangles on one plane (within a tolerance far below the solid's)
/// merged, each with the plane fitted to its corners.
fn merge(p: &Pts<'_>, m: &Mesh, scale: f64) -> Result<(Vec<Vec3>, Vec<HullFace>)> {
    let tol = (scale * 1e-9).max(1e-7) * 0.1;
    let mut group: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (k, first) in m.tris.iter().enumerate() {
        if !first.alive || group.contains_key(&k) {
            continue;
        }
        let g = groups.len();
        let mut members = vec![k];
        group.insert(k, g);
        let mut i = 0;
        while let Some(&mm) = members.get(i) {
            i += 1;
            for (u, w) in m.get(mm)?.edges() {
                let Some(o) = m.across(u, w) else { continue };
                let t = m.get(o)?;
                if group.contains_key(&o) || t.n.dot(first.n) < 1.0 - 1e-9 {
                    continue;
                }
                let mut on = true;
                for v in t.v {
                    on &= first.dist(p.at(v)?).abs() <= tol;
                }
                if on {
                    group.insert(o, g);
                    members.push(o);
                }
            }
        }
        groups.push(members);
    }
    let mut faces: Vec<HullFace> = Vec::new();
    for members in &groups {
        let triangles = |faces: &mut Vec<HullFace>| -> Result<()> {
            for &k in members {
                let t = m.get(k)?;
                faces.push((HalfSpace { n: t.n, d: t.d }, t.v.to_vec()));
            }
            Ok(())
        };
        // The face's outline: the edges of its triangles that no other of them shares, chained.
        let mut next: HashMap<usize, usize> = HashMap::new();
        for &k in members {
            for (u, w) in m.get(k)?.edges() {
                if m.across(u, w).is_none_or(|o| group.get(&o) != group.get(&k)) {
                    next.insert(u, w);
                }
            }
        }
        let Some(&first) = next.keys().min() else { continue };
        let mut cycle = vec![first];
        let mut at = first;
        while let Some(&n) = next.get(&at) {
            if n == first || cycle.len() > next.len() {
                break;
            }
            cycle.push(n);
            at = n;
        }
        if cycle.len() != next.len() {
            // Not one loop: its triangles stay faces of their own.
            triangles(&mut faces)?;
            continue;
        }
        // Newell's normal of the outline, and the plane through its corners' centre.
        let mut n = Vec3::ZERO;
        let mut c = Vec3::ZERO;
        for (a, b) in cycle.iter().zip(cycle.iter().cycle().skip(1)) {
            let (pa, pb) = (p.at(*a)?, p.at(*b)?);
            n += Vec3::new((pa.y - pb.y) * (pa.z + pb.z), (pa.z - pb.z) * (pa.x + pb.x), (pa.x - pb.x) * (pa.y + pb.y));
            c += pa;
        }
        let n = unit(n).ok_or_else(|| invalid("a hull face has no area"))?;
        let d = n.dot(c * (1.0 / cycle.len() as f64));
        let mut flat = true;
        for v in &cycle {
            flat &= (n.dot(p.at(*v)?) - d).abs() <= tol * 10.0;
        }
        if !flat {
            triangles(&mut faces)?;
            continue;
        }
        faces.push((HalfSpace { n, d }, cycle));
    }
    if faces.len() > MAX_HULL_FACES {
        return Err(invalid(&format!("the hull has more than {MAX_HULL_FACES} faces")));
    }
    // The corners the faces use, numbered as the faces first meet them.
    let mut index: HashMap<usize, usize> = HashMap::new();
    let mut verts = Vec::new();
    for (_, cycle) in &mut faces {
        for v in cycle.iter_mut() {
            let next = verts.len();
            let k = *index.entry(*v).or_insert(next);
            if k == next {
                verts.push(p.at(*v)?);
            }
            *v = k;
        }
    }
    Ok((verts, faces))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::indexing_slicing, clippy::disallowed_methods)]
mod tests {
    use super::*;

    fn cube(s: f64) -> Vec<Vec3> {
        let mut v = Vec::new();
        for x in [0.0, s] {
            for y in [0.0, s] {
                for z in [0.0, s] {
                    v.push(Vec3::new(x, y, z));
                }
            }
        }
        v
    }

    #[test]
    fn a_cubes_corners_and_inner_points_give_six_square_faces() {
        let mut pts = cube(10.0);
        pts.extend([Vec3::new(5.0, 5.0, 5.0), Vec3::new(1.0, 2.0, 3.0), Vec3::new(5.0, 5.0, 10.0), Vec3::new(10.0, 5.0, 5.0)]);
        let (verts, faces) = hull_faces(&pts).unwrap();
        assert_eq!(faces.len(), 6);
        assert!(faces.iter().all(|(_, c)| c.len() == 4));
        assert_eq!(verts.len(), 8);
    }

    #[test]
    fn every_point_is_inside_every_face_plane_and_the_faces_close() {
        // A deterministic scatter on and inside a sphere.
        let mut pts = Vec::new();
        for i in 0..400_u32 {
            let t = f64::from(i) * 0.618_033_988_75 * std::f64::consts::TAU;
            let z = 1.0 - 2.0 * (f64::from(i) + 0.5) / 400.0;
            let r = (1.0 - z * z).sqrt() * if i % 3 == 0 { 0.5 } else { 1.0 };
            pts.push(Vec3::new(r * t.cos(), r * t.sin(), z) * 50.0);
        }
        let (verts, faces) = hull_faces(&pts).unwrap();
        for (h, cycle) in &faces {
            assert!(pts.iter().all(|q| h.n.dot(*q) - h.d <= 1e-6));
            assert!(cycle.iter().all(|v| (h.n.dot(verts[*v]) - h.d).abs() <= 1e-6));
            assert!(cycle.len() >= 3);
        }
        let mut uses: HashMap<(usize, usize), i32> = HashMap::new();
        for (_, c) in &faces {
            for i in 0..c.len() {
                *uses.entry((c[i], c[(i + 1) % c.len()])).or_default() += 1;
            }
        }
        assert!(uses.iter().all(|((a, b), n)| *n == 1 && uses.get(&(*b, *a)) == Some(&1)));
    }

    #[test]
    fn flat_or_too_few_points_are_refused() {
        assert!(hull_faces(&cube(1.0)[..3]).is_err());
        let flat: Vec<Vec3> = (0..10_u32).map(|i| Vec3::new(f64::from(i), f64::from(i * i), 0.0)).collect();
        assert!(hull_faces(&flat).unwrap_err().to_string().contains("one plane"));
        assert!(hull_faces(&[Vec3::new(f64::NAN, 0.0, 0.0), Vec3::X, Vec3::Y, Vec3::Z]).is_err());
    }
}
