//! The convex hull of a point set: the least convex polyhedron holding every point. Built
//! incrementally: a first tetrahedron, then each point outside the hull so far replaces the
//! triangles it sees by a fan from it to their horizon. Points within a tolerance of the hull
//! are inside. Adjacent triangles on one plane become one face, its plane fitted to its corners.

use std::collections::HashMap;

use solidcore_record::Vec3;

/// Most points one hull takes.
pub const MAX_HULL_POINTS: usize = 200_000;

/// Most faces a hull may have.
pub const MAX_HULL_FACES: usize = 20_000;

/// A face's plane: `n · x = d`, `n` its outward unit normal.
#[derive(Clone, Copy, Debug)]
pub struct HalfSpace {
    pub n: Vec3,
    pub d: f64,
}

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

/// The vector maths the hull uses.
trait V {
    fn dot(self, o: Vec3) -> f64;
    fn cross(self, o: Vec3) -> Vec3;
    fn len(self) -> f64;
    fn normalized(self) -> Option<Vec3>;
}

impl V for Vec3 {
    fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    fn len(self) -> f64 {
        self.dot(self).sqrt()
    }
    fn normalized(self) -> Option<Vec3> {
        let l = self.len();
        (l.is_finite() && l > 1e-300).then(|| self / l)
    }
}

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
}

fn make(p: &[Vec3], a: usize, b: usize, c: usize) -> Option<Tri> {
    let (pa, pb, pc) = (*p.get(a)?, *p.get(b)?, *p.get(c)?);
    let n = (pb - pa).cross(pc - pa).normalized()?;
    Some(Tri { v: [a, b, c], n, d: n.dot(pa), alive: true })
}

fn invalid(why: &str) -> HullError {
    HullError(why.into())
}

/// The hull of `points`: its corners and its faces: each face's outward half-space and its corners (indices
/// into the corners) counter-clockwise about its outward normal.
pub fn hull_faces(points: &[Vec3]) -> Result<(Vec<Vec3>, Vec<(HalfSpace, Vec<usize>)>)> {
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
    let p = points;
    // The first tetrahedron: the extreme point in x, the point farthest from it, the one
    // farthest from their line, the one farthest from their plane.
    let i0 = (0..p.len()).min_by(|a, b| p[*a].x.total_cmp(&p[*b].x)).unwrap_or(0);
    let far = |f: &dyn Fn(Vec3) -> f64| (0..p.len()).max_by(|a, b| f(p[*a]).total_cmp(&f(p[*b]))).unwrap_or(0);
    let i1 = far(&|q| (q - p[i0]).len());
    let axis = p[i1] - p[i0];
    if axis.len() <= eps * 100.0 {
        return Err(invalid("a hull's points are all at one place"));
    }
    let i2 = far(&|q| axis.cross(q - p[i0]).len());
    let normal = axis.cross(p[i2] - p[i0]);
    if normal.len() <= eps * 100.0 * axis.len() {
        return Err(invalid("a hull's points are all on one line"));
    }
    let i3 = far(&|q| normal.dot(q - p[i0]).abs());
    let normal = normal.normalized().ok_or_else(|| invalid("a hull's points are all on one line"))?;
    if normal.dot(p[i3] - p[i0]).abs() <= eps * 100.0 {
        return Err(invalid("a hull's points are all on one plane: they hold no solid"));
    }
    let inside = (p[i0] + p[i1] + p[i2] + p[i3]) / 4.0;
    let mut tris: Vec<Tri> = Vec::new();
    let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
    let add = |tris: &mut Vec<Tri>, edges: &mut HashMap<(usize, usize), usize>, a: usize, b: usize, c: usize| -> Result<()> {
        let mut t = make(p, a, b, c).ok_or_else(|| invalid("a hull face has no area"))?;
        if t.dist(inside) > 0.0 {
            t = make(p, a, c, b).ok_or_else(|| invalid("a hull face has no area"))?;
        }
        let k = tris.len();
        let [x, y, z] = t.v;
        edges.insert((x, y), k);
        edges.insert((y, z), k);
        edges.insert((z, x), k);
        tris.push(t);
        Ok(())
    };
    for [a, b, c] in [[i0, i1, i2], [i0, i1, i3], [i0, i2, i3], [i1, i2, i3]] {
        add(&mut tris, &mut edges, a, b, c)?;
    }
    for (ip, q) in p.iter().enumerate() {
        if [i0, i1, i2, i3].contains(&ip) {
            continue;
        }
        // The triangles the point sees: a connected region grown from the one it sees most.
        let Some((start, d)) = tris.iter().enumerate().filter(|(_, t)| t.alive).map(|(k, t)| (k, t.dist(*q))).max_by(|a, b| a.1.total_cmp(&b.1)) else {
            break;
        };
        if d <= eps {
            continue;
        }
        let mut visible = vec![start];
        let mut seen = std::collections::HashSet::from([start]);
        let mut i = 0;
        while let Some(&k) = visible.get(i) {
            i += 1;
            let [a, b, c] = tris[k].v;
            for (u, w) in [(a, b), (b, c), (c, a)] {
                if let Some(&o) = edges.get(&(w, u))
                    && !seen.contains(&o)
                    && tris[o].alive
                    && tris[o].dist(*q) > eps
                {
                    seen.insert(o);
                    visible.push(o);
                }
            }
        }
        let mut horizon = Vec::new();
        for &k in &visible {
            let [a, b, c] = tris[k].v;
            for (u, w) in [(a, b), (b, c), (c, a)] {
                if edges.get(&(w, u)).is_none_or(|o| !seen.contains(o)) {
                    horizon.push((u, w));
                }
            }
        }
        for &k in &visible {
            tris[k].alive = false;
            let [a, b, c] = tris[k].v;
            for e in [(a, b), (b, c), (c, a)] {
                if edges.get(&e) == Some(&k) {
                    edges.remove(&e);
                }
            }
        }
        for (u, w) in horizon {
            let t = make(p, u, w, ip).ok_or_else(|| invalid("a hull face has no area"))?;
            let k = tris.len();
            edges.insert((u, w), k);
            edges.insert((w, ip), k);
            edges.insert((ip, u), k);
            tris.push(t);
        }
        if tris.iter().filter(|t| t.alive).count() > 2 * MAX_HULL_FACES {
            return Err(invalid(&format!("the hull has more than {MAX_HULL_FACES} faces")));
        }
    }
    merge(p, &tris, &edges, scale)
}

/// The hull's faces: adjacent triangles on one plane (within a tolerance far below the solid's)
/// merged, each with the plane fitted to its corners.
fn merge(p: &[Vec3], tris: &[Tri], edges: &HashMap<(usize, usize), usize>, scale: f64) -> Result<(Vec<Vec3>, Vec<(HalfSpace, Vec<usize>)>)> {
    let tol = (scale * 1e-9).max(1e-7) * 0.1;
    let alive: Vec<usize> = (0..tris.len()).filter(|k| tris[*k].alive).collect();
    let mut group: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for &k in &alive {
        if group.contains_key(&k) {
            continue;
        }
        let g = groups.len();
        let mut members = vec![k];
        group.insert(k, g);
        let mut i = 0;
        while let Some(&m) = members.get(i) {
            i += 1;
            let [a, b, c] = tris[m].v;
            for (u, w) in [(a, b), (b, c), (c, a)] {
                let Some(&o) = edges.get(&(w, u)) else { continue };
                if group.contains_key(&o) || tris[o].n.dot(tris[k].n) < 1.0 - 1e-9 {
                    continue;
                }
                if tris[o].v.iter().all(|v| tris[k].dist(p[*v]).abs() <= tol) {
                    group.insert(o, g);
                    members.push(o);
                }
            }
        }
        groups.push(members);
    }
    let mut faces: Vec<(HalfSpace, Vec<usize>)> = Vec::new();
    for members in &groups {
        // The face's outline: the edges of its triangles that no other of them shares, chained.
        let mut next: HashMap<usize, usize> = HashMap::new();
        for &m in members {
            let [a, b, c] = tris[m].v;
            for (u, w) in [(a, b), (b, c), (c, a)] {
                if edges.get(&(w, u)).is_none_or(|o| group.get(o) != group.get(&m)) {
                    next.insert(u, w);
                }
            }
        }
        let Some(&first) = next.keys().min() else { continue };
        let mut cycle = vec![first];
        let mut at = first;
        while let Some(&n) = next.get(&at) {
            if n == first {
                break;
            }
            if cycle.len() > next.len() {
                return Err(HullError("a hull face's outline does not close".into()));
            }
            cycle.push(n);
            at = n;
        }
        if cycle.len() != next.len() {
            // Not one loop: keep its triangles as faces of their own.
            for &m in members {
                faces.push((HalfSpace { n: tris[m].n, d: tris[m].d }, tris[m].v.to_vec()));
            }
            continue;
        }
        // Newell's normal of the outline, and the plane through its corners' centre.
        let mut n = Vec3::ZERO;
        let mut c = Vec3::ZERO;
        for (i, &a) in cycle.iter().enumerate() {
            let (pa, pb) = (p[a], p[cycle[(i + 1) % cycle.len()]]);
            n = n + Vec3::new((pa.y - pb.y) * (pa.z + pb.z), (pa.z - pb.z) * (pa.x + pb.x), (pa.x - pb.x) * (pa.y + pb.y));
            c = c + pa;
        }
        let n = n.normalized().ok_or_else(|| HullError("a hull face has no area".into()))?;
        let d = n.dot(c / cycle.len() as f64);
        if cycle.iter().any(|v| (n.dot(p[*v]) - d).abs() > tol * 10.0) {
            for &m in members {
                faces.push((HalfSpace { n: tris[m].n, d: tris[m].d }, tris[m].v.to_vec()));
            }
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
    for (_, cycle) in faces.iter_mut() {
        for v in cycle.iter_mut() {
            *v = *index.entry(*v).or_insert_with(|| {
                verts.push(p[*v]);
                verts.len() - 1
            });
        }
    }
    Ok((verts, faces))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
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
    fn every_point_is_inside_every_face_plane() {
        // A deterministic scatter on and inside a sphere.
        let mut pts = Vec::new();
        for i in 0..400 {
            let t = i as f64 * 0.618_033_988_75;
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / 400.0;
            let r = (1.0 - z * z).sqrt() * if i % 3 == 0 { 0.5 } else { 1.0 };
            pts.push(Vec3::new(r * (t * std::f64::consts::TAU).cos(), r * (t * std::f64::consts::TAU).sin(), z) * 50.0);
        }
        let (verts, faces) = hull_faces(&pts).unwrap();
        for (h, cycle) in &faces {
            assert!(pts.iter().all(|q| h.n.dot(*q) - h.d <= 1e-6));
            assert!(cycle.iter().all(|v| (h.n.dot(verts[*v]) - h.d).abs() <= 1e-6));
            assert!(cycle.len() >= 3);
        }
        // Closed: every edge used once each way.
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
        let flat: Vec<Vec3> = (0..10).map(|i| Vec3::new(i as f64, (i * i) as f64, 0.0)).collect();
        assert!(hull_faces(&flat).unwrap_err().to_string().contains("one plane"));
        assert!(hull_faces(&[Vec3::new(f64::NAN, 0.0, 0.0), Vec3::X, Vec3::Y, Vec3::Z]).is_err());
    }
}
