//! The convex hull of a point set ([`solidcore::hull`]), in the record's vectors: its corners
//! and its faces, each with its plane and its corners counter-clockwise about its outward normal.

use solidcore_record::Vec3;

pub use solidcore::hull::{HullError, MAX_HULL_FACES, MAX_HULL_POINTS};

/// A face's plane: `n · x = d`, `n` its outward unit normal.
#[derive(Clone, Copy, Debug)]
pub struct HalfSpace {
    /// The outward unit normal.
    pub n: Vec3,
    /// The plane's offset along `n`.
    pub d: f64,
}

/// The hull of `points`: its corners and its faces.
///
/// # Errors
/// [`HullError`] for points that hold no solid (fewer than 4, not finite, all on one plane) or a
/// hull of more than [`MAX_HULL_FACES`] faces.
#[allow(clippy::type_complexity)]
pub fn hull_faces(points: &[Vec3]) -> Result<(Vec<Vec3>, Vec<(HalfSpace, Vec<usize>)>), HullError> {
    let to = |v: Vec3| solidcore::Vec3::new(v.x, v.y, v.z);
    let back = |v: solidcore::Vec3| Vec3::new(v.x, v.y, v.z);
    let (verts, faces) = solidcore::hull::hull_faces(&points.iter().map(|p| to(*p)).collect::<Vec<_>>())?;
    Ok((verts.into_iter().map(back).collect(), faces.into_iter().map(|(h, c)| (HalfSpace { n: back(h.n), d: h.d }, c)).collect()))
}
