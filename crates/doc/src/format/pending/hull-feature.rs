//! A `hull` feature: the convex hull of points and bodies, one planar-faced solid. It holds
//! `points` (each `[x, y, z]`, numbers or expressions; absent when empty), `bodies` (absent when
//! empty) and `operation`. Every older design has no hull, so nothing in them is rewritten. The
//! last release opens a design that has one with a newer-format warning, keeps the feature as it
//! came and does not rebuild it, as it does for any feature it does not know.
