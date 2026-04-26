//! Brute-force parallel k-nearest-neighbour search.
//!
//! Provides a single API surface — [`knn_dense_f32`], [`knn_dense_f64`],
//! [`knn_dense_bool`], [`knn_dense_u32`] — over a [`DistanceMetric`] enum
//! covering the standard ML and cheminformatics distance metrics.
//!
//! All functions return `(indices, distances)` where:
//!   - `indices` is an `(n, k)` `u32` array of **non-self** neighbour indices,
//!     sorted ascending by distance.
//!   - `distances` is the corresponding `(n, k)` `f32` array of distances.
//!
//! Brute-force is O(n² × d) — suitable for n up to roughly 50 000 with a
//! moderate feature dimension. For much larger datasets you'd want an ANN
//! index (HNSW, etc.); not provided here.
//!
//! Parallelism: each row's k-nearest search runs on its own rayon task.
//! No shared mutable state during search.

use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::BinaryHeap;

/// Distance metric for KNN search.
///
/// Variants are valid only for matching input dtypes:
///   - Continuous metrics (`Euclidean`, `Cosine`, `Tanimoto`, etc.) work on
///     `f32` / `f64` arrays.
///   - `Jaccard` / `Dice` / `Hamming` work on `bool` arrays (with packed-u64
///     popcount internally for performance).
///   - `HashJaccard` works on `u32` MinHash signatures.
///
/// All distances are non-negative and symmetric, making them valid for UMAP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// √Σ(aᵢ − bᵢ)²
    Euclidean,
    /// Σ(aᵢ − bᵢ)²  — monotonic with Euclidean, faster (no sqrt).
    SquaredEuclidean,
    /// Σ|aᵢ − bᵢ|
    Manhattan,
    /// max|aᵢ − bᵢ|
    Chebyshev,
    /// 1 − a·b / (‖a‖·‖b‖)
    Cosine,
    /// 1 − Pearson correlation
    Correlation,
    /// Continuous Tanimoto: 1 − Σmin(aᵢ, bᵢ) / Σmax(aᵢ, bᵢ).
    /// Inputs must be non-negative.
    Tanimoto,
    /// Bit-vector Jaccard distance: 1 − |A ∩ B| / |A ∪ B|.
    Jaccard,
    /// Bit-vector Dice distance: 1 − 2|A ∩ B| / (|A| + |B|).
    Dice,
    /// Hamming distance, normalised to [0, 1]: (# differing bits) / n_bits.
    Hamming,
    /// MinHash signature distance: 1 − (# matching positions) / signature_length.
    HashJaccard,
    /// Minkowski Lp distance: (Σ|aᵢ − bᵢ|^p)^(1/p).
    /// p=1 → Manhattan, p=2 → Euclidean, p=∞ → Chebyshev.
    Minkowski(f32),
    /// Canberra distance: Σ |aᵢ − bᵢ| / (|aᵢ| + |bᵢ|).
    Canberra,
    /// Bray-Curtis distance: Σ|aᵢ − bᵢ| / Σ|aᵢ + bᵢ|.
    BrayCurtis,
}

impl DistanceMetric {
    /// Parse a metric name (case-insensitive, dashes/underscores/spaces ignored).
    pub fn parse(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase().replace(['-', '_', ' '], "");
        Some(match n.as_str() {
            "euclidean" | "l2" => Self::Euclidean,
            "sqeuclidean" | "squaredeuclidean" => Self::SquaredEuclidean,
            "manhattan" | "cityblock" | "l1" => Self::Manhattan,
            "chebyshev" | "linf" | "linfinity" => Self::Chebyshev,
            "cosine" => Self::Cosine,
            "correlation" => Self::Correlation,
            "tanimoto" | "tanimotocontinuous" => Self::Tanimoto,
            "jaccard" => Self::Jaccard,
            "dice" => Self::Dice,
            "hamming" => Self::Hamming,
            "hashjaccard" | "minhash" => Self::HashJaccard,
            // Minkowski defaults to p=2 (Euclidean equivalent).  Callers
            // wanting a custom p should construct ``Minkowski(p)`` directly,
            // or via the Python binding's ``p`` kwarg.
            "minkowski" => Self::Minkowski(2.0),
            "canberra" => Self::Canberra,
            "braycurtis" => Self::BrayCurtis,
            _ => return None,
        })
    }
}

// ── Distance functions on dense rows ─────────────────────────────────────────

#[inline]
fn dist_f32(a: ArrayView1<f32>, b: ArrayView1<f32>, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean => {
            let mut s = 0.0_f32;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                s += d * d;
            }
            s.sqrt()
        }
        DistanceMetric::SquaredEuclidean => {
            let mut s = 0.0_f32;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                s += d * d;
            }
            s
        }
        DistanceMetric::Manhattan => {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
        }
        DistanceMetric::Chebyshev => {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max)
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0_f32;
            let mut na = 0.0_f32;
            let mut nb = 0.0_f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                let sim = dot / (na.sqrt() * nb.sqrt());
                (1.0 - sim).max(0.0)
            }
        }
        DistanceMetric::Correlation => {
            let n = a.len() as f32;
            let mean_a = a.iter().sum::<f32>() / n;
            let mean_b = b.iter().sum::<f32>() / n;
            let mut dot = 0.0_f32;
            let mut na = 0.0_f32;
            let mut nb = 0.0_f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                let dx = x - mean_a;
                let dy = y - mean_b;
                dot += dx * dy;
                na += dx * dx;
                nb += dy * dy;
            }
            if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                let r = dot / (na.sqrt() * nb.sqrt());
                (1.0 - r).max(0.0)
            }
        }
        DistanceMetric::Tanimoto => {
            let mut min_sum = 0.0_f32;
            let mut max_sum = 0.0_f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                if x < y {
                    min_sum += x;
                    max_sum += y;
                } else {
                    min_sum += y;
                    max_sum += x;
                }
            }
            if max_sum > 0.0 {
                1.0 - min_sum / max_sum
            } else {
                0.0
            }
        }
        DistanceMetric::Minkowski(p) => {
            let mut s = 0.0_f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                s += (x - y).abs().powf(p);
            }
            s.powf(1.0 / p)
        }
        DistanceMetric::Canberra => {
            let mut s = 0.0_f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                let denom = x.abs() + y.abs();
                if denom > 0.0 {
                    s += (x - y).abs() / denom;
                }
            }
            s
        }
        DistanceMetric::BrayCurtis => {
            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for (&x, &y) in a.iter().zip(b.iter()) {
                num += (x - y).abs();
                den += (x + y).abs();
            }
            if den > 0.0 { num / den } else { 0.0 }
        }
        // Bit / hash metrics aren't valid for f32 input — caller's bug.
        DistanceMetric::Jaccard
        | DistanceMetric::Dice
        | DistanceMetric::Hamming
        | DistanceMetric::HashJaccard => panic!(
            "metric {metric:?} is not defined for f32/f64 inputs"
        ),
    }
}

#[inline]
fn dist_f64(a: ArrayView1<f64>, b: ArrayView1<f64>, metric: DistanceMetric) -> f32 {
    // Same body as dist_f32 but in f64 internally; cast to f32 at the end.
    let d = match metric {
        DistanceMetric::Euclidean => {
            let mut s = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                s += d * d;
            }
            s.sqrt()
        }
        DistanceMetric::SquaredEuclidean => {
            let mut s = 0.0_f64;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                s += d * d;
            }
            s
        }
        DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum(),
        DistanceMetric::Chebyshev => a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max),
        DistanceMetric::Cosine => {
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                (1.0 - dot / (na.sqrt() * nb.sqrt())).max(0.0)
            }
        }
        DistanceMetric::Correlation => {
            let n = a.len() as f64;
            let mean_a = a.iter().sum::<f64>() / n;
            let mean_b = b.iter().sum::<f64>() / n;
            let mut dot = 0.0_f64;
            let mut na = 0.0_f64;
            let mut nb = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                let dx = x - mean_a;
                let dy = y - mean_b;
                dot += dx * dy;
                na += dx * dx;
                nb += dy * dy;
            }
            if na == 0.0 || nb == 0.0 {
                1.0
            } else {
                (1.0 - dot / (na.sqrt() * nb.sqrt())).max(0.0)
            }
        }
        DistanceMetric::Tanimoto => {
            let mut min_sum = 0.0_f64;
            let mut max_sum = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                if x < y {
                    min_sum += x;
                    max_sum += y;
                } else {
                    min_sum += y;
                    max_sum += x;
                }
            }
            if max_sum > 0.0 {
                1.0 - min_sum / max_sum
            } else {
                0.0
            }
        }
        DistanceMetric::Minkowski(p) => {
            let p = p as f64;
            let mut s = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                s += (x - y).abs().powf(p);
            }
            s.powf(1.0 / p)
        }
        DistanceMetric::Canberra => {
            let mut s = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                let denom = x.abs() + y.abs();
                if denom > 0.0 {
                    s += (x - y).abs() / denom;
                }
            }
            s
        }
        DistanceMetric::BrayCurtis => {
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for (&x, &y) in a.iter().zip(b.iter()) {
                num += (x - y).abs();
                den += (x + y).abs();
            }
            if den > 0.0 { num / den } else { 0.0 }
        }
        DistanceMetric::Jaccard
        | DistanceMetric::Dice
        | DistanceMetric::Hamming
        | DistanceMetric::HashJaccard => panic!(
            "metric {metric:?} is not defined for f32/f64 inputs"
        ),
    };
    d as f32
}

// ── Bit-packed distance ─────────────────────────────────────────────────────

struct PackedBits {
    data: Vec<u64>,
    counts: Vec<u32>,
    n_rows: usize,
    n_bits: usize,
    words_per_row: usize,
}

impl PackedBits {
    fn from_view(view: ArrayView2<bool>) -> Self {
        let n_rows = view.nrows();
        let n_bits = view.ncols();
        let words_per_row = n_bits.div_ceil(64);
        let mut data = vec![0u64; n_rows * words_per_row];
        let mut counts = vec![0u32; n_rows];
        for i in 0..n_rows {
            let base = i * words_per_row;
            let row = view.row(i);
            for (j, &bit) in row.iter().enumerate() {
                if bit {
                    data[base + j / 64] |= 1u64 << (j % 64);
                    counts[i] += 1;
                }
            }
        }
        Self {
            data,
            counts,
            n_rows,
            n_bits,
            words_per_row,
        }
    }

    #[inline]
    fn row(&self, i: usize) -> &[u64] {
        let base = i * self.words_per_row;
        &self.data[base..base + self.words_per_row]
    }
}

#[inline]
fn intersection_count(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x & y).count_ones()).sum()
}

#[inline]
fn xor_count(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

#[inline]
fn dist_bits(packed: &PackedBits, i: usize, j: usize, metric: DistanceMetric) -> f32 {
    let a = packed.row(i);
    let b = packed.row(j);
    match metric {
        DistanceMetric::Jaccard => {
            let c = intersection_count(a, b) as f32;
            let union = packed.counts[i] as f32 + packed.counts[j] as f32 - c;
            if union > 0.0 {
                (1.0 - c / union).max(0.0)
            } else {
                0.0
            }
        }
        DistanceMetric::Dice => {
            let c = intersection_count(a, b) as f32;
            let s = packed.counts[i] as f32 + packed.counts[j] as f32;
            if s > 0.0 {
                (1.0 - 2.0 * c / s).max(0.0)
            } else {
                0.0
            }
        }
        DistanceMetric::Hamming => {
            let diffs = xor_count(a, b) as f32;
            diffs / packed.n_bits as f32
        }
        DistanceMetric::Euclidean => {
            // sqrt(Hamming count) — useful when caller treats bits as 0/1 vectors.
            let diffs = xor_count(a, b) as f32;
            diffs.sqrt()
        }
        DistanceMetric::SquaredEuclidean => xor_count(a, b) as f32,
        DistanceMetric::Manhattan => xor_count(a, b) as f32,
        _ => panic!("metric {metric:?} is not defined for bool inputs"),
    }
}

// ── u32 hash signature ──────────────────────────────────────────────────────

#[inline]
fn dist_u32(a: ArrayView1<u32>, b: ArrayView1<u32>, metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::HashJaccard => {
            let n = a.len();
            let mut eq = 0u32;
            for (x, y) in a.iter().zip(b.iter()) {
                if x == y {
                    eq += 1;
                }
            }
            (1.0 - eq as f32 / n as f32).max(0.0)
        }
        _ => panic!("metric {metric:?} is not defined for u32 inputs"),
    }
}

// ── Generic top-k select via fixed-size max-heap ────────────────────────────

fn top_k<F>(n: usize, k: usize, self_idx: usize, dist_fn: F) -> Vec<(f32, u32)>
where
    F: Fn(usize) -> f32,
{
    let mut heap: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::with_capacity(k + 1);
    for j in 0..n {
        if j == self_idx {
            continue;
        }
        let d = dist_fn(j);
        if heap.len() < k {
            heap.push((OrderedFloat(d), j as u32));
        } else if d < heap.peek().unwrap().0.into_inner() {
            heap.pop();
            heap.push((OrderedFloat(d), j as u32));
        }
    }
    let mut sorted: Vec<(f32, u32)> = heap
        .into_iter()
        .map(|(of, j)| (of.into_inner(), j))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    sorted
}

fn assemble(rows: Vec<Vec<(f32, u32)>>, n: usize, k: usize) -> (Array2<u32>, Array2<f32>) {
    let mut idx = Array2::<u32>::zeros((n, k));
    let mut dist = Array2::<f32>::zeros((n, k));
    for (i, row) in rows.into_iter().enumerate() {
        for (j, (d, ix)) in row.into_iter().enumerate() {
            idx[[i, j]] = ix;
            dist[[i, j]] = d;
        }
    }
    (idx, dist)
}

// ── Public API ──────────────────────────────────────────────────────────────

/// KNN over a dense `f32` matrix.
pub fn knn_dense_f32(
    data: ArrayView2<f32>,
    k: usize,
    metric: DistanceMetric,
) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();
    assert!(k > 0 && k < n, "k must be in [1, n-1]");
    let rows: Vec<Vec<(f32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row_i = data.row(i);
            top_k(n, k, i, |j| dist_f32(row_i, data.row(j), metric))
        })
        .collect();
    assemble(rows, n, k)
}

/// KNN over a dense `f64` matrix. Distances are returned as `f32`.
pub fn knn_dense_f64(
    data: ArrayView2<f64>,
    k: usize,
    metric: DistanceMetric,
) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();
    assert!(k > 0 && k < n, "k must be in [1, n-1]");
    let rows: Vec<Vec<(f32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row_i = data.row(i);
            top_k(n, k, i, |j| dist_f64(row_i, data.row(j), metric))
        })
        .collect();
    assemble(rows, n, k)
}

/// KNN over a dense `bool` matrix (bit fingerprints).
pub fn knn_dense_bool(
    data: ArrayView2<bool>,
    k: usize,
    metric: DistanceMetric,
) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();
    assert!(k > 0 && k < n, "k must be in [1, n-1]");
    let packed = PackedBits::from_view(data);
    let rows: Vec<Vec<(f32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| top_k(n, k, i, |j| dist_bits(&packed, i, j, metric)))
        .collect();
    assemble(rows, n, k)
}

/// KNN over a dense `u32` matrix (MinHash signatures).
pub fn knn_dense_u32(
    data: ArrayView2<u32>,
    k: usize,
    metric: DistanceMetric,
) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();
    assert!(k > 0 && k < n, "k must be in [1, n-1]");
    let rows: Vec<Vec<(f32, u32)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row_i = data.row(i);
            top_k(n, k, i, |j| dist_u32(row_i, data.row(j), metric))
        })
        .collect();
    assemble(rows, n, k)
}
