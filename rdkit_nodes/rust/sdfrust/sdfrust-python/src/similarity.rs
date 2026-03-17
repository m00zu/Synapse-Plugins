//! Fast pairwise similarity computation for molecular fingerprints.
//!
//! Implements 10 similarity metrics operating on packed bit vectors with
//! hardware popcount.  Uses rayon for parallel row computation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};

// ── Packed bit-vector representation ─────────────────────────────────────────

/// Flat contiguous storage of u64 words — one row per molecule.
/// All fingerprints are stored in a single Vec for cache-friendly access.
struct PackedFps {
    data: Vec<u64>,   // flat: mol i starts at i * words_per_mol
    counts: Vec<u32>, // per-molecule on-bit counts
    n_mols: usize,
    n_bits: usize,
    words_per_mol: usize,
}

impl PackedFps {
    /// Get the packed words for molecule `i`.
    #[inline]
    fn row(&self, i: usize) -> &[u64] {
        let start = i * self.words_per_mol;
        &self.data[start..start + self.words_per_mol]
    }
}

#[cfg(feature = "numpy")]
impl PackedFps {
    fn from_numpy(fingerprints: &numpy::PyReadonlyArray2<'_, bool>) -> Self {
        let shape = fingerprints.shape();
        let n_mols = shape[0];
        let n_bits = shape[1];
        let words_per_mol = (n_bits + 63) / 64;
        let arr = fingerprints.as_array();

        let mut data: Vec<u64> = vec![0u64; n_mols * words_per_mol];
        let mut counts: Vec<u32> = vec![0u32; n_mols];

        for i in 0..n_mols {
            let base = i * words_per_mol;
            let row = arr.row(i);
            for (j, &bit) in row.iter().enumerate() {
                if bit {
                    data[base + j / 64] |= 1u64 << (j % 64);
                    counts[i] += 1;
                }
            }
        }

        PackedFps {
            data,
            counts,
            n_mols,
            n_bits,
            words_per_mol,
        }
    }
}

/// AND + popcount across all u64 words.
#[inline]
fn intersection_count(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x & y).count_ones())
        .sum()
}

// ── Metric computation ───────────────────────────────────────────────────────

/// Normalised metric name (lowercase, no hyphens/underscores/spaces).
fn normalise_metric(raw: &str) -> String {
    raw.to_ascii_lowercase()
        .replace(['-', '_', ' '], "")
}

/// Compute a single similarity value.
///
/// `a` = |A|, `b` = |B|, `c` = |A ∩ B|, `n` = total bits.
#[inline]
fn compute_sim(metric: &str, a: f64, b: f64, c: f64, n: f64, alpha: f64, beta: f64) -> f64 {
    match metric {
        "tanimoto" => {
            let d = a + b - c;
            if d == 0.0 { 0.0 } else { c / d }
        }
        "dice" => {
            let d = a + b;
            if d == 0.0 { 0.0 } else { 2.0 * c / d }
        }
        "braunblanquet" => {
            let d = a.max(b);
            if d == 0.0 { 0.0 } else { c / d }
        }
        "cosine" => {
            let d = (a * b).sqrt();
            if d == 0.0 { 0.0 } else { c / d }
        }
        "kulczynski" => {
            if a == 0.0 || b == 0.0 {
                0.0
            } else {
                0.5 * c * (1.0 / a + 1.0 / b)
            }
        }
        "mcconnaughey" => {
            let d = a * b;
            if d == 0.0 {
                0.0
            } else {
                (c * c - (a - c) * (b - c)) / d
            }
        }
        "rogotgoldberg" => {
            let d = n - a - b + c; // bits in neither
            let t1 = if a + b == 0.0 { 0.0 } else { c / (a + b) };
            let t2 = {
                let denom = 2.0 * n - a - b;
                if denom == 0.0 { 0.0 } else { d / denom }
            };
            t1 + t2
        }
        "russel" => {
            if n == 0.0 { 0.0 } else { c / n }
        }
        "sokal" => {
            let d = 2.0 * a + 2.0 * b - 3.0 * c;
            if d == 0.0 { 0.0 } else { c / d }
        }
        "tversky" => {
            let d = alpha * (a - c) + beta * (b - c) + c;
            if d == 0.0 { 0.0 } else { c / d }
        }
        _ => 0.0,
    }
}

const VALID_METRICS: &[&str] = &[
    "tanimoto",
    "dice",
    "braunblanquet",
    "cosine",
    "kulczynski",
    "mcconnaughey",
    "rogotgoldberg",
    "russel",
    "sokal",
    "tversky",
];

// ── Python-exposed functions ─────────────────────────────────────────────────

/// Compute an NxN pairwise similarity matrix from a fingerprint boolean matrix.
///
/// Args:
///     fingerprints: 2-D boolean numpy array of shape ``(N, n_bits)``.
///     metric:       One of ``"tanimoto"``, ``"dice"``, ``"braun-blanquet"``,
///                   ``"cosine"``, ``"kulczynski"``, ``"mcconnaughey"``,
///                   ``"rogot-goldberg"``, ``"russel"``, ``"sokal"``,
///                   ``"tversky"``.
///     alpha:        α for Tversky (default 0.5).
///     beta:         β for Tversky (default 0.5).
///
/// Returns:
///     2-D float64 numpy array of shape ``(N, N)``.
#[cfg(feature = "numpy")]
#[pyfunction]
#[pyo3(
    name = "pairwise_similarity",
    signature = (fingerprints, metric="tanimoto", alpha=0.5, beta=0.5)
)]
pub fn py_pairwise_similarity<'py>(
    py: Python<'py>,
    fingerprints: &Bound<'py, PyArray2<bool>>,
    metric: &str,
    alpha: f64,
    beta: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let metric_norm = normalise_metric(metric);
    if !VALID_METRICS.contains(&metric_norm.as_str()) {
        return Err(PyValueError::new_err(format!(
            "Unknown metric '{metric}'. Valid: tanimoto, dice, braun-blanquet, cosine, \
             kulczynski, mcconnaughey, rogot-goldberg, russel, sokal, tversky"
        )));
    }

    // Pack bits (GIL held — reading numpy)
    let packed = {
        let readonly = fingerprints.readonly();
        PackedFps::from_numpy(&readonly)
    };
    let n = packed.n_mols;
    let nb = packed.n_bits as f64;

    // Heavy computation — release GIL.
    let result: Vec<Vec<f64>> = py.allow_threads(|| {
        #[cfg(feature = "rayon")]
        let result: Vec<Vec<f64>> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let a = packed.counts[i] as f64;
                    (0..n)
                        .map(|j| {
                            if i == j {
                                1.0
                            } else {
                                let b = packed.counts[j] as f64;
                                let c =
                                    intersection_count(packed.row(i), packed.row(j)) as f64;
                                compute_sim(&metric_norm, a, b, c, nb, alpha, beta)
                            }
                        })
                        .collect()
                })
                .collect()
        };

        #[cfg(not(feature = "rayon"))]
        let result: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let a = packed.counts[i] as f64;
                (0..n)
                    .map(|j| {
                        if i == j {
                            1.0
                        } else {
                            let b = packed.counts[j] as f64;
                            let c = intersection_count(packed.row(i), packed.row(j)) as f64;
                            compute_sim(&metric_norm, a, b, c, nb, alpha, beta)
                        }
                    })
                    .collect()
            })
            .collect();

        result
    });

    PyArray2::from_vec2(py, &result).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Taylor–Butina clustering on a precomputed similarity matrix.
///
/// Args:
///     similarity_matrix: 2-D float64 numpy array ``(N, N)``.
///     threshold:         Similarity threshold for cluster membership.
///
/// Returns:
///     1-D int32 numpy array of cluster labels (length N).
#[cfg(feature = "numpy")]
#[pyfunction]
#[pyo3(name = "butina_cluster", signature = (similarity_matrix, threshold=0.35))]
pub fn py_butina_cluster<'py>(
    py: Python<'py>,
    similarity_matrix: &Bound<'py, PyArray2<f64>>,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let readonly = similarity_matrix.readonly();
    let arr = readonly.as_array();
    let n = arr.nrows();
    if n != arr.ncols() {
        return Err(PyValueError::new_err("Similarity matrix must be square."));
    }

    let labels: Vec<i32> = py.allow_threads(|| {
        // Count neighbours above threshold for each molecule
        let neighbor_counts: Vec<usize> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i && arr[[i, j]] >= threshold).count())
            .collect();

        // Sort by decreasing neighbour count (most-connected first)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| neighbor_counts[b].cmp(&neighbor_counts[a]));

        let mut labels = vec![-1i32; n];
        let mut cluster_id: i32 = 0;
        for &idx in &order {
            if labels[idx] >= 0 {
                continue;
            }
            labels[idx] = cluster_id;
            for j in 0..n {
                if labels[j] < 0 && arr[[idx, j]] >= threshold {
                    labels[j] = cluster_id;
                }
            }
            cluster_id += 1;
        }
        labels
    });

    Ok(PyArray1::from_vec(py, labels))
}

// ── Compact lower-triangle helpers ──────────────────────────────────────────

/// Index into a compact lower-triangle array for pair (i, j) where i > j.
#[inline]
fn tri_idx(i: usize, j: usize) -> usize {
    debug_assert!(i > j);
    i * (i - 1) / 2 + j
}

/// Similarity from packed fingerprints using any supported metric.
#[inline]
fn fp_sim(
    a: &[u64], b: &[u64],
    count_a: u32, count_b: u32,
    n_bits: f64, metric: &str, alpha: f64, beta: f64,
) -> f64 {
    let c = intersection_count(a, b) as f64;
    compute_sim(metric, count_a as f64, count_b as f64, c, n_bits, alpha, beta)
}

/// Taylor–Butina clustering on a compact lower-triangle similarity array.
///
/// The triangle stores ``sim(i, j)`` for ``i > j`` at index
/// ``i*(i-1)/2 + j``, with total length ``N*(N-1)/2``.
/// This uses half the memory of the full NxN matrix.
///
/// Args:
///     sim_triangle: 1-D float64 numpy array of length ``N*(N-1)/2``.
///     n_mols:       Number of molecules ``N``.
///     threshold:    Similarity threshold for cluster membership (default 0.35).
///
/// Returns:
///     1-D int32 numpy array of cluster labels (length N).
#[cfg(feature = "numpy")]
#[pyfunction]
#[pyo3(name = "butina_cluster_tri", signature = (sim_triangle, n_mols, threshold=0.35))]
pub fn py_butina_cluster_tri<'py>(
    py: Python<'py>,
    sim_triangle: &Bound<'py, PyArray1<f64>>,
    n_mols: usize,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let readonly = sim_triangle.readonly();
    let tri = readonly.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let expected_len = n_mols * (n_mols - 1) / 2;
    if tri.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "Expected triangle length {} for {} molecules, got {}",
            expected_len, n_mols, tri.len()
        )));
    }
    let n = n_mols;

    let labels: Vec<i32> = py.allow_threads(|| {
        // Lookup similarity from compact triangle
        let sim = |i: usize, j: usize| -> f64 {
            if i == j {
                1.0
            } else if i > j {
                tri[tri_idx(i, j)]
            } else {
                tri[tri_idx(j, i)]
            }
        };

        let neighbor_counts: Vec<usize> = (0..n)
            .map(|i| (0..n).filter(|&j| j != i && sim(i, j) >= threshold).count())
            .collect();

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| neighbor_counts[b].cmp(&neighbor_counts[a]));

        let mut labels = vec![-1i32; n];
        let mut cluster_id: i32 = 0;
        for &idx in &order {
            if labels[idx] >= 0 {
                continue;
            }
            labels[idx] = cluster_id;
            for j in 0..n {
                if labels[j] < 0 && sim(idx, j) >= threshold {
                    labels[j] = cluster_id;
                }
            }
            cluster_id += 1;
        }
        labels
    });

    Ok(PyArray1::from_vec(py, labels))
}

/// Taylor–Butina clustering directly from boolean fingerprints.
///
/// Computes similarities on-the-fly — never materialises the full
/// NxN matrix.  Memory usage is O(N) instead of O(N²), allowing much larger
/// datasets.  Supports all 10 similarity metrics.
///
/// Args:
///     fingerprints: 2-D boolean numpy array of shape ``(N, n_bits)``.
///     threshold:    Similarity threshold for cluster membership (default 0.35).
///     metric:       Similarity metric (default ``"tanimoto"``).
///     alpha:        α for Tversky (default 0.5).
///     beta:         β for Tversky (default 0.5).
///
/// Returns:
///     1-D int32 numpy array of cluster labels (length N).
#[cfg(feature = "numpy")]
#[pyfunction]
#[pyo3(name = "butina_cluster_fps", signature = (fingerprints, threshold=0.35, metric="tanimoto", alpha=0.5, beta=0.5))]
pub fn py_butina_cluster_fps<'py>(
    py: Python<'py>,
    fingerprints: &Bound<'py, PyArray2<bool>>,
    threshold: f64,
    metric: &str,
    alpha: f64,
    beta: f64,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let metric_norm = normalise_metric(metric);
    if !VALID_METRICS.contains(&metric_norm.as_str()) {
        return Err(PyValueError::new_err(format!(
            "Unknown metric '{metric}'. Valid: tanimoto, dice, braun-blanquet, cosine, \
             kulczynski, mcconnaughey, rogot-goldberg, russel, sokal, tversky"
        )));
    }

    let packed = {
        let readonly = fingerprints.readonly();
        PackedFps::from_numpy(&readonly)
    };
    let n = packed.n_mols;
    let nb = packed.n_bits as f64;

    let labels: Vec<i32> = py.allow_threads(|| {
        // Phase 1: count neighbours (parallelised with rayon)
        #[cfg(feature = "rayon")]
        let neighbor_counts: Vec<usize> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut count = 0usize;
                    for j in 0..n {
                        if j != i
                            && fp_sim(
                                packed.row(i), packed.row(j),
                                packed.counts[i], packed.counts[j],
                                nb, &metric_norm, alpha, beta,
                            ) >= threshold
                        {
                            count += 1;
                        }
                    }
                    count
                })
                .collect()
        };

        #[cfg(not(feature = "rayon"))]
        let neighbor_counts: Vec<usize> = (0..n)
            .map(|i| {
                let mut count = 0usize;
                for j in 0..n {
                    if j != i
                        && fp_sim(
                            packed.row(i), packed.row(j),
                            packed.counts[i], packed.counts[j],
                            nb, &metric_norm, alpha, beta,
                        ) >= threshold
                    {
                        count += 1;
                    }
                }
                count
            })
            .collect();

        // Phase 2: greedy cluster assignment (sequential)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| neighbor_counts[b].cmp(&neighbor_counts[a]));

        let mut labels = vec![-1i32; n];
        let mut cluster_id: i32 = 0;
        for &idx in &order {
            if labels[idx] >= 0 {
                continue;
            }
            labels[idx] = cluster_id;
            for j in 0..n {
                if labels[j] < 0
                    && fp_sim(
                        packed.row(idx), packed.row(j),
                        packed.counts[idx], packed.counts[j],
                        nb, &metric_norm, alpha, beta,
                    ) >= threshold
                {
                    labels[j] = cluster_id;
                }
            }
            cluster_id += 1;
        }
        labels
    });

    Ok(PyArray1::from_vec(py, labels))
}
