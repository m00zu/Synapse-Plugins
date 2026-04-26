use numpy::IntoPyArray;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use umap_rs::Metric; // trait needed for .metric_type()
use umap_rs::{
    DistanceMetric, EuclideanMetric, FittedUmap, GraphParams, LearnedManifold, ManifoldParams,
    Optimizer, OptimizationParams, Umap, UmapConfig,
    knn_dense_bool, knn_dense_f32, knn_dense_f64, knn_dense_u32,
};

// ---------------------------------------------------------------------------
// Helper: build UmapConfig from Python kwargs
// ---------------------------------------------------------------------------

fn build_config(
    n_components: usize,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    set_op_mix_ratio: f32,
    local_connectivity: f32,
    repulsion_strength: f32,
    negative_sample_rate: usize,
    learning_rate: f32,
    n_epochs: Option<usize>,
) -> UmapConfig {
    UmapConfig {
        n_components,
        manifold: ManifoldParams {
            min_dist,
            spread,
            a: None,
            b: None,
        },
        graph: GraphParams {
            n_neighbors,
            local_connectivity,
            set_op_mix_ratio,
            disconnection_distance: None,
            symmetrize: true,
        },
        optimization: OptimizationParams {
            n_epochs,
            learning_rate,
            negative_sample_rate,
            repulsion_strength,
        },
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust helpers (no Python token needed, run inside py.detach())
// ---------------------------------------------------------------------------

fn fit_inner(
    config: UmapConfig,
    data: numpy::ndarray::Array2<f32>,
    knn_indices: numpy::ndarray::Array2<u32>,
    knn_dists: numpy::ndarray::Array2<f32>,
    init: numpy::ndarray::Array2<f32>,
) -> (FittedUmap, numpy::ndarray::Array2<f32>) {
    let umap = Umap::new(config);
    let fitted = umap.fit(
        data.view(),
        knn_indices.view(),
        knn_dists.view(),
        init.view(),
    );
    let embedding = fitted.embedding().to_owned();
    (fitted, embedding)
}

fn learn_manifold_inner(
    config: UmapConfig,
    data: numpy::ndarray::Array2<f32>,
    knn_indices: numpy::ndarray::Array2<u32>,
    knn_dists: numpy::ndarray::Array2<f32>,
) -> LearnedManifold {
    let umap = Umap::new(config);
    umap.learn_manifold(data.view(), knn_indices.view(), knn_dists.view())
}

fn optimize_inner(
    config: UmapConfig,
    manifold: LearnedManifold,
    init: numpy::ndarray::Array2<f32>,
    total_epochs: usize,
) -> numpy::ndarray::Array2<f32> {
    let metric = EuclideanMetric;
    let mut optimizer = Optimizer::new(
        manifold,
        init,
        total_epochs,
        &config,
        metric.metric_type(),
    );
    optimizer.step_epochs(total_epochs, &metric);
    let fitted = optimizer.into_fitted(config);
    fitted.into_embedding()
}

// ---------------------------------------------------------------------------
// PyUMAP — class-based API with step control
// ---------------------------------------------------------------------------

/// UMAP dimensionality reduction (Rust backend).
///
/// Example
/// -------
/// >>> model = UMAP(n_neighbors=15, min_dist=0.1)
/// >>> embedding = model.fit(data, knn_indices, knn_dists, init)
#[pyclass(name = "UMAP")]
struct PyUMAP {
    config: UmapConfig,
    manifold: Option<LearnedManifold>,
    fitted: Option<FittedUmap>,
}

#[pymethods]
impl PyUMAP {
    #[new]
    #[pyo3(signature = (
        n_components = 2,
        n_neighbors = 15,
        min_dist = 0.1,
        spread = 1.0,
        set_op_mix_ratio = 1.0,
        local_connectivity = 1.0,
        repulsion_strength = 1.0,
        negative_sample_rate = 5,
        learning_rate = 1.0,
        n_epochs = None,
    ))]
    fn new(
        n_components: usize,
        n_neighbors: usize,
        min_dist: f32,
        spread: f32,
        set_op_mix_ratio: f32,
        local_connectivity: f32,
        repulsion_strength: f32,
        negative_sample_rate: usize,
        learning_rate: f32,
        n_epochs: Option<usize>,
    ) -> Self {
        let config = build_config(
            n_components,
            n_neighbors,
            min_dist,
            spread,
            set_op_mix_ratio,
            local_connectivity,
            repulsion_strength,
            negative_sample_rate,
            learning_rate,
            n_epochs,
        );
        PyUMAP {
            config,
            manifold: None,
            fitted: None,
        }
    }

    /// Learn manifold + optimize embedding in one call.
    ///
    /// Parameters
    /// ----------
    /// data : np.ndarray[f32]  (n_samples, n_features)
    /// knn_indices : np.ndarray[u32]  (n_samples, n_neighbors)
    /// knn_dists : np.ndarray[f32]  (n_samples, n_neighbors)
    /// init : np.ndarray[f32]  (n_samples, n_components)
    ///
    /// Returns
    /// -------
    /// np.ndarray[f32]  (n_samples, n_components)
    fn fit<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        knn_indices: PyReadonlyArray2<'py, u32>,
        knn_dists: PyReadonlyArray2<'py, f32>,
        init: PyReadonlyArray2<'py, f32>,
    ) -> Bound<'py, PyArray2<f32>> {
        let data_arr = data.as_array().to_owned();
        let knn_idx_arr = knn_indices.as_array().to_owned();
        let knn_dist_arr = knn_dists.as_array().to_owned();
        let init_arr = init.as_array().to_owned();
        let config = self.config.clone();

        // Release GIL for heavy Rust computation
        let (fitted, embedding) = py.detach(move || {
            fit_inner(config, data_arr, knn_idx_arr, knn_dist_arr, init_arr)
        });

        self.fitted = Some(fitted);
        embedding.into_pyarray(py)
    }

    /// Learn manifold structure only (Phase 1).
    fn learn_manifold<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        knn_indices: PyReadonlyArray2<'py, u32>,
        knn_dists: PyReadonlyArray2<'py, f32>,
    ) {
        let data_arr = data.as_array().to_owned();
        let knn_idx_arr = knn_indices.as_array().to_owned();
        let knn_dist_arr = knn_dists.as_array().to_owned();
        let config = self.config.clone();

        let manifold = py.detach(move || {
            learn_manifold_inner(config, data_arr, knn_idx_arr, knn_dist_arr)
        });

        self.manifold = Some(manifold);
    }

    /// Optimize embedding from a previously learned manifold (Phase 2).
    ///
    /// Must call `learn_manifold()` first.
    fn optimize<'py>(
        &mut self,
        py: Python<'py>,
        init: PyReadonlyArray2<'py, f32>,
        n_epochs: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let manifold = self.manifold.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "No manifold learned yet — call learn_manifold() first",
            )
        })?;

        let init_owned = init.as_array().to_owned();
        let n_samples = init_owned.shape()[0];
        let total_epochs = n_epochs
            .or(self.config.optimization.n_epochs)
            .unwrap_or(if n_samples <= 10000 { 500 } else { 200 });

        let config = self.config.clone();
        let embedding = py.detach(move || {
            optimize_inner(config, manifold, init_owned, total_epochs)
        });

        Ok(embedding.into_pyarray(py))
    }

    /// Get the embedding from the last fit() call.
    fn get_embedding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No fitted model — call fit() first")
        })?;
        Ok(fitted.embedding().to_owned().into_pyarray(py))
    }
}

// ---------------------------------------------------------------------------
// Standalone function API
// ---------------------------------------------------------------------------

/// Fit UMAP in one call (convenience function).
///
/// Parameters
/// ----------
/// data : np.ndarray[f32]  (n_samples, n_features)
/// knn_indices : np.ndarray[u32]  (n_samples, n_neighbors)
/// knn_dists : np.ndarray[f32]  (n_samples, n_neighbors)
/// init : np.ndarray[f32]  (n_samples, n_components)
/// n_components : int, default 2
/// n_neighbors : int, default 15
/// min_dist : float, default 0.1
/// spread : float, default 1.0
/// n_epochs : int or None
/// learning_rate : float, default 1.0
/// negative_sample_rate : int, default 5
/// repulsion_strength : float, default 1.0
///
/// Returns
/// -------
/// np.ndarray[f32]  (n_samples, n_components)
#[pyfunction]
#[pyo3(signature = (
    data,
    knn_indices,
    knn_dists,
    init,
    n_components = 2,
    n_neighbors = 15,
    min_dist = 0.1,
    spread = 1.0,
    n_epochs = None,
    learning_rate = 1.0,
    negative_sample_rate = 5,
    repulsion_strength = 1.0,
))]
fn fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    knn_indices: PyReadonlyArray2<'py, u32>,
    knn_dists: PyReadonlyArray2<'py, f32>,
    init: PyReadonlyArray2<'py, f32>,
    n_components: usize,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    learning_rate: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
) -> Bound<'py, PyArray2<f32>> {
    let config = build_config(
        n_components,
        n_neighbors,
        min_dist,
        spread,
        1.0,
        1.0,
        repulsion_strength,
        negative_sample_rate,
        learning_rate,
        n_epochs,
    );

    let data_arr = data.as_array().to_owned();
    let knn_idx_arr = knn_indices.as_array().to_owned();
    let knn_dist_arr = knn_dists.as_array().to_owned();
    let init_arr = init.as_array().to_owned();

    let (_, embedding) = py.detach(move || {
        fit_inner(config, data_arr, knn_idx_arr, knn_dist_arr, init_arr)
    });

    embedding.into_pyarray(py)
}

// ---------------------------------------------------------------------------
// KNN bindings
// ---------------------------------------------------------------------------

fn parse_metric(name: &str, p: f32) -> PyResult<DistanceMetric> {
    let m = DistanceMetric::parse(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown metric '{name}'. Valid: euclidean, sqeuclidean, manhattan, \
             chebyshev, cosine, correlation, tanimoto, minkowski, canberra, \
             braycurtis, jaccard, dice, hamming, hash_jaccard"
        ))
    })?;
    // The ``p`` kwarg only matters for Minkowski; override its default p=2.
    Ok(match m {
        DistanceMetric::Minkowski(_) => DistanceMetric::Minkowski(p),
        other => other,
    })
}

/// Brute-force parallel k-NN over a dense ``f32`` matrix.
///
/// Parameters
/// ----------
/// data : np.ndarray[f32]  (n_samples, n_features)
/// k    : int — number of neighbours per row (excluding self).
/// metric : str — 'euclidean', 'sqeuclidean', 'manhattan', 'chebyshev',
///                'cosine', 'correlation', 'tanimoto'.
///
/// Returns
/// -------
/// (indices: np.ndarray[u32]  (n_samples, k),
///  distances: np.ndarray[f32]  (n_samples, k))
#[pyfunction]
#[pyo3(signature = (data, k, metric = "euclidean", p = 2.0))]
fn knn_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    k: usize,
    metric: &str,
    p: f32,
) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
    let m = parse_metric(metric, p)?;
    let arr = data.as_array().to_owned();
    let (idx, dist) = py.detach(move || knn_dense_f32(arr.view(), k, m));
    Ok((idx.into_pyarray(py), dist.into_pyarray(py)))
}

/// Brute-force parallel k-NN over a dense ``f64`` matrix.
/// Distances are returned as ``f32`` for consistency with the rest of the API.
#[pyfunction]
#[pyo3(signature = (data, k, metric = "euclidean", p = 2.0))]
fn knn_f64<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
    metric: &str,
    p: f32,
) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
    let m = parse_metric(metric, p)?;
    let arr = data.as_array().to_owned();
    let (idx, dist) = py.detach(move || knn_dense_f64(arr.view(), k, m));
    Ok((idx.into_pyarray(py), dist.into_pyarray(py)))
}

/// Brute-force parallel k-NN over a dense ``bool`` matrix (bit fingerprints).
///
/// Metric must be 'jaccard', 'dice', 'hamming', 'euclidean' (= sqrt-Hamming),
/// 'sqeuclidean' (= Hamming count), or 'manhattan' (= Hamming count).
#[pyfunction]
#[pyo3(signature = (data, k, metric = "jaccard", p = 2.0))]
fn knn_bool<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, bool>,
    k: usize,
    metric: &str,
    p: f32,
) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
    let m = parse_metric(metric, p)?;
    let arr = data.as_array().to_owned();
    let (idx, dist) = py.detach(move || knn_dense_bool(arr.view(), k, m));
    Ok((idx.into_pyarray(py), dist.into_pyarray(py)))
}

/// Brute-force parallel k-NN over a dense ``uint32`` matrix (MinHash signatures).
///
/// Metric must be 'hash_jaccard'.
#[pyfunction]
#[pyo3(signature = (data, k, metric = "hash_jaccard", p = 2.0))]
fn knn_u32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, u32>,
    k: usize,
    metric: &str,
    p: f32,
) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
    let m = parse_metric(metric, p)?;
    let arr = data.as_array().to_owned();
    let (idx, dist) = py.detach(move || knn_dense_u32(arr.view(), k, m));
    Ok((idx.into_pyarray(py), dist.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

#[pymodule]
fn umap_rs_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUMAP>()?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(knn_f32, m)?)?;
    m.add_function(wrap_pyfunction!(knn_f64, m)?)?;
    m.add_function(wrap_pyfunction!(knn_bool, m)?)?;
    m.add_function(wrap_pyfunction!(knn_u32, m)?)?;
    Ok(())
}
