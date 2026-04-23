//! Nelder-Mead Simplex optimizer for rxDock local refinement.
//!
//! Standard Nelder-Mead with reflection, expansion, contraction, and shrink.
//! Operates on a flat f64 vector (the chromosome genes).

/// Parameters for the Nelder-Mead simplex optimizer.
#[derive(Debug, Clone)]
pub struct SimplexParams {
    /// Maximum number of function evaluations.
    pub max_calls: usize,
    /// Convergence tolerance (relative improvement in function value).
    pub convergence: f64,
    /// Reflection coefficient (default 1.0).
    pub alpha: f64,
    /// Expansion coefficient (default 2.0).
    pub gamma: f64,
    /// Contraction coefficient (default 0.5).
    pub rho: f64,
    /// Shrink coefficient (default 0.5).
    pub sigma: f64,
}

impl Default for SimplexParams {
    fn default() -> Self {
        SimplexParams {
            max_calls: 200,
            convergence: 1e-4,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Run Nelder-Mead minimization.
///
/// # Arguments
/// * `eval_fn` - Function to minimize: takes &[f64] → f64
/// * `start` - Starting point
/// * `step_sizes` - Initial simplex step sizes per dimension
/// * `params` - Optimizer parameters
///
/// # Returns
/// (best_point, best_value)
pub fn nelder_mead_minimize<F>(
    eval_fn: &mut F,
    start: &[f64],
    step_sizes: &[f64],
    params: &SimplexParams,
) -> (Vec<f64>, f64)
where
    F: FnMut(&[f64]) -> f64,
{
    let n = start.len();
    if n == 0 {
        return (start.to_vec(), eval_fn(start));
    }

    // Build initial simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(start.to_vec());

    for i in 0..n {
        let mut vertex = start.to_vec();
        vertex[i] += step_sizes[i];
        simplex.push(vertex);
    }

    // Evaluate all vertices
    let mut values: Vec<f64> = simplex.iter().map(|v| eval_fn(v)).collect();
    let mut n_calls = n + 1;

    loop {
        // Sort vertices by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

        let best_idx = order[0];
        let worst_idx = order[n];
        let second_worst_idx = order[n - 1];

        let f_best = values[best_idx];
        let f_worst = values[worst_idx];
        let f_second_worst = values[second_worst_idx];

        // Check convergence
        if n_calls >= params.max_calls {
            break;
        }
        if (f_worst - f_best).abs() < params.convergence * (f_best.abs() + 1e-10) {
            break;
        }

        // Compute centroid of all vertices except worst
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + params.alpha * (centroid[j] - simplex[worst_idx][j]))
            .collect();
        let f_reflected = eval_fn(&reflected);
        n_calls += 1;

        if f_reflected < f_second_worst && f_reflected >= f_best {
            // Accept reflection
            simplex[worst_idx] = reflected;
            values[worst_idx] = f_reflected;
            continue;
        }

        if f_reflected < f_best {
            // Try expansion
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + params.gamma * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = eval_fn(&expanded);
            n_calls += 1;

            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                values[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                values[worst_idx] = f_reflected;
            }
            continue;
        }

        // Contraction
        let contracted: Vec<f64> = if f_reflected < f_worst {
            // Outside contraction
            (0..n)
                .map(|j| centroid[j] + params.rho * (reflected[j] - centroid[j]))
                .collect()
        } else {
            // Inside contraction
            (0..n)
                .map(|j| centroid[j] + params.rho * (simplex[worst_idx][j] - centroid[j]))
                .collect()
        };
        let f_contracted = eval_fn(&contracted);
        n_calls += 1;

        if f_contracted < f_worst {
            simplex[worst_idx] = contracted;
            values[worst_idx] = f_contracted;
            continue;
        }

        // Shrink: move all vertices toward best
        for &idx in &order[1..] {
            for j in 0..n {
                simplex[idx][j] = simplex[best_idx][j] + params.sigma * (simplex[idx][j] - simplex[best_idx][j]);
            }
            values[idx] = eval_fn(&simplex[idx]);
            n_calls += 1;
        }
    }

    // Return best
    let best_idx = values.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    (simplex[best_idx].clone(), values[best_idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock_2d() {
        // Minimize Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1, 1)
        let mut eval = |x: &[f64]| -> f64 {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };

        let start = vec![0.0, 0.0];
        let steps = vec![0.5, 0.5];
        let params = SimplexParams { max_calls: 1000, convergence: 1e-8, ..Default::default() };

        let (best, val) = nelder_mead_minimize(&mut eval, &start, &steps, &params);
        assert!((best[0] - 1.0).abs() < 0.05, "x should be near 1.0, got {}", best[0]);
        assert!((best[1] - 1.0).abs() < 0.05, "y should be near 1.0, got {}", best[1]);
        assert!(val < 0.01, "Value should be near 0, got {}", val);
    }

    #[test]
    fn test_quadratic_1d() {
        // f(x) = (x - 3)^2
        let mut eval = |x: &[f64]| -> f64 { (x[0] - 3.0) * (x[0] - 3.0) };

        let (best, val) = nelder_mead_minimize(
            &mut eval,
            &[0.0],
            &[1.0],
            &SimplexParams { max_calls: 100, ..Default::default() },
        );
        assert!((best[0] - 3.0).abs() < 0.001, "x should be ~3.0, got {}", best[0]);
        assert!(val < 1e-5, "value should be ~0, got {}", val);
    }
}
