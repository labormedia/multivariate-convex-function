use nalgebra::{DMatrix, DVector, RowDVector};
use rand::random;

/// Convert alpha_ij to r_ij = (1 - alpha_ij) / alpha_ij, ensuring 0 < alpha < 1.
fn alpha_to_log_ratio(alpha: &DMatrix<f64>) -> Result<DMatrix<f64>, String> {
    let n = alpha.nrows();
    let mut log_r = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let a = alpha[(i, j)];
                if !(0.0 < a && a < 1.0) {
                    return Err(format!("alpha[{i},{j}] = {a} not in (0,1)"));
                }
                let rij = (1.0 - a) / a;
                log_r[(i, j)] = rij.ln();
            }
        }
    }
    Ok(log_r)
}

/// Assemble the linear system A x = b for x = log(beta)
/// Each equation: log(beta_j) - log(beta_i) = log(r_ij)
fn build_equation_system(log_r: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let n = log_r.nrows();
    let mut rows = Vec::new();
    let mut rhs = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut row = vec![0.0; n];
                row[i] = -1.0;
                row[j] = 1.0;
                rows.push(row);
                rhs.push(log_r[(i, j)]);
            }
        }
    }

    // Convert Vec<Vec<f64>> → DMatrix using RowDVector conversion
    let a = DMatrix::from_rows(
        &rows
            .iter()
            .map(|r| RowDVector::from_row_slice(r))
            .collect::<Vec<_>>(),
    );
    let b = DVector::from_vec(rhs);

    (a, b)
}

/// Infer beta from alpha using least-squares over log(beta) differences.
/// Enforces normalization so sum(beta) = 1.
fn infer_beta_from_alpha(alpha: &DMatrix<f64>) -> Result<(DVector<f64>, f64), String> {
    let n = alpha.nrows();
    if n != alpha.ncols() {
        return Err("Alpha matrix must be square".into());
    }

    // Check α_ij + α_ji = 1
    for i in 0..n {
        for j in 0..n {
            if i != j {
                if (alpha[(i, j)] + alpha[(j, i)] - 1.0).abs() > 1e-8 {
                    return Err(format!("alpha[{},{}] + alpha[{},{}] != 1", i, j, j, i));
                }
            }
        }
    }

    // Convert alpha → log ratio
    let log_r = alpha_to_log_ratio(alpha)?;

    // Build A x = b
    let (a, b) = build_equation_system(&log_r);

    // Solve A x ≈ b via least squares using SVD
    let svd = a.clone().svd(true, true);
    let x_opt = svd.solve(&b, 1e-10);

    let x = match x_opt {
        Ok(sol) => sol,
        Err(e) => return Err(format!("SVD solve failed (ill-conditioned system), {}", e)),
    };

    // Make log(beta) all positive
    let min_x = x.min();
    let shifted_x = &x - DVector::from_element(n, min_x);

    // Exponentiate
    let mut beta = shifted_x.map(|v| v.exp());

    // Normalize to simplex
    let sum_beta: f64 = beta.iter().sum();
    beta.iter_mut().for_each(|b| *b /= sum_beta);

    // Compute residual
    let residual = (&a * &x - b).norm_squared();

    Ok((beta, residual))
}

/// Convert global beta → pairwise α_ij = β_i / (β_i + β_j)
fn beta_to_alpha(beta: &DVector<f64>) -> DMatrix<f64> {
    let n = beta.len();
    let mut alpha = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            if i == j {
                alpha[(i, j)] = 0.0;
            } else {
                alpha[(i, j)] = beta[i] / (beta[i] + beta[j]);
            }
        }
    }
    alpha
}

/// Generate a random β vector on the simplex
fn random_beta(n: usize) -> DVector<f64> {
    let mut beta = DVector::from_iterator(n, (0..n).map(|_| random::<f64>() + 1e-6));
    let sum: f64 = beta.iter().sum();
    beta.iter_mut().for_each(|b| *b /= sum);
    beta
}

/// Test harness: generate random β → α → recover β' and compare
fn test_recovery(trials: usize, n: usize) {
    println!("Running {} randomized β → α → β̂ tests (n = {})...", trials, n);

    for t in 0..trials {
        let beta = random_beta(n);
        let alpha = beta_to_alpha(&beta);

        let (beta_hat, residual) = match infer_beta_from_alpha(&alpha) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Error in trial {}: {}", t, e);
                continue;
            }
        };

        // Compare beta_hat to original beta
        let dist = (&beta_hat - &beta).norm();

        println!(
            "Trial {:3}: ||β̂ - β|| = {:.6e}, residual = {:.6e}",
            t, dist, residual
        );
    }
}

fn main() {
    // Example demonstration with known matrix
    let alpha_data = vec![
        0.0,   0.6,    0.4,
        0.4,   0.0,    0.3077,
        0.6, 0.6923,   0.0
    ];

    let alpha = DMatrix::from_row_slice(3, 3, &alpha_data);

    println!("Inferring β from example α:");
    match infer_beta_from_alpha(&alpha) {
        Ok((beta, res)) => {
            println!("β recovered: {}", beta.transpose());
            println!("Residual: {:.3e}", res);
        }
        Err(e) => println!("Error: {}", e),
    }

    // Run the randomized test harness
    test_recovery(10, 5);  // 10 trials, n = 5 goods
}