use nalgebra::{DMatrix, DVector, RowDVector};
use std::f64;

fn infer_beta_from_alpha(alpha: &DMatrix<f64>) -> Result<(DVector<f64>, f64), String> {
    let n = alpha.nrows();

    // 1. Sanity checks
    if alpha.nrows() != alpha.ncols() {
        return Err("Alpha matrix must be square.".into());
    }

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sum = alpha[(i, j)] + alpha[(j, i)];
                if (sum - 1.0).abs() > 1e-8 {
                    return Err(format!("Alpha matrix not symmetric at ({}, {})", i, j));
                }
                if alpha[(i, j)] <= 0.0 || alpha[(i, j)] >= 1.0 {
                    return Err(format!("Alpha_ij values must be in (0,1), got alpha[{},{}]={}", i, j, alpha[(i, j)]));
                }
            }
        }
    }

    // 2. Build system A * x = b, where x = log(beta)
    let mut rows = Vec::new();
    let mut rhs = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut row = vec![0.0; n];
                row[i] = -1.0;
                row[j] = 1.0;
                rows.push(row);
                let alpha_ij = alpha[(i, j)];
                let r_ij = (1.0 - alpha_ij) / alpha_ij;
                rhs.push(r_ij.ln());
            }
        }
    }

    let a = DMatrix::from_rows(&rows.iter().map(|row| RowDVector::from_row_slice(&row)).collect::<Vec<_>>());
    let b = DVector::from_vec(rhs);

    // 3. Solve least squares A x ≈ b for x = log(beta)
    let svd = a.clone().svd(true, true);
    let x = match svd.solve(&b, 1e-10) {
        Ok(sol) => sol,
        Err(e) => return Err("SVD solve failed or system too ill-conditioned".into()),
    };

    // Shift log(beta) to ensure positivity
    let min_x = x.min();
    let shifted_x = &x - DVector::from_element(n, min_x);
    let mut beta = shifted_x.map(|xi| xi.exp());

    // Normalize to sum to 1
    let sum_beta: f64 = beta.iter().sum();
    beta.iter_mut().for_each(|b| *b /= sum_beta);

    // 4. Compute residual norm ||Ax - b||^2
    let residual = &a * &x - b;
    let residual_norm = residual.norm_squared();

    Ok((beta, residual_norm))
}

fn main() {
    // Example: 3 goods
    let alpha_data = vec![
        0.0,   0.6,    0.4,
        0.4,   0.0,    0.3077,
        0.6, 0.6923,   0.0
    ];
    let alpha = DMatrix::from_row_slice(3, 3, &alpha_data);

    match infer_beta_from_alpha(&alpha) {
        Ok((beta, residual)) => {
            println!("Inferred β (normalized):");
            for (i, b) in beta.iter().enumerate() {
                println!("  β[{}] = {:.6}", i+1, b);
            }
            println!("Residual norm ||Ax - b||² = {:.2e}", residual);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
