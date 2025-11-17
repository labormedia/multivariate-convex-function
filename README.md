# 🔁 Cobb–Douglas Utility Recovery from Pairwise Preferences

This Rust crate infers a global Cobb–Douglas utility vector `β = (β₁, ..., βₙ)` from a symmetric matrix of **pairwise Cobb–Douglas preference weights** `αᵢⱼ ∈ (0,1)` that satisfy:

\[
\alpha_{ij} + \alpha_{ji} = 1, \quad \text{and} \quad \alpha_{ij} = \frac{\beta_i}{\beta_i + \beta_j}
\]

This system arises naturally when agents specify pairwise convex preferences over goods but you want to recover a **globally consistent** utility function of the form:

\[
u(\mathbf{x}) = \prod_{i=1}^n x_i^{\beta_i}
\]

---