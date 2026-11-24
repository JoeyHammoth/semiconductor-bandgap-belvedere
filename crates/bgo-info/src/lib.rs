//! Fisher information, sloppiness, and Bayesian experiment design.
//!
//! The crate is pure linear algebra: it accepts a forward-model Jacobian,
//! assembles `F = Jᵀ Σ⁻¹ J` for constant Gaussian noise, and exposes the
//! covariance, eigen-spectrum and two-dimensional ellipses used by S4. Model
//! evaluation stays in `bgo-infer`; this crate owns only the information
//! geometry, so the browser receives one traceable result per temperature
//! range.

#![forbid(unsafe_code)]

/// Crate version, surfaced in the status bar and the "Receipts" footer.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfoError {
    Empty,
    DimensionMismatch,
    NonFinite,
    NonPositiveSigma,
    NonSquare,
    NotSymmetric,
    Singular,
    InvalidIndex,
}

impl std::fmt::Display for InfoError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(formatter, "matrix or Jacobian is empty"),
            Self::DimensionMismatch => write!(formatter, "matrix dimensions do not match"),
            Self::NonFinite => write!(formatter, "values must be finite"),
            Self::NonPositiveSigma => write!(formatter, "noise sigma must be positive"),
            Self::NonSquare => write!(formatter, "matrix must be square"),
            Self::NotSymmetric => write!(formatter, "matrix must be symmetric"),
            Self::Singular => write!(formatter, "matrix is not positive definite"),
            Self::InvalidIndex => write!(formatter, "matrix index is out of range"),
        }
    }
}

impl std::error::Error for InfoError {}

/// Row-major dense matrix used at the information-geometry boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, InfoError> {
        if rows == 0 || cols == 0 {
            return Err(InfoError::Empty);
        }
        if data.len() != rows * cols {
            return Err(InfoError::DimensionMismatch);
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(InfoError::NonFinite);
        }
        Ok(Self { rows, cols, data })
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Result<Self, InfoError> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err(InfoError::Empty);
        }
        let cols = rows[0].len();
        if rows.iter().any(|row| row.len() != cols) {
            return Err(InfoError::DimensionMismatch);
        }
        Self::new(rows.len(), cols, rows.iter().flatten().copied().collect())
    }

    pub fn zeros(rows: usize, cols: usize) -> Result<Self, InfoError> {
        Self::new(rows, cols, vec![0.0; rows * cols])
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    pub fn get(&self, row: usize, column: usize) -> Result<f64, InfoError> {
        if row >= self.rows || column >= self.cols {
            return Err(InfoError::InvalidIndex);
        }
        Ok(self.data[row * self.cols + column])
    }

    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        if self.rows != self.cols {
            return false;
        }
        (0..self.rows).all(|row| {
            ((row + 1)..self.cols).all(|column| {
                let left = self.data[row * self.cols + column];
                let right = self.data[column * self.cols + row];
                (left - right).abs() <= tolerance * (1.0 + left.abs().max(right.abs()))
            })
        })
    }

    /// Cholesky inverse for a symmetric positive-definite matrix.
    pub fn inverse_symmetric(&self) -> Result<Self, InfoError> {
        if self.rows != self.cols {
            return Err(InfoError::NonSquare);
        }
        if !self.is_symmetric(1.0e-10) {
            return Err(InfoError::NotSymmetric);
        }
        let n = self.rows;
        let mut lower = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..=row {
                let mut value = self.data[row * n + column];
                for k in 0..column {
                    value -= lower[row * n + k] * lower[column * n + k];
                }
                if row == column {
                    if !value.is_finite() || value <= 0.0 {
                        return Err(InfoError::Singular);
                    }
                    lower[row * n + column] = value.sqrt();
                } else {
                    lower[row * n + column] = value / lower[column * n + column];
                }
            }
        }

        let mut inverse = vec![0.0; n * n];
        for column in 0..n {
            let mut intermediate = vec![0.0; n];
            for row in 0..n {
                let mut value = if row == column { 1.0 } else { 0.0 };
                for k in 0..row {
                    value -= lower[row * n + k] * intermediate[k];
                }
                intermediate[row] = value / lower[row * n + row];
            }
            for row in (0..n).rev() {
                let mut value = intermediate[row];
                for k in (row + 1)..n {
                    value -= lower[k * n + row] * inverse[k * n + column];
                }
                inverse[row * n + column] = value / lower[row * n + row];
            }
        }
        Self::new(n, n, inverse)
    }

    /// Jacobi eigendecomposition for a real symmetric matrix. Eigenvalues are
    /// sorted from stiffest (largest) to sloppiest (smallest), and each
    /// eigenvector is sign-normalised deterministically.
    pub fn symmetric_eigen(&self) -> Result<EigenDecomposition, InfoError> {
        if self.rows != self.cols {
            return Err(InfoError::NonSquare);
        }
        if !self.is_symmetric(1.0e-10) {
            return Err(InfoError::NotSymmetric);
        }
        let n = self.rows;
        let mut a = self.data.clone();
        let mut vectors = vec![0.0; n * n];
        for index in 0..n {
            vectors[index * n + index] = 1.0;
        }
        let scale = a
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let threshold = 1.0e-14 * scale.max(1.0);

        for _ in 0..100 {
            let off_diagonal = (0..n)
                .map(|row| {
                    ((row + 1)..n)
                        .map(|column| a[row * n + column] * a[row * n + column])
                        .sum::<f64>()
                })
                .sum::<f64>()
                .sqrt();
            if off_diagonal <= threshold {
                break;
            }
            for p in 0..n.saturating_sub(1) {
                for q in (p + 1)..n {
                    let apq = a[p * n + q];
                    if apq.abs() <= threshold {
                        continue;
                    }
                    let app = a[p * n + p];
                    let aqq = a[q * n + q];
                    let theta = (aqq - app) / (2.0 * apq);
                    let tangent = if theta >= 0.0 {
                        1.0 / (theta + (1.0 + theta * theta).sqrt())
                    } else {
                        -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                    };
                    let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                    let sine = tangent * cosine;

                    for k in 0..n {
                        if k != p && k != q {
                            let akp = a[k * n + p];
                            let akq = a[k * n + q];
                            a[k * n + p] = cosine * akp - sine * akq;
                            a[k * n + q] = sine * akp + cosine * akq;
                            a[p * n + k] = a[k * n + p];
                            a[q * n + k] = a[k * n + q];
                        }
                    }
                    a[p * n + p] =
                        cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
                    a[q * n + q] =
                        sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
                    a[p * n + q] = 0.0;
                    a[q * n + p] = 0.0;

                    for k in 0..n {
                        let vkp = vectors[k * n + p];
                        let vkq = vectors[k * n + q];
                        vectors[k * n + p] = cosine * vkp - sine * vkq;
                        vectors[k * n + q] = sine * vkp + cosine * vkq;
                    }
                }
            }
        }

        let mut pairs = (0..n)
            .map(|column| {
                let mut vector = (0..n)
                    .map(|row| vectors[row * n + column])
                    .collect::<Vec<_>>();
                let pivot = vector
                    .iter()
                    .enumerate()
                    .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                if vector[pivot] < 0.0 {
                    for value in &mut vector {
                        *value = -*value;
                    }
                }
                (a[column * n + column], vector)
            })
            .collect::<Vec<_>>();
        pairs.sort_by(|(left, _), (right, _)| right.total_cmp(left));
        Ok(EigenDecomposition {
            eigenvalues: pairs.iter().map(|(value, _)| *value).collect(),
            eigenvectors: pairs.into_iter().map(|(_, vector)| vector).collect(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EigenDecomposition {
    /// Descending: stiffest first, sloppiest last.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors in the same order as `eigenvalues`.
    pub eigenvectors: Vec<Vec<f64>>,
}

/// A Fisher matrix and the geometry implied by its inverse.
#[derive(Debug, Clone, PartialEq)]
pub struct FisherReport {
    pub fisher: Matrix,
    pub covariance: Matrix,
    pub spectrum: EigenDecomposition,
    pub condition_number: f64,
    pub stiffest_index: usize,
    pub sloppiest_index: usize,
}

/// Assemble `F = Jᵀ J / σ²` for independent Gaussian observations.
pub fn fisher_from_jacobian(jacobian: &[Vec<f64>], sigma: f64) -> Result<Matrix, InfoError> {
    if jacobian.is_empty() || jacobian[0].is_empty() {
        return Err(InfoError::Empty);
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(InfoError::NonPositiveSigma);
    }
    let columns = jacobian[0].len();
    if jacobian.iter().any(|row| row.len() != columns) {
        return Err(InfoError::DimensionMismatch);
    }
    let mut fisher = vec![0.0; columns * columns];
    for row in jacobian {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(InfoError::NonFinite);
        }
        for first in 0..columns {
            for second in 0..=first {
                fisher[first * columns + second] += row[first] * row[second];
            }
        }
    }
    for first in 0..columns {
        for second in 0..=first {
            let value = fisher[first * columns + second] / (sigma * sigma);
            fisher[first * columns + second] = value;
            fisher[second * columns + first] = value;
        }
    }
    Matrix::new(columns, columns, fisher)
}

/// Invert a Fisher matrix and label its stiff and sloppy directions.
pub fn analyze_fisher(fisher: &Matrix) -> Result<FisherReport, InfoError> {
    if fisher.rows() != fisher.cols() {
        return Err(InfoError::NonSquare);
    }
    let covariance = fisher.inverse_symmetric()?;
    let spectrum = fisher.symmetric_eigen()?;
    let stiffest = spectrum.eigenvalues[0];
    let sloppiest = *spectrum.eigenvalues.last().ok_or(InfoError::Empty)?;
    if !sloppiest.is_finite() || sloppiest <= 0.0 || !stiffest.is_finite() {
        return Err(InfoError::Singular);
    }
    Ok(FisherReport {
        fisher: fisher.clone(),
        covariance,
        condition_number: stiffest / sloppiest,
        stiffest_index: 0,
        sloppiest_index: spectrum.eigenvalues.len() - 1,
        spectrum,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    /// Semi-axis along the major direction, in parameter units.
    pub semi_major: f64,
    /// Semi-axis along the minor direction, in parameter units.
    pub semi_minor: f64,
    /// Major-axis angle in radians, measured from the first parameter axis.
    pub angle_rad: f64,
}

/// Two-dimensional covariance ellipse for the `level`-sigma contour.
pub fn covariance_ellipse(
    covariance: &Matrix,
    first: usize,
    second: usize,
    level: f64,
) -> Result<Ellipse, InfoError> {
    if first >= covariance.rows() || second >= covariance.cols() {
        return Err(InfoError::InvalidIndex);
    }
    if !level.is_finite() || level <= 0.0 {
        return Err(InfoError::NonFinite);
    }
    let a = covariance.get(first, first)?;
    let b = covariance.get(first, second)?;
    let d = covariance.get(second, second)?;
    let trace = a + d;
    let determinant = a * d - b * b;
    let discriminant = (trace * trace - 4.0 * determinant).max(0.0).sqrt();
    let major = 0.5 * (trace + discriminant);
    let minor = 0.5 * (trace - discriminant);
    if !minor.is_finite() || minor <= 0.0 || !major.is_finite() {
        return Err(InfoError::Singular);
    }
    let angle = 0.5 * (2.0 * b).atan2(a - d);
    Ok(Ellipse {
        semi_major: level * major.sqrt(),
        semi_minor: level * minor.sqrt(),
        angle_rad: angle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_inverse_is_correct() {
        let matrix = Matrix::from_rows(&[vec![4.0, 1.0], vec![1.0, 3.0]]).unwrap();
        let inverse = matrix.inverse_symmetric().unwrap();
        let expected = [[3.0 / 11.0, -1.0 / 11.0], [-1.0 / 11.0, 4.0 / 11.0]];
        for (row, expected_row) in expected.iter().enumerate() {
            for (column, expected_value) in expected_row.iter().enumerate() {
                assert!((inverse.get(row, column).unwrap() - expected_value).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn jacobi_recovers_a_rotated_spectrum() {
        let matrix = Matrix::from_rows(&[vec![2.0, 1.0], vec![1.0, 2.0]]).unwrap();
        let eigen = matrix.symmetric_eigen().unwrap();
        assert!((eigen.eigenvalues[0] - 3.0).abs() < 1e-14);
        assert!((eigen.eigenvalues[1] - 1.0).abs() < 1e-14);
        assert!((eigen.eigenvectors[0][0] - eigen.eigenvectors[0][1]).abs() < 1e-14);
    }

    #[test]
    fn fisher_assembly_scales_by_noise() {
        let jacobian = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let fisher = fisher_from_jacobian(&jacobian, 2.0).unwrap();
        assert!((fisher.get(0, 0).unwrap() - 0.25).abs() < 1e-15);
        assert!((fisher.get(1, 1).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn ellipse_axes_follow_covariance() {
        let covariance = Matrix::from_rows(&[vec![4.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let ellipse = covariance_ellipse(&covariance, 0, 1, 1.0).unwrap();
        assert!((ellipse.semi_major - 2.0).abs() < 1e-14);
        assert!((ellipse.semi_minor - 1.0).abs() < 1e-14);
    }
}
