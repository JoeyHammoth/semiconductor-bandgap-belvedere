//! Monotonic rational-quadratic spline used by each coupling layer.
//!
//! The formulas follow Durkan et al. (2019), "Neural Spline Flows". The
//! transform is the identity outside `[-bound, bound]`; the log-determinant is
//! therefore zero there. The inverse is solved analytically per bin.

#[derive(Debug, Clone)]
pub struct Spline {
    pub bins: usize,
    pub bound: f64,
}

impl Spline {
    pub fn new(bins: usize, bound: f64) -> Result<Self, String> {
        if bins < 2 {
            return Err("spline needs at least two bins".into());
        }
        if !bound.is_finite() || bound <= 0.0 {
            return Err("spline bound must be positive and finite".into());
        }
        Ok(Self { bins, bound })
    }

    fn parameters(&self, raw: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        if raw.len() != 3 * self.bins - 1 {
            return Err(format!(
                "spline expects {} parameters, got {}",
                3 * self.bins - 1,
                raw.len()
            ));
        }
        if raw.iter().any(|value| !value.is_finite()) {
            return Err("spline parameters must be finite".into());
        }
        let widths = softmax(&raw[..self.bins])
            .into_iter()
            .map(|value| value * 2.0 * self.bound)
            .collect();
        let heights = softmax(&raw[self.bins..2 * self.bins])
            .into_iter()
            .map(|value| value * 2.0 * self.bound)
            .collect();
        let mut derivatives = Vec::with_capacity(self.bins + 1);
        derivatives.push(1.0);
        for value in &raw[2 * self.bins..] {
            derivatives.push(softplus(*value) + 1.0e-6);
        }
        derivatives.push(1.0);
        Ok((widths, heights, derivatives))
    }

    pub fn forward(&self, value: f64, raw: &[f64]) -> Result<(f64, f64), String> {
        if !value.is_finite() {
            return Err("spline input must be finite".into());
        }
        let (widths, heights, derivatives) = self.parameters(raw)?;
        if value <= -self.bound || value >= self.bound {
            return Ok((value, 0.0));
        }
        let (cum_widths, cum_heights) = cumulative(&widths, &heights, self.bound);
        let index = bin_index(&cum_widths, value, self.bins);
        let width = widths[index];
        let height = heights[index];
        let left = derivatives[index];
        let right = derivatives[index + 1];
        let theta = (value - cum_widths[index]) / width;
        let one_minus = 1.0 - theta;
        let delta = height / width;
        let denominator = delta + (right + left - 2.0 * delta) * theta * one_minus;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("spline denominator is not positive".into());
        }
        let numerator = height * (delta * theta * theta + left * theta * one_minus);
        let output = cum_heights[index] + numerator / denominator;
        let derivative_numerator = delta
            * delta
            * (right * theta * theta
                + 2.0 * delta * theta * one_minus
                + left * one_minus * one_minus);
        let derivative = derivative_numerator / (denominator * denominator);
        if !output.is_finite() || !derivative.is_finite() || derivative <= 0.0 {
            return Err("spline produced a non-positive derivative".into());
        }
        Ok((output, derivative.ln()))
    }

    pub fn inverse(&self, value: f64, raw: &[f64]) -> Result<f64, String> {
        if !value.is_finite() {
            return Err("spline input must be finite".into());
        }
        let (widths, heights, derivatives) = self.parameters(raw)?;
        if value <= -self.bound || value >= self.bound {
            return Ok(value);
        }
        let (cum_widths, cum_heights) = cumulative(&widths, &heights, self.bound);
        let index = bin_index(&cum_heights, value, self.bins);
        let width = widths[index];
        let height = heights[index];
        let left = derivatives[index];
        let right = derivatives[index + 1];
        let delta = height / width;
        let offset = value - cum_heights[index];
        let quadratic = right + left - 2.0 * delta;
        let a = offset * quadratic + height * (delta - left);
        let b = height * left - offset * quadratic;
        let c = -offset * delta;
        let theta = if a.abs() > 1.0e-14 {
            let discriminant = (b * b - 4.0 * a * c).max(0.0);
            let root = discriminant.sqrt();
            let first = (-b + root) / (2.0 * a);
            let second = (-b - root) / (2.0 * a);
            if (0.0..=1.0).contains(&first) {
                first
            } else {
                second
            }
        } else if b.abs() > 1.0e-14 {
            -c / b
        } else {
            0.0
        };
        let theta = theta.clamp(0.0, 1.0);
        Ok(cum_widths[index] + theta * width)
    }
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exponentials = values
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let total = exponentials.iter().sum::<f64>();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

fn cumulative(widths: &[f64], heights: &[f64], bound: f64) -> (Vec<f64>, Vec<f64>) {
    let mut cum_widths = Vec::with_capacity(widths.len() + 1);
    let mut cum_heights = Vec::with_capacity(heights.len() + 1);
    cum_widths.push(-bound);
    cum_heights.push(-bound);
    for (width, height) in widths.iter().zip(heights) {
        cum_widths.push(cum_widths.last().copied().unwrap_or(-bound) + width);
        cum_heights.push(cum_heights.last().copied().unwrap_or(-bound) + height);
    }
    (cum_widths, cum_heights)
}

fn bin_index(knots: &[f64], value: f64, bins: usize) -> usize {
    let mut index = 0usize;
    while index + 1 < knots.len() - 1 && value >= knots[index + 1] {
        index += 1;
    }
    index.min(bins - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spline_round_trips() {
        let spline = Spline::new(8, 4.0).expect("spline");
        let raw = (0..23)
            .map(|index| (index as f64 - 11.0) * 0.1)
            .collect::<Vec<_>>();
        for value in [-3.9, -1.0, 0.0, 1.3, 3.9] {
            let (transformed, _) = spline.forward(value, &raw).expect("forward");
            let back = spline.inverse(transformed, &raw).expect("inverse");
            assert!((back - value).abs() < 1.0e-10, "{value} -> {back}");
        }
    }
}
