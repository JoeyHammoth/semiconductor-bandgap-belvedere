//! Planck spectra, the analytic CIE 1931 fit and sRGB conversion.
//!
//! The colour-matching functions use the multi-lobe Gaussian approximation of
//! Wyman, Sloan and Shirley (2013). The sRGB conversion is the standard IEC
//! 61966-2-1 matrix and transfer function. Out-of-gamut colours are clipped;
//! the UI may perform a perceptual gamut map later, but the core must not
//! hide that a spectrum lies outside sRGB.

use crate::units::{PLANCK, SPEED_OF_LIGHT};

/// CIE XYZ tristimulus values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// CIE xy chromaticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chromaticity {
    pub x: f64,
    pub y: f64,
}

/// Linear sRGB values, nominally in `[0, 1]` after clipping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearSrgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

/// Gamma-encoded sRGB values in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Srgb {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

/// CIE 1931 colour-matching functions from the Wyman et al. approximation.
///
/// Defined over 380–780 nm; values outside that interval are zero.
pub fn cie_xyz(wavelength_nm: f64) -> Xyz {
    if !(380.0..=780.0).contains(&wavelength_nm) {
        return Xyz {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }

    Xyz {
        x: 1.056 * lobe(wavelength_nm, 599.8, 37.9, 31.0)
            + 0.362 * lobe(wavelength_nm, 442.0, 16.0, 26.7)
            - 0.065 * lobe(wavelength_nm, 501.1, 20.4, 26.2),
        y: 0.821 * lobe(wavelength_nm, 568.8, 46.9, 40.5)
            + 0.286 * lobe(wavelength_nm, 530.9, 16.3, 31.1),
        z: 1.217 * lobe(wavelength_nm, 437.0, 11.8, 36.0)
            + 0.681 * lobe(wavelength_nm, 459.0, 26.0, 13.8),
    }
}

/// Planck spectral radiance per unit wavelength, `W sr^-1 m^-3`.
pub fn planck_spectral_radiance(wavelength_nm: f64, temperature_k: f64) -> f64 {
    assert!(
        wavelength_nm > 0.0 && temperature_k > 0.0,
        "wavelength and temperature must be positive"
    );
    let wavelength_m = wavelength_nm * 1e-9;
    let numerator = 2.0 * PLANCK * SPEED_OF_LIGHT * SPEED_OF_LIGHT / wavelength_m.powi(5);
    let exponent =
        PLANCK * SPEED_OF_LIGHT / (wavelength_m * crate::units::BOLTZMANN * temperature_k);
    numerator / exponent.exp_m1()
}

/// Integrate a blackbody spectrum against the CIE fit over 380–780 nm.
pub fn blackbody_xyz(temperature_k: f64) -> Xyz {
    assert!(temperature_k > 0.0, "temperature must be positive");
    let mut xyz = Xyz {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let step_nm = 1.0;
    let mut wavelength_nm = 380.0;
    while wavelength_nm <= 780.0 {
        let radiance = planck_spectral_radiance(wavelength_nm, temperature_k);
        let matching = cie_xyz(wavelength_nm);
        // Trapezoidal weight, half at both endpoints.
        let weight = if wavelength_nm == 380.0 || (wavelength_nm - 780.0).abs() < 1e-12 {
            0.5 * step_nm
        } else {
            step_nm
        };
        xyz.x += radiance * matching.x * weight;
        xyz.y += radiance * matching.y * weight;
        xyz.z += radiance * matching.z * weight;
        wavelength_nm += step_nm;
    }
    xyz
}

/// Chromaticity coordinates, or `None` for a black spectrum.
pub fn chromaticity(xyz: Xyz) -> Option<Chromaticity> {
    let sum = xyz.x + xyz.y + xyz.z;
    if sum <= 0.0 || !sum.is_finite() {
        return None;
    }
    Some(Chromaticity {
        x: xyz.x / sum,
        y: xyz.y / sum,
    })
}

/// Convert XYZ to linear sRGB using the IEC 61966-2-1 matrix.
///
/// The values are normalised to `Y = 1` so that the result describes colour,
/// not absolute radiance; components are clipped to `[0, 1]`.
pub fn linear_srgb_from_xyz(xyz: Xyz) -> LinearSrgb {
    let scale = if xyz.y > 0.0 { 1.0 / xyz.y } else { 1.0 };
    let x = xyz.x * scale;
    let y = xyz.y * scale;
    let z = xyz.z * scale;
    LinearSrgb {
        r: (3.2406 * x - 1.5372 * y - 0.4986 * z).clamp(0.0, 1.0),
        g: (-0.9689 * x + 1.8758 * y + 0.0415 * z).clamp(0.0, 1.0),
        b: (0.0557 * x - 0.2040 * y + 1.0570 * z).clamp(0.0, 1.0),
    }
}

/// Apply the sRGB transfer function.
pub fn encode_srgb(linear: LinearSrgb) -> Srgb {
    Srgb {
        r: encode_channel(linear.r),
        g: encode_channel(linear.g),
        b: encode_channel(linear.b),
    }
}

fn encode_channel(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Blackbody colour as gamma-encoded sRGB.
pub fn blackbody_srgb(temperature_k: f64) -> Srgb {
    encode_srgb(linear_srgb_from_xyz(blackbody_xyz(temperature_k)))
}

fn lobe(x: f64, mu: f64, sigma_left: f64, sigma_right: f64) -> f64 {
    let sigma = if x < mu { sigma_left } else { sigma_right };
    let t = (x - mu) / sigma;
    (-0.5 * t * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn cie_fit_matches_known_peaks() {
        close(cie_xyz(555.0).y, 1.0, 0.01);
        close(cie_xyz(600.0).x, 1.056, 0.01);
        close(cie_xyz(445.0).z, 1.776, 0.02);
        assert_eq!(
            cie_xyz(379.0),
            Xyz {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        );
    }

    #[test]
    fn equal_energy_white_has_neutral_chromaticity() {
        let mut xyz = Xyz {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        for wavelength_nm in 380..=780 {
            let matching = cie_xyz(wavelength_nm as f64);
            xyz.x += matching.x;
            xyz.y += matching.y;
            xyz.z += matching.z;
        }
        let chroma = chromaticity(xyz).unwrap();
        close(chroma.x, 1.0 / 3.0, 5e-4);
        close(chroma.y, 1.0 / 3.0, 5e-4);
    }

    #[test]
    fn planck_peak_moves_towards_blue_with_temperature() {
        let cool_ratio =
            planck_spectral_radiance(500.0, 3000.0) / planck_spectral_radiance(700.0, 3000.0);
        let hot_ratio =
            planck_spectral_radiance(500.0, 6500.0) / planck_spectral_radiance(700.0, 6500.0);
        assert!(hot_ratio > cool_ratio);
    }

    #[test]
    fn planck_radiance_increases_with_temperature() {
        assert!(planck_spectral_radiance(550.0, 4000.0) > planck_spectral_radiance(550.0, 3000.0));
    }

    #[test]
    fn blackbody_6500_k_is_near_d65() {
        let xyz = blackbody_xyz(6500.0);
        let chroma = chromaticity(xyz).unwrap();
        close(chroma.x, 0.3127, 0.01);
        close(chroma.y, 0.3290, 0.01);
    }

    #[test]
    fn srgb_transfer_function_has_the_expected_endpoints() {
        let black = encode_srgb(LinearSrgb {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        });
        assert_eq!(black.r, 0.0);
        let white = encode_srgb(LinearSrgb {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        });
        close(white.r, 1.0, 1e-12);
        close(encode_channel(0.003_130_8), 12.92 * 0.003_130_8, 1e-15);
    }

    #[test]
    fn scaled_xyz_keeps_the_same_chromaticity_and_colour() {
        let xyz = Xyz {
            x: 0.4,
            y: 0.5,
            z: 0.6,
        };
        let a = linear_srgb_from_xyz(xyz);
        let b = linear_srgb_from_xyz(Xyz {
            x: 4.0,
            y: 5.0,
            z: 6.0,
        });
        close(a.r, b.r, 1e-12);
        close(a.g, b.g, 1e-12);
        close(a.b, b.b, 1e-12);
    }
}
