//! Audit the arithmetic printed in the report's Table 1.
//!
//! This is deliberately **not** a regression against recorded runs. The five
//! original logs no longer exist, so the intercepts cannot be refitted. These
//! tests only check that the surviving summary is internally consistent and
//! that its known red-wavelength discrepancy is detected rather than copied.

use bgo_physics::energy_to_wavelength_nm;

/// `(name, measured E_g, reference E_g, quoted percentage error)`.
const TABLE_1: [(&str, f64, f64, f64); 5] = [
    ("Ge", 0.646, 0.67, 3.58),
    ("Si", 1.002, 1.12, 10.54),
    ("red LED", 2.030, 1.97, 3.05),
    ("blue LED", 2.796, 2.67, 4.72),
    ("green LED", 2.303, 2.30, 0.13),
];

#[test]
fn table_1_percentage_errors_match_the_quoted_values() {
    for (name, measured, reference, quoted) in TABLE_1 {
        let computed = (reference - measured).abs() / reference * 100.0;
        assert!(
            (computed - quoted).abs() < 0.011,
            "{name}: quoted {quoted}%, recomputed {computed}%"
        );
    }
}

#[test]
fn led_wavelengths_mostly_round_but_the_red_value_does_not() {
    let red = energy_to_wavelength_nm(2.030);
    assert!((red - 610.76).abs() < 0.01, "red derived wavelength: {red}");
    assert!(
        613.0 - red > 2.0,
        "the report's 613 nm must remain distinguishable from the 610.76 nm derived value"
    );

    let blue = energy_to_wavelength_nm(2.796);
    let green = energy_to_wavelength_nm(2.303);
    assert!(
        (444.0 - blue).abs() < 1.0,
        "blue derived wavelength: {blue}"
    );
    assert!(
        (539.0 - green).abs() < 1.0,
        "green derived wavelength: {green}"
    );
}

#[test]
fn red_percentage_uses_the_quoted_wavelength_not_the_gap() {
    let derived = energy_to_wavelength_nm(2.030);
    let quoted_error: f64 = (630.0 - 613.0) / 630.0 * 100.0;
    let derived_error: f64 = (630.0 - derived) / 630.0 * 100.0;

    assert!((quoted_error - 2.70).abs() < 0.011);
    assert!(
        derived_error - quoted_error > 0.3,
        "the report's 2.70% uses 613 nm; the gap implies {derived_error:.2}%"
    );
}
