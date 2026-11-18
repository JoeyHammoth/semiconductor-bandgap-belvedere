#!/usr/bin/env python3
"""Reproduce the numbers in docs/physics-audit.md.

Stdlib only, so it runs anywhere the repository is checked out:

    python scripts/physics-audit.py

Every table printed here appears in the audit note. If a number changes,
change both and say why in the commit message.
"""

from __future__ import annotations

import math

K_B = 8.617333262e-5  # eV/K
HC_EV_NM = 1239.84  # eV·nm
T_LOW = 300.0
T_HIGH = 383.15  # the report's stated ceiling
SAMPLE = 4001

# Varshni's 1967 Table I (indirect gaps) and the handout's printed table.
VARSHNI = {
    "Si (Varshni 1967)": (1.1557, 7.021e-4, 1108.0),
    "Si (1.166 / 4.73e-4 / 636)": (1.1660, 4.73e-4, 636.0),
    "Si (handout, literal)": (1.1200, 7.02e-4, 1108.0),
    "Ge (Varshni 1967)": (0.7412, 4.56e-4, 210.0),
    "Ge (handout, literal)": (0.6700, 4.56e-4, 210.0),
}


def varshni(t: float, e0: float, alpha: float, beta: float) -> float:
    return e0 - alpha * t * t / (t + beta)


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (intercept, slope) of an ordinary least-squares line."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def grid(lo: float = T_LOW, hi: float = T_HIGH, n: int = SAMPLE) -> list[float]:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def gaps() -> None:
    print("== Varshni gaps (eV) ==")
    print(f"{'set':28s} {'Eg(0)':>7s} {'Eg(300)':>8s} {'Eg(383)':>8s} {'chord':>7s}")
    for name, (e0, alpha, beta) in VARSHNI.items():
        chord, _ = linear_fit(grid(), [varshni(t, e0, alpha, beta) for t in grid()])
        print(
            f"{name:28s} {e0:7.4f} {varshni(300, e0, alpha, beta):8.4f} "
            f"{varshni(383.15, e0, alpha, beta):8.4f} {chord:7.4f}"
        )


def eta_shift() -> None:
    print("\n== chord intercept of the -eta*(kT/q)*ln T term, 300-383.15 K ==")
    ts = grid()
    one, _ = linear_fit(ts, [-1 * K_B * t * math.log(t) for t in ts])
    three, _ = linear_fit(ts, [-3 * K_B * t * math.log(t) for t in ts])
    print(f"eta = 1: {one * 1e3:+7.2f} mV")
    print(f"eta = 3: {three * 1e3:+7.2f} mV")
    print(f"per unit eta: {(three - one) / 2 * 1e3:+7.2f} mV")
    print(f"span eta 1.5..5: {(three - one) / 2 * 3.5 * 1e3:7.1f} mV")


def model_intercepts() -> None:
    print("\n== full V(T) chord intercepts ==")
    ts = grid()
    for name, (e0, alpha, beta) in VARSHNI.items():
        for eta in (0.0, 1.0, 3.0):
            ys = [varshni(t, e0, alpha, beta) - eta * K_B * t * math.log(t) for t in ts]
            intercept, slope = linear_fit(ts, ys)
            print(f"{name:28s} eta={eta:3.0f}  {intercept:7.4f} eV  {slope * 1e3:+7.3f} meV/K")


def report_comparison() -> None:
    print("\n== report intercepts ==")
    report = {
        "Ge": (0.646, 0.7412, 0.6607, 0.7778),
        "Si": (1.002, 1.1557, 1.1108, 1.1987),
    }
    for diode, (measured, eg0, eg300, chord) in report.items():
        print(
            f"{diode}: measured {measured:.3f}, Eg(0) {eg0:.4f}, Eg(300) {eg300:.4f}, "
            f"chord {chord:.4f}, deficit {measured - chord:+.4f} eV, "
            f"required eta {(measured - chord) / 0.02932:+.1f}"
        )
    for name, value in (("red", 2.030), ("blue", 2.796), ("green", 2.303)):
        print(f"{name:5s} {value:.3f} eV -> {HC_EV_NM / value:.3f} nm")


def currents() -> None:
    print("\n== LM234 set current (gamma = 227 uV/K) ==")
    for r_set in (220.0, 2200.0):
        i_298 = 227e-6 * 298.0 / r_set
        i_383 = 227e-6 * 383.15 / r_set
        print(
            f"R_set = {r_set:6.0f} ohm: {i_298 * 1e6:7.1f} uA at 298 K, "
            f"{i_383 * 1e6:7.1f} uA at 383 K, V_Rtemp(150 ohm) = {i_298 * 150 * 1e3:.1f} mV"
        )
    print(f"10x current step: {K_B * 300 * math.log(10) * 1e3:.1f} mV (n=1), "
          f"{2 * K_B * 300 * math.log(10) * 1e3:.1f} mV (n=2)")


if __name__ == "__main__":
    gaps()
    eta_shift()
    model_intercepts()
    report_comparison()
    currents()
