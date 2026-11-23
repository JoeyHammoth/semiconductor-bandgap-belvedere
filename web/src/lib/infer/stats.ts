export function extent(values: ArrayLike<number>): [number, number] {
  let low = Number.POSITIVE_INFINITY;
  let high = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    if (value < low) low = value;
    if (value > high) high = value;
  }
  return Number.isFinite(low) ? [low, high] : [0, 1];
}

export function quantile(values: ArrayLike<number>, probability: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = Array.from(values).sort((left, right) => left - right);
  const position = Math.min(Math.max(probability, 0), 1) * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

export function mean(values: ArrayLike<number>): number {
  if (values.length === 0) return Number.NaN;
  let total = 0;
  for (let index = 0; index < values.length; index += 1) total += values[index];
  return total / values.length;
}

export function fixed(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

export function compact(value: number): string {
  return new Intl.NumberFormat('en', { maximumFractionDigits: 0 }).format(value);
}

export function padExtent(low: number, high: number, fraction = 0.04): [number, number] {
  if (!Number.isFinite(low) || !Number.isFinite(high)) return [0, 1];
  if (low === high) {
    const delta = Math.max(Math.abs(low) * 0.01, 1e-9);
    return [low - delta, high + delta];
  }
  const pad = (high - low) * fraction;
  return [low - pad, high + pad];
}
