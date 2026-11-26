/** Five design-aware summaries used by the frozen M0 NPE. */
export function npeSummaries(voltage: ArrayLike<number>): Float64Array {
  const values = Array.from(voltage);
  if (values.length < 4) throw new Error('NPE summary needs at least four points');
  const n = values.length;
  const step = 2 / (n - 1);
  const sums = [0, 0, 0, 0, 0];
  const rhs = [0, 0, 0];
  for (let index = 0; index < n; index += 1) {
    const t = -1 + step * index;
    const powers = [1, t, t * t, t * t * t, t * t * t * t];
    for (let power = 0; power < 5; power += 1) sums[power] += powers[power];
    for (let power = 0; power < 3; power += 1) rhs[power] += powers[power] * values[index];
  }
  const matrix: [number[], number[], number[]] = [
    [sums[0], sums[1], sums[2]],
    [sums[1], sums[2], sums[3]],
    [sums[2], sums[3], sums[4]],
  ];
  const coefficients = solve3(matrix, rhs);
  let residual = 0;
  for (let index = 0; index < n; index += 1) {
    const t = -1 + step * index;
    const fitted = coefficients[0] + coefficients[1] * t + coefficients[2] * t * t;
    residual += (values[index] - fitted) ** 2;
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / n;
  return new Float64Array([
    mean,
    coefficients[1],
    coefficients[2],
    Math.sqrt(residual / n),
    values[n - 1] - values[0],
  ]);
}

function solve3(matrix: [number[], number[], number[]], rhs: number[]): number[] {
  const augmented = matrix.map((row, index) => [...row, rhs[index]]);
  for (let pivot = 0; pivot < 3; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[best][pivot])) best = row;
    }
    if (Math.abs(augmented[best][pivot]) < 1e-14) throw new Error('singular NPE summary fit');
    [augmented[pivot], augmented[best]] = [augmented[best], augmented[pivot]];
    const divisor = augmented[pivot][pivot];
    for (let column = pivot; column < 4; column += 1) augmented[pivot][column] /= divisor;
    for (let row = 0; row < 3; row += 1) {
      if (row === pivot) continue;
      const factor = augmented[row][pivot];
      for (let column = pivot; column < 4; column += 1) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }
  return augmented.map((row) => row[3]);
}
