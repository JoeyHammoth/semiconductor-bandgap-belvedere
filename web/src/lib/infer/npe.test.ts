import { describe, expect, it } from 'vitest';
import { npeSummaries } from './npe';

describe('NPE summary features', () => {
  it('recovers a linear sweep', () => {
    const values = Array.from({ length: 41 }, (_, index) => -1 + (2 * index) / 40);
    const summary = npeSummaries(values);
    expect(summary[0]).toBeCloseTo(0, 12);
    expect(summary[1]).toBeCloseTo(1, 12);
    expect(summary[2]).toBeCloseTo(0, 12);
    expect(summary[3]).toBeCloseTo(0, 12);
    expect(summary[4]).toBeCloseTo(2, 12);
  });

  it('rejects a short sweep', () => {
    expect(() => npeSummaries([1, 2, 3])).toThrow();
  });
});
