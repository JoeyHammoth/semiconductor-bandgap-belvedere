import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const fixtureUrl = new URL('../../../static/data/m0-report-si.json', import.meta.url);
const referenceUrl = new URL('../../../../train/references/m0-report-si.json', import.meta.url);

function read(url: URL) {
  const raw = readFileSync(url, 'utf8');
  return { raw, value: JSON.parse(raw) as Record<string, any> };
}

describe('infer fixture', () => {
  it('is byte-pinned to the committed synthetic reference', () => {
    const fixture = read(fixtureUrl);
    const reference = read(referenceUrl);
    const digest = createHash('sha256').update(reference.raw).digest('hex');

    expect(fixture.value.reference_sha256).toBe(digest);
    expect(fixture.value.source_reference).toBe('train/references/m0-report-si.json');
    expect(fixture.value.case_id).toBe(reference.value.case.case_id);
    expect(fixture.value.model).toBe('M0');
    expect(fixture.value.synthetic).toBe(true);
    expect(fixture.value.observations.synthetic).toBe(true);
    expect(fixture.value.observations).toEqual(reference.value.observations);
    expect(fixture.value.anchor).toEqual(reference.value.case.anchor);
    expect(fixture.value.config_sha256).toBe(reference.value.case.config_sha256);
    expect(fixture.value.seed).toBe(reference.value.case.seed);
  });
});
