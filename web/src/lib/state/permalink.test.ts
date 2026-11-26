import { describe, expect, it } from 'vitest';
import { DEFAULT_STATE, decodeState, encodeState } from './permalink';

describe('permalink state', () => {
  it('round-trips exactly', () => {
    const state = { model: 'M4', dataset: 'si-1n4007-20240412-01', temperature: 342.5, seed: 42 };
    expect(decodeState(encodeState(state))).toEqual(state);
  });

  it('round-trips scene and case fields', () => {
    const state = { ...DEFAULT_STATE, scene: 'npe', caseId: 'm0-report-si' };
    expect(decodeState(encodeState(state))).toEqual(state);
  });

  it('accepts a legacy version-1 payload', () => {
    const raw = JSON.stringify({ v: 1, model: 'M1', dataset: null, temperature: 280, seed: 3 });
    const encoded = btoa(raw).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    expect(decodeState(encoded)).toEqual({ model: 'M1', dataset: null, temperature: 280, seed: 3 });
  });

  it('produces a url-safe string', () => {
    expect(encodeState(DEFAULT_STATE)).toMatch(/^[A-Za-z0-9_-]+$/);
  });

  it('falls back on garbage', () => {
    expect(decodeState('not-base64')).toEqual(DEFAULT_STATE);
  });

  it('falls back on a future version', () => {
    const raw = JSON.stringify({ v: 99, model: 'M6' });
    const encoded = btoa(raw).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    expect(decodeState(encoded)).toEqual(DEFAULT_STATE);
  });
});
