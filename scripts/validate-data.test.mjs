import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { validateDirectory, validateRun } from "./validate-data.mjs";

const schema = JSON.parse(readFileSync(new URL("../data/schema.json", import.meta.url), "utf8"));

function validRun(overrides = {}) {
  return {
    run_id: "ge-1n60p-20240412-01",
    material: "ge",
    device: "1N60P",
    recorded_at: "2024-04-12",
    instrument: {
      adc_fsr_v: 4.096,
      adc_bits: 16,
      r_set_ohm: 220,
      r_temp_ohm: 150,
      r1_ohm: null,
      supply_v: 5,
    },
    samples: {
      t_s: [0, 60, 120],
      adc_counts: [1000, 1200, 1400],
      voltage_v: [0.32, 0.3, 0.28],
      current_a: [0.000307, 0.000307, 0.000307],
      temp_c: [20, 40, 60],
      lm234_current_a: [0.000307, 0.000307, 0.000307],
    },
    notes: "Synthetic fixture; not a recording.",
    provenance: {
      operator: "fixture",
      consent: "granted",
      attribution: "Synthetic fixture for validator tests",
    },
    ...overrides,
  };
}

function mutated(mutate) {
  const run = validRun();
  mutate(run);
  return validateRun(run, schema);
}

test("a complete, consented run passes", () => {
  assert.deepEqual(validateRun(validRun(), schema), []);
});

test("malformed values are rejected", async (t) => {
  const cases = [
    ["run id pattern", (run) => (run.run_id = "ge-1n60p-2024"), /does not match/],
    ["unknown material", (run) => (run.material = "gan"), /expected one of/],
    ["missing instrument field", (run) => delete run.instrument.supply_v, /supply_v: required/],
    ["zero resistance", (run) => (run.instrument.r_set_ohm = 0), /greater than 0/],
    ["fractional ADC count", (run) => (run.samples.adc_counts[0] = 1.5), /expected integer/],
    ["short sample array", (run) => run.samples.t_s.push(180), /4 timestamp/],
    ["impossible date", (run) => (run.recorded_at = "2024-02-30"), /calendar date/],
    ["unknown property", (run) => (run.extra = true), /unknown property/],
    ["consent pending", (run) => (run.provenance.consent = "pending"), /must be "granted"/],
    ["missing attribution", (run) => delete run.provenance.attribution, /attribution: required/],
  ];

  for (const [name, mutate, pattern] of cases) {
    await t.test(name, () => {
      const errors = mutated(mutate);
      assert.ok(errors.length > 0, `${name} should fail`);
      assert.ok(errors.some((error) => pattern.test(error)), `${name}: ${errors.join("; ")}`);
    });
  }
});

function fixtureDirectory(runs) {
  const dir = mkdtempSync(join(tmpdir(), "bgo-data-"));
  for (const run of runs) {
    writeFileSync(join(dir, `${run.run_id}.json`), JSON.stringify(run, null, 2));
    writeFileSync(join(dir, `${run.run_id}.md`), `# ${run.run_id}\n\nSynthetic fixture.\n`);
  }
  return dir;
}

test("a run without its note is rejected", (t) => {
  const dir = fixtureDirectory([validRun()]);
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  rmSync(join(dir, "ge-1n60p-20240412-01.md"));

  const { errors } = validateDirectory(dir, schema);
  assert.ok(errors.some((error) => /missing human-readable note/.test(error)), errors.join("; "));
});

test("filename and run_id must agree", (t) => {
  const dir = mkdtempSync(join(tmpdir(), "bgo-data-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const run = validRun();
  writeFileSync(join(dir, "ge-wrong-20240412-01.json"), JSON.stringify(run));
  writeFileSync(join(dir, "ge-wrong-20240412-01.md"), "# note\n");

  const { errors } = validateDirectory(dir, schema);
  assert.ok(errors.some((error) => /does not match filename stem/.test(error)), errors.join("; "));
});

test("--require-complete needs all five materials", (t) => {
  const dir = fixtureDirectory([validRun()]);
  t.after(() => rmSync(dir, { recursive: true, force: true }));

  const { errors } = validateDirectory(dir, schema, { requireComplete: true });
  for (const material of ["si", "led-red", "led-green", "led-blue"]) {
    assert.ok(errors.some((error) => error.includes(`missing a ${material} run`)), errors.join("; "));
  }
  assert.ok(errors.some((error) => /expected exactly 5 runs/.test(error)));
});

test("a complete five-run set passes in strict mode", (t) => {
  const materials = ["ge", "si", "led-red", "led-green", "led-blue"];
  const runs = materials.map((material, index) =>
    validRun({
      run_id: `${material}-fixture-2024041${index + 1}-01`,
      material,
      recorded_at: `2024-04-1${index + 1}`,
    }),
  );
  const dir = fixtureDirectory(runs);
  t.after(() => rmSync(dir, { recursive: true, force: true }));

  const { errors } = validateDirectory(dir, schema, { requireComplete: true });
  assert.deepEqual(errors, []);
});

test("an empty data directory is reported, not treated as complete", (t) => {
  const dir = mkdtempSync(join(tmpdir(), "bgo-data-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));

  const loose = validateDirectory(dir, schema);
  assert.deepEqual(loose.errors, []);
  assert.equal(loose.runs.length, 0);

  const strict = validateDirectory(dir, schema, { requireComplete: true });
  assert.ok(strict.errors.length >= 5, strict.errors.join("; "));
});
