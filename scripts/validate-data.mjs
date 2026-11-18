#!/usr/bin/env node
// Validate committed diode runs against data/schema.json and the audit rules in
// docs/data.md. Dependency-free so it runs before `npm ci` in CI.
//
// Usage:
//   node scripts/validate-data.mjs
//   node scripts/validate-data.mjs --require-complete
//
// `--require-complete` additionally demands all five materials and a sibling
// note for each run. Enable it once the recordings are committed.
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const ANNOTATIONS = new Set([
  "$schema",
  "$id",
  "$comment",
  "title",
  "description",
]);

const SUPPORTED = new Set([
  ...ANNOTATIONS,
  "type",
  "enum",
  "pattern",
  "minLength",
  "format",
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "required",
  "properties",
  "additionalProperties",
  "items",
  "minItems",
]);

function typeOf(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number" && Number.isInteger(value)) return "integer";
  return typeof value;
}

function matchesType(value, expected) {
  if (expected === "null") return value === null;
  if (expected === "number") return typeof value === "number" && Number.isFinite(value);
  if (expected === "integer") return Number.isInteger(value);
  if (expected === "object") return typeof value === "object" && value !== null && !Array.isArray(value);
  if (expected === "array") return Array.isArray(value);
  return typeof value === expected;
}

function validDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function validateSchema(value, schema, path, errors) {
  for (const keyword of Object.keys(schema)) {
    if (!SUPPORTED.has(keyword)) {
      throw new Error(`schema uses unsupported keyword "${keyword}" at ${path}; extend scripts/validate-data.mjs`);
    }
  }

  if (schema.type !== undefined) {
    const expected = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!expected.some((type) => matchesType(value, type))) {
      errors.push(`${path}: expected ${expected.join(" or ")}, got ${typeOf(value)}`);
      return;
    }
  }

  if (schema.enum !== undefined && !schema.enum.includes(value)) {
    errors.push(`${path}: expected one of ${schema.enum.map((item) => JSON.stringify(item)).join(", ")}`);
  }

  if (typeof value === "string") {
    if (schema.minLength !== undefined && value.length < schema.minLength) {
      errors.push(`${path}: shorter than ${schema.minLength} character(s)`);
    }
    if (schema.pattern !== undefined && !new RegExp(schema.pattern).test(value)) {
      errors.push(`${path}: does not match ${schema.pattern}`);
    }
    if (schema.format === "date" && !validDate(value)) {
      errors.push(`${path}: expected a real calendar date (YYYY-MM-DD)`);
    }
  }

  if (typeof value === "number") {
    if (schema.minimum !== undefined && value < schema.minimum) {
      errors.push(`${path}: below minimum ${schema.minimum}`);
    }
    if (schema.maximum !== undefined && value > schema.maximum) {
      errors.push(`${path}: above maximum ${schema.maximum}`);
    }
    if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) {
      errors.push(`${path}: must be greater than ${schema.exclusiveMinimum}`);
    }
  }

  if (Array.isArray(value)) {
    if (schema.minItems !== undefined && value.length < schema.minItems) {
      errors.push(`${path}: needs at least ${schema.minItems} item(s)`);
    }
    if (schema.items !== undefined) {
      value.forEach((item, index) => validateSchema(item, schema.items, `${path}[${index}]`, errors));
    }
  }

  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    for (const key of schema.required ?? []) {
      if (!(key in value)) errors.push(`${path}.${key}: required`);
    }
    for (const [key, child] of Object.entries(value)) {
      if (schema.properties?.[key] !== undefined) {
        validateSchema(child, schema.properties[key], `${path}.${key}`, errors);
      } else if (schema.additionalProperties === false) {
        errors.push(`${path}.${key}: unknown property`);
      }
    }
  }
}

/** Validate one run record against the schema plus the audit rules. */
export function validateRun(record, schema) {
  const errors = [];
  validateSchema(record, schema, "run", errors);
  if (errors.length > 0) return errors;

  if (record.provenance.consent !== "granted") {
    errors.push(`run.provenance.consent: must be "granted" before a run is committed`);
  }
  if (!record.provenance.attribution?.trim()) {
    errors.push(`run.provenance.attribution: required when consent is granted`);
  }

  const samples = record.samples;
  const length = samples.t_s.length;
  for (const field of ["adc_counts", "voltage_v", "current_a", "temp_c", "lm234_current_a"]) {
    if (samples[field] !== undefined && samples[field].length !== length) {
      errors.push(`run.samples.${field}: ${samples[field].length} value(s) for ${length} timestamp(s)`);
    }
  }

  return errors;
}

function noteFor(dir, runId) {
  return join(dir, `${runId}.md`);
}

/** Validate every JSON run in a directory. Returns { errors, runs }. */
export function validateDirectory(dir, schema, { requireComplete = false } = {}) {
  const errors = [];
  const runs = [];

  if (!existsSync(dir)) {
    errors.push(`${dir}: directory is missing`);
    return { errors, runs };
  }

  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith(".json") || entry.name.startsWith(".")) continue;
    const path = join(dir, entry.name);
    let record;
    try {
      record = JSON.parse(readFileSync(path, "utf8"));
    } catch (error) {
      errors.push(`${path}: invalid JSON (${error.message})`);
      continue;
    }

    runs.push(record);
    const runErrors = validateRun(record, schema);
    errors.push(...runErrors.map((message) => `${entry.name}: ${message}`));

    const stem = entry.name.slice(0, -".json".length);
    if (record?.run_id !== stem) {
      errors.push(`${entry.name}: run_id ${JSON.stringify(record?.run_id)} does not match filename stem`);
    }

    if (record?.run_id) {
      const note = noteFor(dir, record.run_id);
      if (!existsSync(note)) {
        errors.push(`${entry.name}: missing human-readable note ${record.run_id}.md`);
      } else if (readFileSync(note, "utf8").trim().length === 0) {
        errors.push(`${entry.name}: note ${record.run_id}.md is empty`);
      }
    }
  }

  const ids = runs.map((run) => run?.run_id).filter(Boolean);
  for (const id of new Set(ids)) {
    if (ids.filter((other) => other === id).length > 1) errors.push(`duplicate run_id ${id}`);
  }

  if (requireComplete) {
    const required = ["ge", "si", "led-red", "led-green", "led-blue"];
    for (const material of required) {
      if (!runs.some((run) => run?.material === material)) {
        errors.push(`missing a ${material} run`);
      }
    }
    if (runs.length !== required.length) {
      errors.push(`expected exactly ${required.length} runs, found ${runs.length}`);
    }
  }

  return { errors, runs };
}

function main() {
  const schema = JSON.parse(readFileSync("data/schema.json", "utf8"));
  const requireComplete = process.argv.includes("--require-complete");
  const { errors, runs } = validateDirectory("data/raw", schema, { requireComplete });

  if (errors.length > 0) {
    console.error(`data/raw: ${errors.length} problem(s)`);
    for (const error of errors) console.error(`  - ${error}`);
    process.exit(1);
  }

  if (runs.length === 0) {
    console.log("data/raw: 0 runs committed; schema and audit rules loaded, awaiting the recordings");
  } else {
    console.log(`data/raw: ${runs.length} run(s) valid${requireComplete ? " and complete" : ""}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("validate-data.mjs")) {
  main();
}
