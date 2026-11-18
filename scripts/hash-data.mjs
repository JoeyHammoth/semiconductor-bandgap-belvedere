#!/usr/bin/env node
// Regenerate or verify data/manifest.json.
// Usage: node scripts/hash-data.mjs [--check]
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

const check = process.argv.includes("--check");
const root = "data";
const manifestPath = join(root, "manifest.json");

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir).sort()) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...walk(path));
    else if (path !== manifestPath) out.push(path);
  }
  return out;
}

const files = {};
for (const path of walk(root)) {
  const rel = relative(root, path).split("\\").join("/");
  if (rel === "README.md" || rel === "schema.json") continue;
  if (rel.split("/").some((part) => part.startsWith("."))) continue;
  files[rel] = createHash("sha256").update(readFileSync(path)).digest("hex");
}

if (process.argv.includes("--print")) {
  console.log(files);
  process.exit(0);
}

if (check) {
  const current = JSON.parse(readFileSync(manifestPath, "utf8"));
  const same = JSON.stringify(current.files, null, 2) === JSON.stringify(files, null, 2);
  if (!same) {
    console.error("data manifest is stale; run `node scripts/hash-data.mjs`");
    process.exit(1);
  }
  console.log("data manifest: up to date");
  process.exit(0);
}

const manifest = {
  generated_at: new Date().toISOString(),
  generator: "scripts/hash-data.mjs",
  algorithm: "sha256",
  files,
};
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
console.log(`hashed ${Object.keys(files).length} data file(s)`);
