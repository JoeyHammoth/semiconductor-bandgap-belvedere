#!/usr/bin/env node
// Enforce the gzip budgets from docs/architecture.md §8.
// Usage: node scripts/size-budget.mjs [buildDir]
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, extname, relative } from "node:path";
import { gzipSync } from "node:zlib";

const buildDir = process.argv[2] ?? "web/build";

// Budgets in bytes, gzip. Keep in sync with docs/architecture.md.
const budgets = {
  js: 150 * 1024,
  css: 30 * 1024,
  wasm: 700 * 1024,
  weights: 700 * 1024,
  three: 250 * 1024,
  fonts: 150 * 1024,
  data: 2 * 1024 * 1024,
  total: 60 * 1024 * 1024,
};

const groups = {
  js: { budget: budgets.js, bytes: 0, files: [] },
  css: { budget: budgets.css, bytes: 0, files: [] },
  wasm: { budget: budgets.wasm, bytes: 0, files: [] },
  weights: { budget: budgets.weights, bytes: 0, files: [] },
  fonts: { budget: budgets.fonts, bytes: 0, files: [] },
  data: { budget: budgets.data, bytes: 0, files: [] },
  other: { budget: null, bytes: 0, files: [] },
};

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    const st = statSync(path);
    if (st.isDirectory()) out.push(...walk(path));
    else out.push(path);
  }
  return out;
}

let totalBytes = 0;

let files;
try {
  files = walk(buildDir);
} catch {
  console.error(`size-budget: build directory not found: ${buildDir}`);
  process.exit(1);
}

for (const file of files) {
  const rel = relative(buildDir, file).split("\\").join("/");
  if (rel === ".nojekyll" || rel.endsWith(".map")) continue;
  const ext = extname(file).toLowerCase();
  const gz = gzipSync(readFileSync(file), { level: 9 }).byteLength;
  let key = "other";
  if (ext === ".js" || ext === ".mjs") key = "js";
  else if (ext === ".css") key = "css";
  else if (ext === ".wasm") key = "wasm";
  else if (ext === ".woff2" || ext === ".woff" || ext === ".ttf") key = "fonts";
  else if (ext === ".json" && rel.startsWith("data/")) key = "data";
  else if (ext === ".bin" || rel.startsWith("weights/")) key = "weights";
  groups[key].bytes += gz;
  groups[key].files.push([rel, gz]);
  totalBytes += gz;
}

const kb = (n) => `${(n / 1024).toFixed(1)} KB`;
let failed = false;

console.log(`size budgets for ${buildDir} (gzip)`);
for (const [name, g] of Object.entries(groups)) {
  if (!g.budget) {
    console.log(`  ${name.padEnd(8)} ${kb(g.bytes).padStart(10)}  (unbudgeted)`);
    continue;
  }
  const over = g.bytes > g.budget;
  failed ||= over;
  console.log(
    `  ${name.padEnd(8)} ${kb(g.bytes).padStart(10)} / ${kb(g.budget).padStart(10)} ${over ? "OVER" : "ok"}`,
  );
  if (over) {
    for (const [rel, bytes] of g.files.sort((a, b) => b[1] - a[1]).slice(0, 5)) {
      console.log(`      ${kb(bytes).padStart(10)}  ${rel}`);
    }
  }
}

console.log(`  ${"total".padEnd(8)} ${kb(totalBytes).padStart(10)} / ${kb(budgets.total).padStart(10)} ${totalBytes > budgets.total ? "OVER" : "ok"}`);
if (totalBytes > budgets.total) failed = true;

if (failed) {
  console.error("size-budget: FAILED");
  process.exit(1);
}
console.log("size-budget: passed");
