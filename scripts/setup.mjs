#!/usr/bin/env node
// Developer setup: check the toolchains the site needs, add the wasm target and
// install the web dependencies. Works from any working directory; paths resolve
// from the repository root.
//
//   node scripts/setup.mjs           check, fix what can be fixed, print next steps
//   node scripts/setup.mjs --check   report only; exit 1 if a required tool is missing
//
// It never installs Node, Rust or wasm-pack. It prints the exact command instead,
// so nothing outside this repository is changed without you seeing it first.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const checkOnly = process.argv.includes("--check");
const windows = process.platform === "win32";
const requiredNode = 24;
const wasmPackVersion = "0.15.1";

const missing = [];

function call(command, args, options = {}) {
  return spawnSync(command, args, {
    shell: windows,
    stdio: "inherit",
    ...options,
  });
}

function capture(command, args) {
  const result = spawnSync(command, args, { encoding: "utf8", shell: windows });
  return result.status === 0 ? (result.stdout || "").trim() : null;
}

function ok(label) {
  console.log(`  ok      ${label}`);
}

function missingTool(label, hint) {
  missing.push(label);
  console.log(`  MISSING ${label}`);
  console.log(`          ${hint}`);
}

console.log(`\nBand-Gap Observatory setup${checkOnly ? " (check only)" : ""}\n`);

// Node runs the web toolchain and this script.
const nodeMajor = Number(process.versions.node.split(".")[0]);
if (nodeMajor >= requiredNode) {
  ok(`node ${process.version}`);
} else {
  missingTool(`node ${process.version}`, `install Node ${requiredNode} (see .nvmrc)`);
}

// Rust builds the crates and the wasm core.
const cargo = capture("cargo", ["--version"]);
if (cargo) {
  ok(cargo);
} else {
  missingTool("cargo", "install Rust from https://rustup.rs, then re-run this script");
}

// The wasm target is required by crates/bgo-wasm.
const rustup = capture("rustup", ["--version"]);
if (!rustup) {
  missingTool("rustup", "install Rust with rustup so the wasm target can be added");
} else {
  const installed = (capture("rustup", ["target", "list", "--installed"]) || "").split(/\s+/);
  if (installed.includes("wasm32-unknown-unknown")) {
    ok("wasm32-unknown-unknown");
  } else if (checkOnly) {
    missingTool("wasm32-unknown-unknown", "run: rustup target add wasm32-unknown-unknown");
  } else if (call("rustup", ["target", "add", "wasm32-unknown-unknown"])) {
    ok("wasm32-unknown-unknown (added)");
  } else {
    missingTool("wasm32-unknown-unknown", "run: rustup target add wasm32-unknown-unknown");
  }
}

// wasm-pack builds the core every scene loads.
const wasmPack = capture("wasm-pack", ["--version"]);
if (wasmPack && wasmPack.includes(wasmPackVersion)) {
  ok(`wasm-pack ${wasmPackVersion}`);
} else if (wasmPack) {
  missingTool(
    wasmPack,
    `this repo pins wasm-pack ${wasmPackVersion}: cargo install wasm-pack --version ${wasmPackVersion}`,
  );
} else {
  missingTool(
    "wasm-pack",
    `install ${wasmPackVersion} from https://rustwasm.github.io/wasm-pack/installer/ or: cargo install wasm-pack --version ${wasmPackVersion}`,
  );
}

// Web dependencies.
if (existsSync(join(root, "web", "node_modules"))) {
  ok("web dependencies (run npm ci to refresh)");
} else if (checkOnly) {
  missingTool("web/node_modules", "run this script without --check, or: cd web && npm ci");
} else {
  console.log("  ......  installing web dependencies (npm ci)");
  if (call("npm", ["ci"], { cwd: join(root, "web") })) {
    ok("web dependencies");
  } else {
    missingTool("web dependencies", "npm ci failed above; check the network and the Node version");
  }
}

// Python is optional: only train/ needs it.
if (capture("uv", ["--version"])) {
  ok("uv (optional, train/ references)");
} else {
  console.log("  note    uv not found: offline references and NPE training are optional");
}

if (missing.length > 0) {
  console.error(`\n${missing.length} required item(s) missing. See the hints above.\n`);
  process.exit(1);
}

console.log(`
Next:
  RUSTFLAGS="-C target-feature=+simd128" \\
    wasm-pack build crates/bgo-wasm --release --target web --out-dir ../../web/static/wasm
  cd web && npm run dev
`);
