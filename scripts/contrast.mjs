#!/usr/bin/env node
// Check the token palette's WCAG contrast for body text and links.
// Reads web/src/lib/ui/tokens.css so the check cannot drift from the CSS.
//
// The helpers are exported so scripts/contrast.test.mjs can exercise the colour
// maths directly. Run as a CLI it prints one line per pair and exits non-zero
// when a pair fails or a component uses a raw hex colour.
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

export const THRESHOLD = 4.5;

export const THEMES = [
  { name: "darkroom", selector: ":root {" },
  { name: "daylight", selector: ":root[data-theme='daylight']" },
];

export const PAIRS = [
  ["ink", "bg"],
  ["ink-dim", "bg"],
  ["accent", "bg"],
  ["accent", "surface"],
];

const TOKENS = ["bg", "surface", "ink", "ink-dim", "accent"];

export function block(css, selector) {
  const start = css.indexOf(selector);
  if (start < 0) throw new Error(`selector not found: ${selector}`);
  const open = css.indexOf("{", start);
  const close = css.indexOf("}", open);
  return css.slice(open + 1, close);
}

export function variable(source, name) {
  const line = source
    .split("\n")
    .find((candidate) => candidate.trim().startsWith(`--${name}:`));
  if (!line) throw new Error(`variable not found: --${name}`);
  const value = line.slice(line.indexOf(":") + 1).replace(";", "").trim();
  const open = value.indexOf("oklch(");
  const close = value.indexOf(")", open);
  if (open < 0 || close < 0) throw new Error(`--${name} is not an oklch colour: ${value}`);
  const [l, c, h] = value
    .slice(open + 6, close)
    .trim()
    .split(/\s+/)
    .map(Number);
  return { l, c, h };
}

export function oklchToLinear({ l, c, h }) {
  const hue = (h * Math.PI) / 180;
  const a = c * Math.cos(hue);
  const b = c * Math.sin(hue);
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = l - 0.0894841775 * a - 1.291485548 * b;
  const L = l_ ** 3;
  const M = m_ ** 3;
  const S = s_ ** 3;
  const clamp = (value) => Math.min(1, Math.max(0, value));
  return [
    clamp(4.0767416621 * L - 3.3077115913 * M + 0.2309699292 * S),
    clamp(-1.2684380046 * L + 2.6097574011 * M - 0.3413193965 * S),
    clamp(-0.0041960863 * L - 0.7034186147 * M + 1.707614701 * S),
  ];
}

export function luminance(colour) {
  const [r, g, b] = oklchToLinear(colour);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function ratio(a, b) {
  const la = luminance(a);
  const lb = luminance(b);
  const [light, dark] = la > lb ? [la, lb] : [lb, la];
  return (light + 0.05) / (dark + 0.05);
}

// The five token colours declared by one theme.
export function themeColours(css, selector) {
  const source = block(css, selector);
  const colours = {};
  for (const token of TOKENS) colours[token] = variable(source, token);
  return colours;
}

// Every foreground/background pair in every theme, with its ratio and verdict.
export function checkPalette(css, threshold = THRESHOLD) {
  const rows = [];
  for (const theme of THEMES) {
    const colours = themeColours(css, theme.selector);
    for (const [fg, bg] of PAIRS) {
      const value = ratio(colours[fg], colours[bg]);
      rows.push({ theme: theme.name, fg, bg, value, ok: value >= threshold });
    }
  }
  return { rows, failed: rows.some((row) => !row.ok) };
}

// Components must use tokens, never raw hex colours (issue #13).
export function hexColours(source) {
  const hex = /#[0-9a-fA-F]{3,8}(?![0-9a-zA-Z])/g;
  return [...source.matchAll(hex)].map((match) => match[0]);
}

export function svelteFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...svelteFiles(path));
    else if (path.endsWith(".svelte")) out.push(path);
  }
  return out;
}

function main() {
  const css = readFileSync("web/src/lib/ui/tokens.css", "utf8");
  let failed = false;

  for (const row of checkPalette(css).rows) {
    failed ||= !row.ok;
    console.log(
      `${row.theme.padEnd(9)} ${`${row.fg} on ${row.bg}`.padEnd(18)} ${row.value.toFixed(2)}:1 ${row.ok ? "ok" : "FAIL"}`,
    );
  }

  for (const file of svelteFiles("web/src")) {
    for (const hex of hexColours(readFileSync(file, "utf8"))) {
      failed = true;
      console.log(`${file}: raw hex colour ${hex}`);
    }
  }

  if (failed) {
    console.error(`contrast: FAILED (threshold ${THRESHOLD}:1; tokens only)`);
    process.exit(1);
  }
  console.log(`contrast: passed (threshold ${THRESHOLD}:1; no raw hex colours)`);
}

const invokedDirectly = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedDirectly) main();
