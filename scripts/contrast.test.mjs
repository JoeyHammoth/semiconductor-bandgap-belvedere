import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { checkPalette, hexColours, ratio, themeColours } from "./contrast.mjs";

const tokens = readFileSync(new URL("../web/src/lib/ui/tokens.css", import.meta.url), "utf8");

const BLACK = { l: 0, c: 0, h: 0 };
const WHITE = { l: 1, c: 0, h: 0 };

// A palette in the same shape as tokens.css whose pairs are all too close.
const LOW_CONTRAST = `
:root {
  --bg: oklch(0.20 0 0);
  --surface: oklch(0.26 0 0);
  --ink: oklch(0.30 0 0);
  --ink-dim: oklch(0.32 0 0);
  --accent: oklch(0.34 0 0);
}
:root[data-theme='daylight'] {
  --bg: oklch(0.90 0 0);
  --surface: oklch(0.84 0 0);
  --ink: oklch(0.80 0 0);
  --ink-dim: oklch(0.78 0 0);
  --accent: oklch(0.76 0 0);
}
`;

test("black against white is 21:1", () => {
  assert.equal(ratio(BLACK, WHITE).toFixed(2), "21.00");
});

test("a colour against itself is 1:1", () => {
  assert.equal(ratio(WHITE, WHITE).toFixed(4), "1.0000");
});

test("the ratio is symmetric", () => {
  const accent = themeColours(tokens, ":root {").accent;
  assert.equal(ratio(accent, BLACK), ratio(BLACK, accent));
});

test("the committed palette passes every pair", () => {
  const { rows, failed } = checkPalette(tokens);
  assert.equal(rows.length, 8);
  assert.equal(failed, false);
  for (const row of rows) assert.ok(row.value >= 4.5, `${row.theme} ${row.fg} on ${row.bg}`);
});

test("a low-contrast palette is reported as a failure", () => {
  const { rows, failed } = checkPalette(LOW_CONTRAST);
  assert.equal(failed, true);
  assert.equal(rows.length, 8);
  for (const row of rows) assert.equal(row.ok, false);
  const lowest = rows.find((row) => row.theme === "darkroom" && row.fg === "ink");
  assert.ok(lowest.value < 4.5);
});

test("a threshold above the measured ratio fails a pair that otherwise passes", () => {
  const { failed } = checkPalette(tokens, 100);
  assert.equal(failed, true);
});

test("a raw hex colour in a component is rejected", () => {
  assert.deepEqual(hexColours('<p style="color: #FFEE00">x</p>'), ["#FFEE00"]);
  assert.deepEqual(hexColours("<p>#fff</p>"), ["#fff"]);
});

test("token colours are not mistaken for hex colours", () => {
  assert.deepEqual(hexColours("color: oklch(0.5 0.1 210);"), []);
});

test("a missing token or selector is an error, not a silent pass", () => {
  assert.throws(() => checkPalette(":root { --bg: oklch(1 0 0); }"), /variable not found|selector not found/);
  assert.throws(() => themeColours(tokens, ":root[data-theme='nope']"), /selector not found/);
});
