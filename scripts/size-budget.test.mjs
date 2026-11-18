// Negative test for scripts/size-budget.mjs: an oversized asset must fail.
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { randomBytes } from "node:crypto";

function run(dir) {
  return spawnSync(process.execPath, ["scripts/size-budget.mjs", dir], {
    encoding: "utf8",
  });
}

test("oversized JS fails the budget", () => {
  const dir = mkdtempSync(join(tmpdir(), "bgo-budget-over-"));
  try {
    writeFileSync(join(dir, "bundle.js"), randomBytes(200 * 1024));
    const result = run(dir);
    assert.equal(result.status, 1);
    assert.match(result.stdout, /OVER/);
    assert.match(result.stderr, /FAILED/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("small assets pass the budget", () => {
  const dir = mkdtempSync(join(tmpdir(), "bgo-budget-ok-"));
  try {
    writeFileSync(join(dir, "app.js"), "console.log('ok');");
    writeFileSync(join(dir, "app.css"), "body{color:black}");
    const result = run(dir);
    assert.equal(result.status, 0);
    assert.match(result.stdout, /passed/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
