import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('landing page renders the story', async ({ page }) => {
  await page.goto('./');
  await expect(page).toHaveTitle(/Band-Gap Observatory/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('How wide is the gap');
  await expect(page.getByRole('contentinfo', { name: 'Compute status' })).toBeVisible();
  await expect(page.getByText(/original per-run recordings no longer exist/i).first()).toBeVisible();
});

test('a value computed in Rust/WASM is rendered', async ({ page }) => {
  await page.goto('./');
  const value = page.locator('.value');
  await expect(value).toContainText(/\d\.\d{3}/, { timeout: 20_000 });
  await expect(value).toContainText('1.111');

  await page.locator('#temp').fill('350');
  await expect(value).toContainText('1.097');

  await expect(page.getByRole('contentinfo', { name: 'Compute status' })).toContainText(
    /backend (webgpu|webgl2|static)/,
  );
});

test('the hero renders a scene or its static fallback', async ({ page }) => {
  await page.goto('./');
  const scene = page.locator('.scene canvas, .scene svg').first();
  await expect(scene).toBeVisible();
  await expect(page.locator('.scene')).toHaveAttribute('data-backend', /webgpu|webgl2|static/);
});

test('reduced motion renders the static hero', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('./');
  await expect(page.locator('.scene')).toHaveAttribute('data-backend', 'static');
  await expect(page.locator('.scene svg')).toBeVisible();
});

test('landing page has no serious accessibility violations', async ({ page }) => {
  await page.goto('./');
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter(
    (violation) => violation.impact === 'serious' || violation.impact === 'critical',
  );
  expect(serious).toEqual([]);
});
