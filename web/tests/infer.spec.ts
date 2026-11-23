import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('infer route exposes the synthetic case and production settings', async ({ page }) => {
  await page.goto('infer/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('The posterior');
  await expect(page.getByText('Synthetic observations')).toBeVisible();
  await expect(page.getByText(/not recovered data/i).first()).toBeVisible();
  const run = page.getByRole('button', { name: /run m0 inference/i });
  await expect(run).toBeEnabled({ timeout: 30_000 });
  await expect(page.getByText(/5,000/).first()).toBeVisible();
});

test('infer route has no serious accessibility violations', async ({ page }) => {
  await page.goto('infer/');
  await expect(page.getByRole('button', { name: /run m0 inference/i })).toBeEnabled({
    timeout: 30_000,
  });
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter(
    (violation) => violation.impact === 'serious' || violation.impact === 'critical',
  );
  expect(serious).toEqual([]);
});

test('a production M0 run renders the fail-closed result', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'one production sampler run is enough for CI time');
  test.setTimeout(180_000);
  await page.goto('infer/');
  const run = page.getByRole('button', { name: /run m0 inference/i });
  await expect(run).toBeEnabled({ timeout: 30_000 });
  await run.click();

  const badge = page.getByTestId('diagnostics-badge');
  await expect(badge).toBeVisible({ timeout: 150_000 });
  await expect(badge).toHaveAttribute('data-pass', 'true');
  await expect(page.getByTestId('corner-plot')).toBeVisible();
  await expect(page.getByTestId('s1-plot')).toBeVisible();
  await expect(page.getByTestId('energy-trace')).toBeVisible();
  await expect(page.getByRole('img', { name: /S1 noisy line/i })).toBeVisible();
  await expect(page.getByRole('img', { name: /Energy-error trace/i })).toBeVisible();
});
