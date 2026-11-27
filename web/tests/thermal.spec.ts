import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('S6 runs the Rust thermal twin and reports the recovered lag', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'one WASM thermal run is enough for CI time');
  await page.goto('thermal/');
  await expect(page.getByTestId('thermal-run')).toBeEnabled({ timeout: 30_000 });
  await page.getByTestId('thermal-run').click();
  await expect(page.getByTestId('thermal')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('thermal-plot')).toBeVisible();
  await expect(page.getByTestId('thermal-lag')).toContainText(/τ sensor/);
  await expect(page.getByTestId('thermal-smoother')).toContainText('RTS junction RMSE');
  await expect(page.getByTestId('thermal-gp')).toContainText('GP residual RMS');
  await expect(page.getByTestId('thermal-matern')).toContainText('Matérn discrepancy RMS');
  await expect(page.getByTestId('thermal-likelihood')).toContainText('Kalman log-likelihood');
  await expect(page.getByTestId('thermal-modes')).toContainText('DMDc τ');
  await expect(page.getByRole('table', { name: /S6 accessible equivalent/i })).toBeVisible();
});

test('S6 thermal twin has no serious accessibility violations', async ({ page }) => {
  await page.goto('thermal/');
  await expect(page.getByTestId('thermal-table')).toBeVisible({ timeout: 30_000 });
  const results = await new AxeBuilder({ page }).analyze();
  expect(
    results.violations.filter(
      (violation) => violation.impact === 'serious' || violation.impact === 'critical',
    ),
  ).toEqual([]);
});
