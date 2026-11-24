import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

async function expectNoSeriousViolations(page: import('@playwright/test').Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter(
    (violation) => violation.impact === 'serious' || violation.impact === 'critical',
  );
  expect(serious).toEqual([]);
}

test('evidence route exposes both synthetic scenes and the S4 report', async ({ page }) => {
  await page.goto('evidence/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('What the data');
  await expect(page.getByText('Synthetic cases')).toBeVisible();
  await expect(page.getByTestId('nested-trace')).toBeVisible();
  await expect(page.getByTestId('sloppiness')).toBeVisible();
  await expect(page.getByText(/condition number/i).first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/Fisher spectrum, stiffest first/i)).toBeVisible();
});

test('S2 runs the Rust nested sampler and exposes a scrubbable trace', async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== 'chromium', 'one nested-sampling browser run is enough for CI time');
  test.setTimeout(120_000);
  await page.goto('evidence/');
  const run = page.getByRole('button', { name: /run nested sampling/i });
  await expect(run).toBeEnabled({ timeout: 30_000 });
  await run.click();
  await expect(page.getByText(/Final ln Z/i)).toBeVisible({ timeout: 90_000 });
  await expect(page.getByTestId('nested-scrub')).toBeEnabled();
  await expect(page.getByText(/closed-form/i).first()).toBeVisible();
});

test('S4 recomputes Fisher information when the temperature range changes', async ({ page }) => {
  await page.goto('evidence/');
  const summary = page.getByText(/the stiffest combination/i);
  await expect(summary).toBeVisible({ timeout: 30_000 });
  const initial = await summary.textContent();
  await page.locator('#sloppiness-low').fill('260');
  await expect(summary).not.toHaveText(initial ?? '', { timeout: 30_000 });
  await expect(page.getByText(/The Fisher condition number/i)).toBeVisible();
});

test('evidence route has no serious accessibility violations', async ({ page }) => {
  await page.goto('evidence/');
  await expect(page.getByTestId('sloppiness')).toBeVisible({ timeout: 30_000 });
  await expectNoSeriousViolations(page);
});

test('ledger renders the committed comparison and prior-sensitivity flag', async ({ page }) => {
  await page.goto('ledger/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('evidence');
  await expect(page.getByText(/The comparison table/i)).toBeVisible();
  await expect(page.getByText(/Bayes factor and LOO difference/i)).toBeVisible();
  await expect(page.getByText(/Prior sensitivity/i)).toBeVisible();
  await expect(page.getByText('Synthetic M1 observations', { exact: true })).toBeVisible();
});

test('ledger has no serious accessibility violations', async ({ page }) => {
  await page.goto('ledger/');
  await expect(page.getByText(/The comparison table/i)).toBeVisible();
  await expectNoSeriousViolations(page);
});
