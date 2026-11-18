import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('landing page renders the story', async ({ page }) => {
  await page.goto('./');
  await expect(page).toHaveTitle(/Band-Gap Observatory/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('How wide is the gap');
  await expect(page.getByRole('contentinfo', { name: 'Compute status' })).toBeVisible();
});

test('landing page has no serious accessibility violations', async ({ page }) => {
  await page.goto('./');
  const results = await new AxeBuilder({ page }).analyze();
  const serious = results.violations.filter(
    (violation) => violation.impact === 'serious' || violation.impact === 'critical',
  );
  expect(serious).toEqual([]);
});
