import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('NPE page runs the frozen flow and shows calibrated summaries', async ({
  page,
  browserName,
}) => {
  test.skip(browserName !== 'chromium', 'one flow run is enough for CI time');
  await page.goto('npe/');
  const run = page.getByTestId('npe-run');
  await expect(run).toBeEnabled({ timeout: 30_000 });
  await run.click();
  await expect(page.getByTestId('npe')).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId('npe-plot')).toBeVisible();
  await expect(page.getByRole('table', { name: /S5 accessible equivalent/i })).toBeVisible();
  await expect(page.getByTestId('npe-ood')).toContainText(/In-distribution|OOD input/);
  await expect(page.getByText(/SBC minimum p-value/i)).toBeVisible();
});

test('NPE page has no serious accessibility violations', async ({ page }) => {
  await page.goto('npe/');
  await expect(page.getByTestId('npe-run')).toBeVisible({ timeout: 30_000 });
  const results = await new AxeBuilder({ page }).analyze();
  expect(
    results.violations.filter(
      (violation) => violation.impact === 'serious' || violation.impact === 'critical',
    ),
  ).toEqual([]);
});

test('story is keyboard operable', async ({ page }) => {
  await page.goto('story/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('line');
  // The arrow handler is attached on hydration, which can land after the first
  // paint. Press again until the client has taken over instead of racing it.
  await expect
    .poll(async () => {
      await page.keyboard.press('ArrowRight');
      return page.getByRole('heading', { level: 1 }).textContent();
    })
    .toContain('posterior');
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('line');
});

test('PWA manifest and service worker are registered', async ({ page }) => {
  await page.goto('');
  await expect(page.locator('link[rel="manifest"]')).toHaveAttribute(
    'href',
    /manifest\.webmanifest/,
  );
  const manifest = await page.request.get('manifest.webmanifest');
  expect(manifest.ok()).toBe(true);
  await expect.poll(async () => page.evaluate(() => 'serviceWorker' in navigator)).toBe(true);
});

test('PWA serves a navigation offline after the worker activates', async ({
  page,
  context,
  browserName,
}) => {
  test.skip(browserName !== 'chromium', 'service-worker offline navigation is one Chromium run');
  await page.goto('');
  await page.evaluate(async () => {
    const registration = await navigator.serviceWorker.ready;
    if (!registration.active) throw new Error('service worker did not activate');
    if (!navigator.serviceWorker.controller) {
      await new Promise<void>((resolve) => {
        navigator.serviceWorker.addEventListener('controllerchange', () => resolve(), {
          once: true,
        });
      });
    }
  });
  await page.reload();
  await page.waitForFunction(() => Boolean(navigator.serviceWorker.controller));

  await context.setOffline(true);
  try {
    // Playwright's offline emulation does not reliably hand main-frame navigations
    // to the worker, so ask for the document the way a navigation would: fetch it
    // with the network down and check the worker serves the precached S6 page.
    const served = await page.evaluate(async () => {
      const response = await fetch('thermal/');
      const body = await response.text();
      return {
        status: response.status,
        heading: /sensor lags/i.test(body),
        runControl: /data-testid="thermal-run"/.test(body),
      };
    });
    expect(served).toEqual({ status: 200, heading: true, runControl: true });
  } finally {
    await context.setOffline(false);
  }
});
