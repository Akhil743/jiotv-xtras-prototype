// Opens the Streamlit app in a real browser so it counts as a visit, and clicks
// "get this app back up" if it has gone to sleep. Run by .github/workflows/keep-awake.yml.
import puppeteer from 'puppeteer-core';

const APP_URL = 'https://jiotv-xtras-prototype-exo33ekezeaaycvdgdehse.streamlit.app/';
const CHROME = process.env.CHROME_PATH || '/usr/bin/google-chrome';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage();
  await page.goto(APP_URL, { waitUntil: 'networkidle2', timeout: 90_000 });

  const woke = await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((b) => /get this app back up/i.test(b.textContent));
    if (button) button.click();
    return Boolean(button);
  });
  console.log(woke ? 'App was asleep; clicked wake.' : 'App was awake.');

  // Wait up to 4 minutes for the app itself to render inside its frame.
  for (let i = 0; i < 24; i++) {
    for (const frame of page.frames()) {
      const text = await frame.evaluate(() => document.body?.innerText || '').catch(() => '');
      if (/JioTV\+ Xtras/.test(text)) {
        console.log(`App is up after ${i * 10}s.`);
        process.exit(0);
      }
    }
    await sleep(10_000);
  }
  console.error('App did not render within 4 minutes.');
  process.exit(1);
} finally {
  await browser.close();
}
