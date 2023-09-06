import { defineConfig, devices } from '@playwright/test';
const path = require('node:path'); 

const config = {
  projects: [
    /* Test against desktop browsers */
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    }
  ],
  timeout: 300000,
  use: {
    channel: "latest",
    headless: false,
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',
        '--enable-gpu',
        '--ignore-gpu-blocklist',
        '--enable-dawn-features=allow_unsafe_apis',
        '--disable-dawn-features=disallow_unsafe_apis'
      ],
      executablePath: path.join(process.env.HOME, 'AppData', 'Local', 'Google', 'Chrome SxS', 'Application', 'chrome.exe')
    }
  },
  expect: { timeout: 300000 }
}
module.exports = config
