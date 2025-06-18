import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';

export default defineConfig({
  workers: 1,
  projects: [
    {
      name: 'canary-macos',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'canary',
        headless: false,
        launchOptions: {
          args: [
            "--enable-features=SharedArrayBuffer,webgpu-developer-features,WebMachineLearningNeuralNetwork",
            "--enable-dawn-features=allow_unsafe_apis"
          ],
          executablePath: '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'
        }
      }
    },
    {
      name: 'canary-windows',
      use: {
        ...devices['Desktop Chrome'],
        channel: 'canary',
        headless: false,
        launchOptions: {
          args: [
            "--enable-features=SharedArrayBuffer,webgpu-developer-features,WebMachineLearningNeuralNetwork",
            "--enable-dawn-features=allow_unsafe_apis"
          ],
          executablePath: path.join(process.env.HOME, 'AppData', 'Local', 'Google', 'Chrome SxS', 'Application', 'chrome.exe')

        }
      }
    }
  ],
  expect: { timeout: 300000 }
});
