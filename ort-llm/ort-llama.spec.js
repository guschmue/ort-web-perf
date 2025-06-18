const fs = require('fs');
const { test, expect } = require('@playwright/test');

const base = "http://localhost:8888/ort-llama.html";
const timeout = 900 * 1000;
const rows = ["name", "took", "token_per_sec", "tag", "platform"];
const tick = performance.now();
const profiler = 0;
const csv = "results.csv";
const models = ['tinyllama', 'tinyllama_fp16', 'stablelm', 'phi2'];

fs.appendFileSync(csv, rows.join(",") + "\n");

for (var i in models) {
    const model = models[i];
    const tracefile = `/tmp/ort-perf/${model}.log`;

    test(`${model}`, async ({ page }) => {
        test.setTimeout(timeout);
        page.on('console', msg => {
            const txt = msg.text();
            if (txt.includes("@@1")) {
                // results are tagged with @@1
                console.log(txt);
            }
            if (txt.includes("@@2")) {
                // csv results are tagged with @@2
                fs.appendFileSync(csv, txt + "\n");
            }
            if (profiler) {
                // if profiling, write all outpout to file
                fs.appendFileSync(tracefile, txt + "\n");
            }
        });
        let url = `${base}?csv=1&model=${model}&max_tokens=48&profiler=${profiler}`;
        await page.goto(url, { waitUntil: 'domcontentloaded' });
        try {
            await expect(page.locator('text=@@2').first()).toBeVisible({ timeout: timeout });
        } catch (e) {
            const txt = `${model.n},${provider},0,${threads},,,,,,,,,,expect-fail @@2\n`;
            fs.appendFileSync(csv, txt);
            console.log(e);
        }
    });
};
