const fs = require('fs');
const { test, expect } = require('@playwright/test');
const { models } = require("./ort-perf-models.js");

const base = "http://localhost:8888/ort-perf.html?go=1&csv=1&";
const timeout = 300 * 1000;
const rows = ["name", "provider", "pass", "threads", "avg", "min", "max", "p50", "p95", "p99", "count", "mem", "tag", "platform", "fist_run", "load"];
const tick = performance.now();
const profiler = 0;
const providers = ["wasm", "webgpu"];
const threads = 1;
const filter = "default";

const csv = "results.csv";


fs.appendFileSync(csv, rows.join(",") + "\n");

for (var i in  models) {
    let model = models[i];
    if (model.e === undefined) {
        model.e = "default";
    }
    if (model.e != filter) {
        continue;
    }
    let opt = "";
    if (model.o !== undefined) {
        opt = model.o;
    }
    const name = model.n;
    if (name.length < 2) {
        continue;
    }
    for (var j in providers) {
        const provider = providers[j];
        const tracefile = `/tmp/ort-perf/${name}-${provider}.log`;

        test(`${name}--${provider}`, async ({ page }) => {
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
            let url = `${base}name=${model.n}&filter=${filter}&threads=${threads}&provider=${provider}${opt}`;
            if (profiler) {
                url += "&profiler=1";
            }
            console.log(url);
            await page.goto(url, { waitUntil: 'domcontentloaded' });
            try {
                await expect(page.locator('text=@@1').first()).toBeVisible({ timeout: timeout });
            } catch (e) {
                const txt = `${model.n},${provider},0,${threads},,,,,,,,,,expect-fail @@2\n`;
                fs.appendFileSync(csv, txt);
                console.log(e);
            }
        });
    }
};
