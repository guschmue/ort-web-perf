const fs = require('fs');
const { test, expect } = require('@playwright/test');

const base = "http://localhost:8888/ort-perf/ort-perf.html?go=1&csv=1&";
const timeout = 900 * 1000;
const rows = ["name", "provider", "pass", "threads", "avg", "min", "max", "p50", "p95", "p99", "count", "mem", "tag", "platform", "fist_run", "load"];
const tick = performance.now();
const profiler = 0;
//const providers = ["wasm", "webgpu"];
const providers = ["webgpu"];
const threads = [1, 4];
const filter = "default";
const csv = "results.csv";

// load models from json file
const models = require("./ort-models.json");

fs.appendFileSync(csv, rows.join(",") + "\n");

for (var i in  models) {
    let model = models[i];
    if (model.category != filter) {
        continue;
    } 
    let opt = "";
    if (model.args !== undefined) {
        opt = model.args;
    }
    const name = model.name;
    if (name.length < 2) {
        continue;
    }
    for (var j in providers) {
        for (var th1 in threads) {
            const th = threads[th1];
            const provider = providers[j];
            if (provider === "wasm" && model.nowasm) {
                continue;
            }
            if (model.noweb !== undefined) {
                continue;
            }
            if (provider === "webgpu" && th != 1) {
                continue;
            }
            const tracefile = `/tmp/ort-perf/${name}-${provider}.log`;

            test(`${name}--${provider}--${th}`, async ({ page }) => {
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
                let url = `${base}name=${name}&filter=${filter}&threads=${th}&provider=${provider}${opt}`;
                if (profiler) {
                    url += "&profiler=1";
                }
                if (model.ext !== undefined) {
                    url += `&ext=${model.ext}`;
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
    }
};
