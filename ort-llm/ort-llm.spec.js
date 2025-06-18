const fs = require('fs');
const { test, expect } = require('@playwright/test');

const base = "http://localhost:8888/ort-llm/ort-llm.html";
const timeout = 900 * 1000;
const rows = ["name", "took", "token_per_sec", "tokens", "input_tokens", "prompt_token_per_sec", "gen_token_per_sec", "ttft", "task,precision", "tag", "platform,provider", "generator", "filter"];

const tick = performance.now();
const profiler = 0;
const csv = "results.csv";
//const models = ["llama3.2-1b", "llama3.2-3b", "deepseek-r1", "phi3.5", "phi4", "tinyllama", "gemma3-1b", "smollm2"]
const models = ["llama3.2-1b", "llama3.2-3b", "phi3.5", "tinyllama", "gemma3-1b"]


const tasks = ["generation", "prefill-500", "prefill-1000"];
const max_tokens = (profiler) ? 100 : 512;

fs.appendFileSync(csv, rows.join(",") + "\n");

for (var i in models) {
    const model = models[i];
    for (var j in tasks) {
        const task = tasks[j];
        const tracefile = `/tmp/ort-perf/${model}.${task}.log`;

        test(`${model},${task}`, async ({ page }) => {
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
            let url = `${base}?csv=1&model=${model}&task=${task}&max_tokens=${max_tokens}&quiet=1&profiler=${profiler}`;
            await page.goto(url, { waitUntil: 'domcontentloaded' });
            try {
                await expect(page.locator('text=@@2').first()).toBeVisible({ timeout: timeout });
            } catch (e) {
                const txt = `${model},${provider},0,${threads},,,,,,,,,,expect-fail @@2\n`;
                fs.appendFileSync(csv, txt);
                console.log(e);
            }
        });
    }
};
