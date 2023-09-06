const fs = require('fs');
const { test, expect } = require('@playwright/test');
const base = "http://localhost:8888/ort-perf.html?go=1&csv=1&";
const timeout = 300 * 1000;
const rows = ["name", "provider", "pass", "threads", "avg", "min", "max", "p50", "p95", "p99", "count", "mem", "tag"];
const tick = performance.now();
const profiler = 0;
const providers = ["wasm", "webgpu"];
// const providers = ["webgpu"];
const filter = "tjs-demo";

// const providers = ["webgpu"];

const models = [
    { n: "mobilenet-v2" },
    { n: "albert-base-v2" },
    { n: "t5-encoder" },
    { n: "t5-decoder" },
    { n: "t5-v1.1-encoder" },
    { n: "t5-v1.1-decoder" },
    { n: "flan-t5-encoder" },
    { n: "flan-t5-decoder" },
    { n: "bert-base-uncased" },
    { n: "distilbert-base-uncased" },
    { n: "gpt-neo-125m" },
    { n: "whisper-decoder" },
    { n: "whisper-encoder" },
    { n: "dino-vitb16", opt: "&min_query_count=15" },
    { n: "sentence-trans-msbert" },
    { n: "sam-b-encoder", opt: "&min_query_count=10" },
    { n: "sam-b-decoder" },
    { n: "sam-h-decoder-static" },
    { n: "clip-vit-base-patch16", opt: "&min_query_count=10" },
    { n: "distilbart-cnn-6-6", },
    { n: "nli-deberta-v3-small", opt: "&min_query_count=10" },
    { n: "vit-base-patch16-224" },
    { n: "bart-large-encoder", opt: "&min_query_count=10" },
    { n: "distilgpt2" },
    { n: "gpt2" },
    { n: "vit-gpt2-image-captioning" },

    // transformers.js demo example
    { e: "tjs-demo", n: "t5-encoder", opt: "&seqlen=128" },
    { e: "tjs-demo", n: "t5-decoder", opt: "&seqlen=1&enc_seqlen=128" },
    { e: "tjs-demo", n: "gpt2", opt: "&seqlen=16" },
    { e: "tjs-demo", n: "bert-base-cased", opt: "&seqlen=9" },
    { e: "tjs-demo", n: "bert-base-sentiment", opt: "&seqlen=63" },
    { e: "tjs-demo", n: "distilbert-base-uncased-mnli", opt: "&seqlen=50" },
    { e: "tjs-demo", n: "distilbert-distilled-squad", opt: "&seqlen=262" },
    { e: "tjs-demo", n: "distilbart-cnn-6-6-encoder", opt: "&seqlen=168" },
    { e: "tjs-demo", n: "distilbart-cnn-6-6-decoder", opt: "&seqlen=168" },
    { e: "tjs-demo", n: "whisper-decoder", opt: "&seqlen=1" },
    { e: "tjs-demo", n: "whisper-encoder" },
    { e: "tjs-demo", n: "vit-gpt2-image-captioning-encoder", opt: "&seqlen=168" },
    { e: "tjs-demo", n: "vit-gpt2-image-captioning-decoder", opt: "&seqlen=168&min_query_count=10" },
    { e: "tjs-demo", n: "vit-base-patch16-224" },
    { e: "tjs-demo", n: "clip-vit-base-patch16", opt: "&min_query_count=10" },
    { e: "tjs-demo", n: "detr-resnet-50" },

    { e: "" }
];

const threads = 1;
const csv = "results.csv";
fs.appendFileSync(csv, rows.join(",") + "\n");

for (var j in models) {
    let model = models[j];
    if (model.e === undefined) {
        model.e = "default";
    }
    if (model.e != filter) {
        continue;
    }
    let opt = "";
    if (model.opt !== undefined) {
        opt = model.opt;
    }
    for (var i in providers) {
        let provider = providers[i];
        const tracefile = `/tmp/ort-perf/${model.n}-${provider}.log`;
        test(`${model.n}-${provider}`, async ({ page }) => {
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
            let url = `${base}name=${model.n}&filter=${filter}&provider=${provider}${opt}`;
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
