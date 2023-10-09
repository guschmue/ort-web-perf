const fs = require('fs');
const { test, expect } = require('@playwright/test');
const base = "http://localhost:8888/ort-perf.html?go=1&csv=1&";
const timeout = 300 * 1000;
const rows = ["name", "provider", "pass", "threads", "avg", "min", "max", "p50", "p95", "p99", "count", "mem", "tag", "platform", "fist_run", "load"];
const tick = performance.now();
const profiler = 0;
const providers = ["wasm", "webgpu"];
const filter = "default";

const models = [
    { n: "mobilenet-v2", g: "img224", m: "mobilenet_v2/model-12.onnx" },
    { n: "albert-base-v2", g: "bert64", m: "tjs/albert-base-v2/onnx/model.onnx" },
    { n: "bert-base-uncased", g: "bert64", m: "tjs/bert-base-uncased/onnx/model.onnx" },
    { n: "distilbert-base-uncased", g: "bert64", m: "tjs/distilbert-base-uncased/onnx/model.onnx" },
    { n: "roberta-base", g: "bert64", m: "tjs/roberta-base/onnx/model.onnx" },
    { n: "sentence-trans-msbert", g: "bert64", m: "tjs/sentence-transformers/msmarco-distilbert-base-v4/onnx/model.onnx" },
    { n: "nli-deberta-v3-small", g: "bert64", m: "tjs/cross-encoder/nli-deberta-v3-small/onnx/model.onnx" },
    { n: "t5-encoder", g: "t5-encoder", m: "tjs/t5-small/onnx/encoder_model.onnx" },
    { n: "t5-decoder-seq1", g: "t5-decoder", m: "tjs/t5-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "t5-v1.1-encoder", g: "t5-encoder", m: "tjs/google/t5-v1_1-small/onnx/encoder_model.onnx" },
    { n: "t5-v1.1-decoder-seq1", g: "flan-t5-decoder", m: "tjs/google/t5-v1_1-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "flan-t5-encoder", g: "t5-encoder", m: "tjs/google/flan-t5-small/onnx/encoder_model.onnx" },
    { n: "flan-t5-decoder-seq1", g: "flan-t5-decoder", m: "tjs/google/flan-t5-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },

    // FIXME: use_cache_branch=false because some issue with the model on wasm and webgpu
    { n: "gpt-neo-125m", g: "llm-decoder", m: "tjs/EleutherAI/gpt-neo-125M/onnx/decoder_model_merged.onnx", o: "&use_cache_branch=0" },
    { n: "distilgpt2-seq16", g: "llm-decoder", m: "tjs/distilgpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=16&use_cache_branch=0" },
    { n: "gpt2", g: "llm-decoder", m: "tjs/gpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=8&use_cache_branch=0" },

    { n: "whisper-decoder", g: "whisper-decoder", m: "tjs/openai/whisper-tiny/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "whisper-encoder", g: "whisper-encoder", m: "tjs/openai/whisper-tiny/onnx/encoder_model.onnx" },
    { n: "dino-vitb16", g: "img224", m: "tjs/facebook/dino-vitb16/onnx/model.onnx" },
    { n: "clip-vit-base-patch16", g: "clip", m: "tjs/openai/clip-vit-base-patch16/onnx/model.onnx" },
    { n: "vit-base-patch16-224", g: "img224", m: "tjs/google/vit-base-patch16-224/onnx/model.onnx" },
    { n: "vit-gpt2-image-captioning", g: "bart-large-12", m: "tjs/nlpconnect/vit-gpt2-image-captioning/onnx/decoder_model_merged.onnx" },
    { n: "sam-b-encoder", g: "sam-encoder", m: "sam/sam_vit_b-encoder.onnx" },
    { n: "sam-b-decoder", g: "sam-decoder", m: "sam/sam_vit_b-decoder.onnx", o: "&min_query_count=10" },
    { n: "sam-h-decoder", g: "sam-decoder", m: "sam/sam_vit_h-decoder.onnx" },
    { n: "sam-h-decoder-static", g: "sam-decoder", m: "sam/segment-anything-vit-h-static-shapes-static.onnx" },
    { n: "bart-large-encoder", g: "bert64", m: "tjs/facebook/bart-large-cnn/onnx/encoder_model.onnx" },
    { n: "distilbert-base-uncased-mnli", g: "bert64", m: "tjs/distilbert-base-uncased-mnli/onnx/model.onnx", o: "&seqlen=50" },
    { n: "distilbart-cnn-6-6-encoder", g: "bert64", m: "tjs/distilbart-cnn-6-6/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { n: "distilbart-cnn-6-6-decoder", g: "bart-cnn", m: "tjs/distilbart-cnn-6-6/onnx/decoder_model_merged.onnx", o: "&seqlen=168" },
    { n: "vit-gpt2-image-captioning-encoder", g: "img224", m: "tjs/vit-gpt2-image-captioning/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { n: "vit-gpt2-image-captioning-decoder", g: "bart-large-12", m: "tjs/vit-gpt2-image-captioning/onnx/decoder_model_merged.onnx", o: "&seqlen=168" },
    { n: "yolo-small", g: "img640x480", m: "tjs/hustvl/yolos-small/onnx/model.onnx" },
    { n: "detr-resnet-50", g: "detr", m: "tjs/facebook/detr-resnet-50/onnx/model.onnx" },
    { n: "squeezebert", g: "bert64", m: "tjs/squeezebert/squeezebert-uncased/onnx/model.onnx" },
    { n: "yolox-s", g: "img640x640", m: "partya/yolox/yolox_s.onnx" },
    { n: "yolox-l", g: "img640x640", m: "partya/yolox/yolox_l.onnx" },
    { n: "efficientvit-l1", g: "img512x512", m: "partya/efficientvit/l1.onnx" },
    { n: "efficientformer-l1", g: "img224", m: "partya/efficientformer/efficientformer_l1.onnx" },
    { n: "efficientformer-l3", g: "img224", m: "partya/efficientformer/efficientformer_l3.onnx" },
    { n: "topformer", g: "img512x512", m: "partya/TopFormer/TopFormer-S_512x512_2x8_160k.onnx" },
    // { n: "segformer", g: "img640x640", m: "partya/SegFormer/segformer-b5-finetuned-ade-640-640.onnx" },

    // transformers.js demo example
    { e: "tjs-demo", n: "t5-encoder", o: "&seqlen=128" },
    { e: "tjs-demo", n: "t5-decoder-seq1", o: "&seqlen=1&enc_seqlen=128" },
    { e: "tjs-demo", n: "gpt2-seq16", o: "&seqlen=16" },
    { e: "tjs-demo", n: "bert-base-cased", o: "&seqlen=9" },
    { e: "tjs-demo", n: "bert-base-sentiment", o: "&seqlen=63" },
    { e: "tjs-demo", n: "distilbert-base-uncased-mnli", o: "&seqlen=50" },
    { e: "tjs-demo", n: "distilbert-distilled-squad", o: "&seqlen=262" },
    { e: "tjs-demo", n: "distilbart-cnn-6-6-encoder", o: "&seqlen=168" },
    { e: "tjs-demo", n: "distilbart-cnn-6-6-decoder", o: "&seqlen=168" },
    { e: "tjs-demo", n: "whisper-decoder-seq1", o: "&seqlen=1" },
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
    if (model.o !== undefined) {
        opt = model.o;
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
