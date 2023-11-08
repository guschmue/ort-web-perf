
const models = [
    { n: "mobilenet-v2", g: "img224", m: "mobilenet_v2/model-12.onnx" },
    { e: "fp16", n: "mobilenet-v2-fp16", g: "img224-fp16", m: "mobilenet_v2/model-12-fp16.onnx" },
    { n: "albert-base-v2", g: "bert64", m: "tjs/albert-base-v2/onnx/model.onnx" },
    { n: "bert-base-uncased", g: "bert64", m: "tjs/bert-base-uncased/onnx/model.onnx" },
    { n: "distilbert-base-uncased", g: "bert64", m: "tjs/distilbert-base-uncased/onnx/model.onnx" },
    { n: "roberta-base", g: "bert64", m: "tjs/roberta-base/onnx/model.onnx" },
    { n: "sentence-trans-msbert", g: "bert64", m: "tjs/sentence-transformers/msmarco-distilbert-base-v4/onnx/model.onnx" },
    { n: "nli-deberta-v3-small", g: "bert64", m: "tjs/cross-encoder/nli-deberta-v3-small/onnx/model.onnx" },
    { n: "t5-encoder", g: "t5-encoder", m: "tjs/t5-small/onnx/encoder_model.onnx" },
    { n: "t5-decoder-seq1", g: "t5-decoder", m: "tjs/t5-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "t5-decoder-seq1-i32", g: "t5-decoder-i32", m: "tjs/t5-small/onnx/decoder_model_merged_i32.onnx", o: "&seqlen=1" },
    { n: "t5-v1.1-encoder", g: "t5-encoder", m: "tjs/google/t5-v1_1-small/onnx/encoder_model.onnx" },
    { n: "t5-v1.1-decoder-seq1", g: "flan-t5-decoder", m: "tjs/google/t5-v1_1-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "flan-t5-encoder", g: "t5-encoder", m: "tjs/google/flan-t5-small/onnx/encoder_model.onnx" },
    { n: "flan-t5-decoder-seq1", g: "flan-t5-decoder", m: "tjs/google/flan-t5-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },

    // FIXME: use_cache_branch=false because some issue with the model on wasm and webgpu
    { n: "gpt-neo-125m", g: "llm-decoder", m: "tjs/EleutherAI/gpt-neo-125M/onnx/decoder_model_merged.onnx" },
    { n: "distilgpt2-seq16", g: "llm-decoder", m: "tjs/distilgpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=16" },
    { n: "gpt2", g: "llm-decoder", m: "tjs/gpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=8" },

    { n: "whisper-decoder", g: "whisper-decoder", m: "tjs/openai/whisper-tiny/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "whisper-encoder", g: "whisper-encoder", m: "tjs/openai/whisper-tiny/onnx/encoder_model.onnx" },
    { n: "dino-vitb16", g: "img224", m: "tjs/facebook/dino-vitb16/onnx/model.onnx" },
    { n: "clip-vit-base-patch16", g: "clip", m: "tjs/openai/clip-vit-base-patch16/onnx/model.onnx" },
    { n: "vit-base-patch16-224", g: "img224", m: "tjs/google/vit-base-patch16-224/onnx/model.onnx" },
    { n: "vit-gpt2-image-captioning", g: "bart-large-12", m: "tjs/nlpconnect/vit-gpt2-image-captioning/onnx/decoder_model_merged.onnx" },
    { n: "sam-b-encoder", g: "sam-encoder", m: "sam/sam_vit_b-encoder.onnx", o: "&min_query_count=10" },
    { n: "sam-b-decoder", g: "sam-decoder", m: "sam/sam_vit_b-decoder.onnx" },
    { n: "sam-h-decoder", g: "sam-decoder", m: "sam/sam_vit_h-decoder.onnx" },
    { n: "sam-h-decoder-static", g: "sam-decoder", m: "sam/segment-anything-vit-h-static-shapes-static.onnx" },
    { n: "sam-mobile-encoder", g: "sam-mobile-encoder", m: "sam/mobile_sam-encoder.onnx" },
    { n: "bart-large-encoder", g: "bert64", m: "tjs/facebook/bart-large-cnn/onnx/encoder_model.onnx" },
    { n: "distilbert-base-uncased-mnli", g: "bert64", m: "tjs/distilbert-base-uncased-mnli/onnx/model.onnx", o: "&seqlen=50" },
    { n: "distilbart-cnn-6-6-encoder", g: "bert64", m: "tjs/distilbart-cnn-6-6/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { n: "distilbart-cnn-6-6-decoder", g: "bart-cnn", m: "tjs/distilbart-cnn-6-6/onnx/decoder_model_merged.onnx", o: "&seqlen=168" },
    { n: "vit-gpt2-image-captioning-encoder", g: "img224", m: "tjs/vit-gpt2-image-captioning/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { n: "vit-gpt2-image-captioning-decoder", g: "bart-large-12", m: "tjs/vit-gpt2-image-captioning/onnx/decoder_model_merged.onnx", o: "&seqlen=168" },
    { n: "yolo-small", g: "img640x480", m: "tjs/hustvl/yolos-small/onnx/model.onnx" },
    { n: "detr-resnet-50", g: "detr", m: "tjs/facebook/detr-resnet-50/onnx/model.onnx" },
    { n: "squeezebert", g: "bert64", m: "tjs/squeezebert/squeezebert-uncased/onnx/model.onnx" },
    { n: "swin-tiny", g: "img224", m: "tjs/microsoft/swin-tiny-patch4-window7-224/onnx/model.onnx" },

    // https://github.com/Megvii-BaseDetection/YOLOX
    { n: "yolox-s", g: "img640x640", m: "partya/yolox/yolox_s.onnx" },
    { n: "yolox-l", g: "img640x640", m: "partya/yolox/yolox_l.onnx" },
    // https://github.com/mit-han-lab/efficientvit
    { n: "efficientvit-l1", g: "img512x512", m: "partya/efficientvit/l1.onnx" },
    // https://github.com/snap-research/EfficientFormer
    { n: "efficientformer-l1", g: "img224", m: "partya/efficientformer/efficientformer_l1.onnx" },
    { n: "efficientformer-l3", g: "img224", m: "partya/efficientformer/efficientformer_l3.onnx" },
    // https://github.com/hustvl/TopFormer
    // https://github.com/ibaiGorordo/ONNX-TopFormer-Semantic-Segmentation
    { n: "topformer", g: "img512x512", m: "partya/TopFormer/model.onnx" },
    // https://github.com/NVlabs/SegFormer
    // https://huggingface.co/spaces/chansung/segformer-tf-transformers/blob/main/segformer-b5-finetuned-ade-640-640.onnx
    { n: "segformer", g: "img640x640", m: "partya/SegFormer/segformer-b5-finetuned-ade-640-640.onnx" },

    { n: "-", g: "-", m: "p-" },
    { n: "-", g: "-", m: "p-" },
    { n: "-", g: "-", m: "p-" },

    // transformers.js demo example
    { e: "tjs-demo", n: "t5-encoder", g: "t5-encoder", m: "tjs/t5-small/onnx/encoder_model.onnx", o: "&seqlen=128" },
    { e: "tjs-demo", n: "t5-decoder-seq1", g: "t5-decoder", m: "tjs/t5-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1&enc_seqlen=128" },
    { e: "tjs-demo", n: "distilgpt2", g: "llm-decoder", m: "tjs/distilgpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=16" },
    { e: "tjs-demo", n: "bert-base-cased", g: "bert64", m: "tjs/bert-base-cased/onnx/model.onnx", o: "&seqlen=9" },
    { e: "tjs-demo", n: "bert-base-sentiment", g: "bert64", m: "tjs/bert-base-multilingual-uncased-sentiment/onnx/model.onnx", o: "&seqlen=63" },
    { e: "tjs-demo", n: "distilbert-base-uncased-mnli", g: "bert64", m: "tjs/distilbert-base-uncased-mnli/onnx/model.onnx", o: "&seqlen=50" },
    { e: "tjs-demo", n: "distilbert-distilled-squad", g: "bert64", m: "tjs/distilbert-base-cased-distilled-squad/onnx/model.onnx", o: "&seqlen=262" },
    { e: "tjs-demo", n: "distilbart-cnn-6-6-encoder", g: "bert64", m: "tjs/distilbart-cnn-6-6/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { e: "tjs-demo", n: "distilbart-cnn-6-6-decoder", g: "bart-cnn", m: "tjs/distilbart-cnn-6-6/onnx/decoder_model_merged.onnx", o: "&seqlen=168" },
    { e: "tjs-demo", n: "whisper-decoder-seq1", g: "whisper-decoder", m: "tjs/openai/whisper-tiny/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { e: "tjs-demo", n: "whisper-encoder", g: "whisper-encoder", m: "tjs/openai/whisper-tiny/onnx/encoder_model.onnx" },
    { e: "tjs-demo", n: "vit-gpt2-image-captioning-encoder", g: "img224", m: "tjs/vit-gpt2-image-captioning/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { e: "tjs-demo", n: "vit-gpt2-image-captioning-decoder", g: "bart-large-12", m: "tjs/vit-gpt2-image-captioning/onnx/decoder_model_merged.onnx", o: "&seqlen=168" },
    { e: "tjs-demo", n: "vit-base-patch16-224", g: "img224", m: "tjs/google/vit-base-patch16-224/onnx/model.onnx" },
    { e: "tjs-demo", n: "clip-vit-base-patch16", g: "clip", m: "tjs/openai/clip-vit-base-patch16/onnx/model.onnx" },
    { e: "tjs-demo", n: "detr-resnet-50", g: "detr", m: "tjs/facebook/detr-resnet-50/onnx/model.onnx" },

    { e: "sd", n: "sd-unet-fp16", g: "sd-unet", m: "sd-fp16/unet/model.onnx" },
    { e: "sd", n: "sd-vae-fp32", g: "sd-vae-fp32", m: "sd-fp32/vae_decoder/model.onnx" },
    { e: "sd", n: "sd-vae-fp16", g: "sd-vae", m: "sd-fp16/vae_decoder/model.onnx" },
    { e: "sd", n: "sd-win-unet-fp16", g: "sd-unet", m: "sd-win/Stable-Diffusion-v1.5-unet-fixed-size-batch-1-float16-no-shape-ops-embedded-weights.onnx" },
    { e: "sd", n: "sd-win-vae-fp16", g: "sd-vae", m: "sd-win/sd2.1-inpainting-vae-decoder-float16-zeroed-weights.onnx" },

    // not working

    // Conv > 2D
    { e: "error", n: "wav2vec", g: "wav2vec", m: "tjs/facebook/wav2vec2-base-960h/onnx/model.onnx" },

    // Transpose freemem
    { e: "error", n: "mobilevit", g: "mobilevit", m: "tjs/apple/mobilevit-small/onnx/model.onnx" },

    // matmul fails
    { e: "error", n: "tiny_starcoder_py", g: "starcoder", m: "tjs/bigcode/tiny_starcoder_py/onnx/decoder_model_merged.onnx" },

    // decoder: Gather" failed. Error: Error: no GPU data for input: 1238119040
    { e: "error", n: "bart-large-decoder", g: "bart-large", m: "tjs/facebook/bart-large-cnn/onnx/decoder_model_merged.ort" },
    { e: "error", n: "bart-large-cnn", g: "bart-large", m: "tjs/facebook/bart-large-cnn/onnx/decoder_model_merged.ort" },

    // OOM
    { e: "error", n: "codegen-350M-mono", g: "llm-decoder", m: "tjs/Salesforce/codegen-350M-mono/onnx/decoder_model_merged.ort" },

    // Gather fails
    { e: "error", n: "xlm-roberta-base", g: "bert64", m: "tjs/xlm-roberta-base/onnx/model.ort" },
];

if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = { models };
}
