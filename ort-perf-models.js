
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
    { n: "t5-decoder-iob-seq1", g: "t5-decoder", m: "tjs/t5-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1&io_binding=1" },
    { n: "t5-v1.1-encoder", g: "t5-encoder", m: "tjs/google/t5-v1_1-small/onnx/encoder_model.onnx" },
    { n: "t5-v1.1-decoder-seq1", g: "flan-t5-decoder", m: "tjs/google/t5-v1_1-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "flan-t5-encoder", g: "t5-encoder", m: "tjs/google/flan-t5-small/onnx/encoder_model.onnx" },
    { n: "flan-t5-decoder-seq1", g: "flan-t5-decoder", m: "tjs/google/flan-t5-small/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },

    { n: "gpt-neo-125m-seq1", g: "llm-decoder", m: "tjs/EleutherAI/gpt-neo-125M/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "distilgpt2-seq1", g: "llm-decoder", m: "tjs/distilgpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "distilgpt2-seq16", g: "llm-decoder", m: "tjs/distilgpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=16" },
    { n: "gpt2-seq1", g: "llm-decoder", m: "tjs/gpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "gpt2-seq16", g: "llm-decoder", m: "tjs/gpt2/onnx/decoder_model_merged.onnx", o: "&seqlen=16" },

    { n: "whisper-decoder", g: "whisper-decoder", m: "tjs/openai/whisper-tiny/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { n: "whisper-decoder-iob", g: "whisper-decoder", m: "tjs/openai/whisper-tiny/onnx/decoder_model_merged.onnx", o: "&seqlen=1&io_binding=1" },
    { n: "whisper-encoder", g: "whisper-encoder", m: "tjs/openai/whisper-tiny/onnx/encoder_model.onnx" },

    { n: "dino-vitb16", g: "img224", m: "tjs/facebook/dino-vitb16/onnx/model.onnx" },
    { n: "clip-vit-base-patch16", g: "clip", m: "tjs/openai/clip-vit-base-patch16/onnx/model.onnx" },
    { n: "vit-base-patch16-224", g: "img224", m: "tjs/google/vit-base-patch16-224/onnx/model.onnx" },
    { n: "vit-gpt2-image-captioning", g: "bart-large-12", m: "tjs/nlpconnect/vit-gpt2-image-captioning/onnx/decoder_model_merged.onnx" },
    { n: "sam-b-encoder", g: "sam-encoder", m: "sam/sam_vit_b-encoder.onnx", o: "&min_query_count=3" },
    { n: "sam-b-decoder", g: "sam-decoder", m: "sam/sam_vit_b-decoder.onnx" },
    { n: "sam-h-decoder", g: "sam-decoder", m: "sam/sam_vit_h-decoder.onnx" },
    { n: "sam-h-decoder-static", g: "sam-decoder", m: "sam/segment-anything-vit-h-static-shapes-static.onnx" },
    { n: "sam-mobile-encoder", g: "sam-mobile-encoder", m: "sam/mobile_sam-encoder.onnx" },
    { n: "bart-large-encoder", g: "bert64", m: "tjs/facebook/bart-large-cnn/onnx/encoder_model.onnx" },
    { n: "distilbert-base-uncased-mnli", g: "bert64", m: "tjs/distilbert-base-uncased-mnli/onnx/model.onnx", o: "&seqlen=50" },
    { n: "distilbart-cnn-6-6-encoder", g: "bert64", m: "tjs/distilbart-cnn-6-6/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { n: "distilbart-cnn-6-6-decoder", g: "bart-cnn", m: "tjs/distilbart-cnn-6-6/onnx/decoder_model_merged.onnx", o: "&seqlen=168&min_query_count=5" },
    { n: "vit-gpt2-image-captioning-encoder", g: "img224", m: "tjs/vit-gpt2-image-captioning/onnx/encoder_model.onnx", o: "&seqlen=168" },
    { n: "vit-gpt2-image-captioning-decoder", g: "bart-large-12", m: "tjs/vit-gpt2-image-captioning/onnx/decoder_model_merged.onnx", o: "&seqlen=168" },
    { n: "yolo-small", g: "img640x480", m: "tjs/hustvl/yolos-small/onnx/model.onnx" },
    { n: "detr-resnet-50", g: "detr", m: "tjs/facebook/detr-resnet-50/onnx/model.onnx" },
    { n: "detr-resnet-50-fp16", g: "detr", m: "tjs/facebook/detr-resnet-50/onnx/model-fp16.onnx" },
    { n: "detr-resnet-50-800", g: "detr-800", m: "tjs/facebook/detr-resnet-50/onnx/model.onnx" },
    { n: "squeezebert", g: "bert64", m: "tjs/squeezebert/squeezebert-uncased/onnx/model.onnx" },
    { n: "swin-tiny", g: "img224", m: "tjs/microsoft/swin-tiny-patch4-window7-224/onnx/model.onnx" },
    // Xenova/convnextv2-base-1k-224
    { n: "convnextv2-base-1k-224", g: "img224", m: "tjs/convnextv2-base-1k-224/onnx/model.onnx" },
    // Xenova/convnextv2-tiny-1k-224
    { n: "convnextv2-tiny-1k-224", g: "img224", m: "tjs/convnextv2-tiny-1k-224/onnx/model.onnx" },
    // Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr
    { n: "swin2SR-realworld-sr-x4-64", g: "img224", m: "tjs/swin2SR-realworld-sr-x4-64-bsrgan-psnr/onnx/model.onnx", o: "&min_query_count=3" },
    // Xenova/swin2SR-lightweight-x2-64
    { n: "swin2SR-lightweight-x2-64", g: "img224", m: "tjs/swin2SR-lightweight-x2-64/onnx/model.onnx", o: "&min_query_count=3" },
    // Xenova/vitmatte-small-distinctions-646
    { n: "vitmatte-small", g: "img4x224x224", m: "tjs/vitmatte-small-distinctions-646/onnx/model.onnx" },

    // https://github.com/Megvii-BaseDetection/YOLOX
    { n: "yolox-s", g: "img640x640", m: "partya/yolox/yolox_s.onnx" },
    { n: "yolox-l", g: "img640x640", m: "partya/yolox/yolox_l.onnx", o: "&min_query_count=5" },
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
    { n: "segformer", g: "img640x640", m: "partya/SegFormer/segformer-b5-finetuned-ade-640-640.onnx", o: "&min_query_count=5" },

    { n: "gte-small", g: "bert64", m: "gte-small/model.onnx" },
    { n: "clipseg-rd64-refined", g: "clipseg", m: "tjs/clipseg-rd64-refined/onnx/model.onnx" },
    { n: "segformer_b2_clothes", g: "img224", m: "tjs/segformer_b2_clothes/onnx/model.onnx" },

    { n: "sd-turbo-textencoder-hf", g: "sd-text-encoder", m: "onnx-sd-turbo-fp16/text_encoder/model.onnx", nowasm: true },
    { n: "sd-turbo-unet-hf", g: "sd-turbo-unet", m: "onnx-sd-turbo-fp16/unet/model.onnx", nowasm: true  },
    { n: "sd-turbo-vae-hf", g: "sd-turbo-vae", m: "onnx-sd-turbo-fp16/vae_decoder/model.onnx", nowasm: true  },
    { n: "sd-turbo-textencode-opt", g: "sd-text-encoder", m: "sd-opt/text_encoder/model.onnx", nowasm: true  },
    { n: "sd-turbo-unet-opt", g: "sd-turbo-unet", m: "sd-opt/unet/model.onnx", nowasm: true  },
    { n: "sd-turbo-vae-opt", g: "sd-turbo-vae", m: "sd-opt/vae_decoder/model.onnx", nowasm: true },

    { n: "-", g: "-", m: "p-" },
    { n: "-", g: "-", m: "p-" },
    { n: "-", g: "-", m: "p-" },

    // need to test
    { e: "new", n: "mms-tts-eng", g: "bert64", m: "tjs/mms-tts-eng/onnx/model.onnx" },

    // stable-diffusion
    { e: "sd", n: "sd-unet-fp16", g: "sd-unet", m: "sd-fp16/unet/model.onnx" },
    { e: "sd", n: "sd-vae-fp16", g: "sd-vae", m: "sd-fp16/vae_decoder/model.onnx" },
    { e: "sd", n: "lcm-vae", g: "sd-vae-fp32", m: "lcm/vae_decoder/model.onnx" },
    { e: "sd", n: "lcm-unet", g: "sd-unet-fp32", m: "lcm/unet/model.onnx" },

    // ----------- not working -----------

    // distil-whisper/distil-small.en
    { e: "error", n: "distil-whisper-decoder", g: "distil-whisper-decoder", m: "tjs/distil-whisper/distil-small.en/onnx/decoder_model_merged.onnx", o: "&seqlen=1" },
    { e: "error", n: "distil-whisper-decoder-iob", g: "distil-whisper-decoder", m: "tjs/distil-whisper/distil-small.en/onnx/decoder_model_merged.onnx", o: "&seqlen=1&io_binding=1" },
    { e: "error", n: "distil-whisper-encoder", g: "whisper-encoder", m: "tjs/distil-whisper/distil-small.en/onnx/encoder_model.onnx" },

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
