import os
import numpy as np
import json
from onnx import numpy_helper


type_dict = {
    'tensor(float16)': np.float16,
    'tensor(float)': np.float32,
    'float32': np.float32,
    'float16': np.float16,
    'tensor(double)': np.float64,
    'tensor(int32)': np.int32,
    'tensor(int8)': np.int8,
    'tensor(uint8)': np.uint8,
    'tensor(int16)': np.int16,
    'tensor(uint16)': np.uint16,
    'tensor(int64)': np.int64,
    'int64': np.int64,
    'tensor(uint64)': np.uint64,
    'tensor(bool)': bool,
    'bool': bool,
}


def named_type_to_np(name):
    return type_dict[name]


def fill(shape, dtype, val=1):
    x = np.empty(shape, dtype=dtype)
    x.fill(val)
    return x


def ramp(shape, step, dtype=np.float32, start=0.):
    n = np.prod(shape)
    stop = start + n * step
    x = np.arange(start, stop, step).reshape(shape).astype(dtype)
    return x


def create_image(shape):
    img = np.zeros(shape, dtype=np.float32)
    center_y, center_x = shape[0] // 2, shape[1] // 2
    radius = min(shape[0], shape[1]) // 16  # circle radius is 1/12 of the smallest dimension
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = (y - center_y) ** 2 + (x - center_x) ** 2 <= radius ** 2
    img[mask] = 1.0
    img = np.stack([img] * 3, axis=0)  # stack to create 3 channels
    # save image to png file using PIL
    # from PIL import Image 
    # img1 = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # convert to uint8 and transpose to HWC format
    # img1 = Image.fromarray(img1)
    # img1.save("/tmp/circle.png")
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img


def transformer(feeds, tokens, meta, kv_len=0):
    if isinstance(tokens, list):
        tokens = np.array(tokens, dtype=np.int64).reshape(1, len(tokens))
    seqlen = len(tokens[0])
    if "input_ids" in meta:
        feeds["input_ids"] = tokens
    elif "inputs_embeds" in meta:
        feeds["inputs_embeds"] = tokens
    for mask in ["attention_mask", "input_mask", "token_type_ids", "segment_ids", "encoder_attention_mask"]:
        if mask in meta:
            feeds[mask] = fill([1, seqlen], meta[mask].dtype, val=1)
    if "position_ids" in meta:
        feeds['position_ids'] = np.arange(seqlen, dtype=np.int64).reshape(1, seqlen)
    if "use_cache_branch" in meta:
        feeds["use_cache_branch"] = np.array([False], dtype=bool)
    for k, v in meta.items():
        if k.startswith("past_key_values"):
            shape = v.shape
            shape[2] = kv_len
            feeds[k] = fill(shape, v.dtype, 1.)
    return feeds


def gen_one_input(feed, name, type, meta):
    if type in ["sentence-encoder"]:
        tokens = [101, 3945, 1035, 23032, 1024, 2054, 2003, 24529, 2638, 1029, 102, 0, 0, 0, 0, 0, 101, 3945, 1035, 23032, 1024, 2040, 2003, 10294, 2015, 3158, 4315, 5003, 3686, 2078, 1029, 102]
        transformer(feed, tokens, meta)
    elif type in ["kokoro"]:
        tokens = [0, 50, 83, 54, 156, 57, 135, 4, 16, 81, 102, 61, 16, 156, 76, 158, 46, 102, 157, 57, 135, 16, 46, 147, 156, 86, 56, 85, 123, 157, 47, 102, 125, 177, 46, 16, 44, 43, 102, 16, 53, 83, 53, 156, 76, 158, 123, 57, 135, 5, 0]
        style = [-0.24795426428318024, 0.1338161677122116, 0.06700889766216278, -0.09083234518766403, -0.289242684841156, 0.262271523475647, -0.06581351906061172, -0.20899277925491333, -0.19587525725364685, 0.11188706010580063, -0.15732786059379578, 0.07457935810089111, 0.2731393575668335, 0.1515142172574997, 0.0014903922565281391, -0.3734884560108185, 0.00624218862503767, -0.16857711970806122, 0.0058748903684318066, -0.2995207607746124, -0.17109881341457367, -0.1464318484067917, 0.0958787202835083, 0.010619520209729671, -0.041546229273080826, 0.024865610525012016, -0.15756352245807648, 0.16213154792785645, -0.01941429264843464, 0.1691819578409195, 0.22503052651882172, -0.18372882902622223, -0.11051906645298004, -0.012209123000502586, -0.040588945150375366, 0.014558388851583004, -0.15434794127941132, -0.03063286282122135, 0.011701852083206177, 0.13056758046150208, -0.05477725714445114, -0.0037457970902323723, -0.07555022835731506, 0.30040261149406433, -0.29972851276397705, -0.1868869513273239, 0.09964413195848465, 0.016701819375157356, 0.0882839784026146, 0.121697336435318, -0.06211097165942192, -0.013683944940567017, 0.23499691486358643, 0.1656334102153778, 0.017871351912617683, 0.16263000667095184, 0.23277032375335693, -0.05172961950302124, -0.1148824542760849, 0.06715935468673706, 0.12813922762870789, -0.0026368864346295595, -0.18799611926078796, -0.03666650131344795, 0.06646652519702911, 0.14488264918327332, -0.04604625329375267, 0.5179643630981445, -0.11372148245573044, -0.1621483862400055, 0.14382317662239075, -0.1109749972820282, 0.039128705859184265, -0.21671780943870544, -0.38694310188293457, -0.18338650465011597, 0.03619689866900444, -0.09438976645469666, 0.18678954243659973, -0.20850075781345367, 0.21362043917179108, 0.14895914494991302, -0.20318159461021423, 0.11670200526714325, 0.24071235954761505, 0.05267072468996048, -0.048550549894571304, -0.15825942158699036, -0.24454697966575623, 0.11479644477367401, -0.015535812824964523, -0.05869020149111748, -0.0229669027030468, 0.04387282207608223, -0.10104963183403015, -0.17126299440860748, 0.04762035980820656, 0.047435320913791656, -0.06177264451980591, -0.0465497151017189, -0.21667355298995972, -0.3662424683570862, 0.10997497290372849, -0.006136247422546148, 0.15906278789043427, 0.28938817977905273, 0.05473345145583153, 0.18217052519321442, 0.2625107765197754, 0.0010020756162703037, 0.03955361992120743, -0.10128777474164963, 0.056408245116472244, 0.05765051394701004, -0.21881981194019318, 0.10425154119729996, -0.20411954820156097, 0.07677004486322403, 0.08993423730134964, -0.1667548418045044, 0.027711395174264908, 0.22852228581905365, -0.1607397347688675, 0.24704760313034058, 0.08645891398191452, -0.22631043195724487, 0.1332206279039383, -0.26941365003585815, -0.12412530183792114, 0.2368716299533844, 0.2130863070487976, 0.16018088161945343, -0.3362795412540436, 0.05821998417377472, 0.44190144538879395, -0.0071663158014416695, -0.06984671205282211, -0.40545058250427246, 0.13538339734077454, -0.18319948017597198, 0.2524869740009308, -0.05552295595407486, -0.09272931516170502, -0.12234407663345337, -0.0025546224787831306, 0.17200878262519836, 0.1301029771566391, -0.2586514949798584, 0.20385591685771942, -0.11571730673313141, -0.05336975306272507, 0.3135416507720947, -0.18703360855579376, 0.011971375904977322, -0.006857029162347317, -0.13974881172180176, -0.27306312322616577, 0.16871687769889832, 0.17794503271579742, -0.11541739106178284, 0.1430143266916275, 0.1489647626876831, 0.0531061589717865, 0.4186581075191498, 0.41715171933174133, 0.2523469030857086, 0.06060232222080231, -0.025356024503707886, 0.012731916271150112, -0.0928533598780632, 0.1879071295261383, 0.2503766715526581, -0.25279220938682556, 0.2461116909980774, -0.20918074250221252, -0.5854116082191467, -0.30969947576522827, 0.07790165394544601, 0.1912315934896469, -0.09404738247394562, 0.1484833061695099, 0.18205595016479492, 0.07462356984615326, -0.017541563138365746, 0.5140320658683777, -0.2232881486415863, 0.01817263290286064, -0.10544224083423615, -0.13483119010925293, -0.25258776545524597, -0.17391861975193024, 0.11897670477628708, 0.17084628343582153, 0.3620717525482178, 0.03714647889137268, 0.014582810923457146, 0.24124981462955475, 0.19698452949523926, -0.01561416033655405, -0.0057082874700427055, 0.03306499496102333, -0.42797666788101196, -0.48292168974876404, 0.22070863842964172, 0.36470723152160645, 0.18103823065757751, 0.23546016216278076, 0.031840190291404724, 0.024735674262046814, 0.18023765087127686, -0.03965626284480095, -0.14427220821380615, -0.28584444522857666, -0.10464715957641602, 0.022148240357637405, -0.1133788600564003, 0.14148995280265808, -0.0038788672536611557, -0.04177098721265793, 0.3123432695865631, 0.17356684803962708, 0.18005062639713287, -0.2717491686344147, 0.10513272881507874, -0.2739059031009674, 0.10081014037132263, -0.031733717769384384, 0.029638178646564484, -0.07570099085569382, 0.23769965767860413, -0.09160297363996506, 0.17463311553001404, -0.11542663723230362, 0.3706754148006439, 0.18904975056648254, -0.29093438386917114, -0.01883920654654503, -0.29375627636909485, -0.08776254206895828, 0.22491639852523804, -0.14468906819820404, 0.08872667700052261, 0.09172000735998154, 0.4222434461116791, -0.21699510514736176, 0.06316738575696945, -0.004107693675905466, 0.3961399793624878, -0.0665103942155838, -0.3044799566268921, -0.21313011646270752, 0.5354771614074707, 0.07801689952611923, -0.359889954328537, -0.12797550857067108, 0.13279223442077637]
        feed['input_ids'] = np.array(tokens, dtype=np.int64).reshape([1, len(tokens)])
        feed['style'] = np.array(style, dtype=np.float32).reshape([1, len(style)])
        # feeds['input_ids'] = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64).reshape([1, 8])
        # feeds['style'] = fill([1, 256], np.float32, 0.5)
        feed['speed'] = fill([1], np.float32, 1)
    elif type in ["sam", "sam-static", "sam-decoder"]:
        feed["input_points"] = np.array([327.1111, 426.875, 241.77777, 341.5, 398.22223, 498.02084], dtype=np.float32).reshape([1, 1, 3, 2])
        feed["input_labels"] = np.array([0., 2., 3.], dtype=np.int64).reshape([1, 1, 3])
        feed["image_embeddings"] = np.random.randn(1, 256, 64, 64).astype("float32")
        feed["image_positional_embeddings"] = fill([1, 256, 64, 64], np.float32, 0.5)
    elif type in ["sam-encoder", "slimsam-encoder"]:
        feed["pixel_values"] = fill([1, 3, 1024, 1024], np.float32, 0.5)
    elif type == "slimsam-decoder":
        feed["image_embeddings"] = fill([1, 256, 64, 64], np.float32, 0.5)
        feed["image_positional_embeddings"] = fill([1, 256, 64, 64], np.float32, 0.5)
        feed["input_labels"] = fill([1, 1, 1], np.int64, 1)
        feed["input_points"] = np.array([400., 500.], dtype=np.float32).reshape([1, 1, 1, 2])
    elif type == "llm-decoder":
        tokens = fill([1, 128], np.int64, 99)
        transformer(feed, tokens, meta)
    elif type == "nomic-embed":
        tokens = [101, 3945, 1035, 23032, 1024, 2054, 2003, 24529, 2638, 1029, 102, 0, 0, 0, 0, 0, 101, 3945, 1035, 23032, 1024, 2040, 2003, 10294, 2015, 3158, 4315, 5003, 3686, 2078, 1029, 102]
        transformer(feed, tokens, meta)
    elif type == "phi2":
        tokens = [24446, 502, 546, 262, 46371, 286, 27872]
        transformer(feed, tokens, meta)
    elif type == "phi3":
        tokens = [1, 24948, 592, 1048, 23194, 1991, 29889]
        transformer(feed, tokens, meta)
    elif type == "gemma3n-emb":
        tokens = [2, 105, 2364, 107, 54593, 786, 1003, 127086, 236761, 106, 107, 105, 4368, 107]
        feed["input_ids"] = np.array(tokens, dtype=np.int64).reshape([1, len(tokens)])
    elif type == "gemma3n-decoder":
        seqlen = 14
        tokens = fill([1, seqlen, 2048], meta['inputs_embeds'].dtype, 0.5)
        feed["per_layer_inputs"] = fill([1, seqlen, 30, 256], meta['per_layer_inputs'].dtype, 0.5)
        transformer(feed, tokens, meta)
    elif type == "gemma3n-audio":
        seqlen = 1024
        feed["input_features"] = fill([1, seqlen, 128], meta['input_features'].dtype, 0.5)
        feed["input_features_mask"] = fill([1, seqlen], meta['input_features_mask'].dtype, True)
    elif type == "gemma3n-vision":
        feed["pixel_values"] = fill([1, 3, 768, 768], meta['pixel_values'].dtype, 0.5)
    elif type == "florence-embed":
        tokens = [0, 35438, 162, 59, 5, 2274, 2]
        transformer(feed, tokens, meta)
    elif type == "florence-decoder":
        tokens = fill([1, 128, 768], meta['inputs_embeds'].dtype, 0.5)
        feed["encoder_hidden_states"] = fill([1, 128, 768], meta['encoder_hidden_states'].dtype, 0.5)
        transformer(feed, tokens, meta)
    elif type == "florence-encoder":
        tokens = fill([1, 128, 768], meta['inputs_embeds'].dtype, 0.5)
        transformer(feed, tokens, meta)
    elif type in ["tinyllama", "tinyllama-fp16", "stablelm", "stablelm-fp16"]:
        tokens = [529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 19780, 20255, 29889, 2, 13, 29966,
                  29989, 1792, 29989, 29958, 13, 29911, 514, 592, 1048, 23194, 1991, 29889, 2, 13, 29966, 29989,
                  465, 22137, 29989, 29958, 13]
        transformer(feed, tokens, meta)
    elif type in ["opus-mt-mul-en-decoder"]:
        seqlen = 128
        tokens = fill([1, seqlen], np.int64, 99)
        transformer(feed, tokens, meta, kv_len=seqlen)
        feed['encoder_hidden_states'] = fill([1, seqlen, 512], np.float32, 1)
    elif type in ["t5-decoder", "flan-t5-decoder"]:
        seqlen = 1 if type == "t5-decoder" else 1
        tokens = fill([1, seqlen], np.int64, 99)
        transformer(feed, tokens, meta, kv_len=seqlen)
        feed['encoder_hidden_states'] = fill([1, 128, 512], np.float32, 1)
    elif type in ["opus-mt-mul-en-encoder"]:
        seqlen = 128
        tokens = fill([1, seqlen], np.int64, 99)
        transformer(feed, tokens, meta, kv_len=seqlen)
    elif type in ["t5-encoder", "flan-t5-encoder"]:
        seqlen = 128
        tokens = fill([1, seqlen], np.int64, 99)
        transformer(feed, tokens, meta, kv_len=seqlen)
    elif type in ["vit", "dino"]:
        feed["pixel_values"] = np.random.randn(1, 3, 224, 224).astype("float32")
    elif type == "detr":
        feed["pixel_values"] = np.random.randn(1, 3, 224, 224).astype("float32")
        if "pixel_mask" in meta:
            feed["pixel_mask"] = fill([1, 64, 64], np.int64, 1)
    elif type == "detr-800":
        feed["pixel_values"] = np.random.randn(1, 3, 800, 800).astype("float32")
        if "pixel_mask" in meta:
            feed["pixel_mask"] = fill([1, 64, 64], np.int64, 1)
    elif type == "detr-640x480":
        feed["pixel_values"] = np.random.randn(1, 3, 640, 480).astype("float32")
    elif type == "detr-560x560":
        feed["pixel_values"] = np.random.randn(1, 3, 560, 560).astype("float32")
    elif type in ["clip", "clipseg"]:
        feed["input_ids"] = fill([1, 77], np.int64, 49407)
        feed["pixel_values"] = fill([1, 3, 224, 224], np.float32, 99)
        feed["attention_mask"] = fill([1, 77], np.int64, 1)
    elif type == "vti":
        feed["input"] = fill([1, 3, 224, 224], np.float32, 99)
    elif type == "vae-encoder":
        feed["sample"] = np.random.randn(1, 3, 512, 512).astype("float16")
    elif type == "vae-encoder-fp32":
        feed["sample"] = np.random.randn(1, 3, 512, 512).astype("float32")
    elif type in ["vae-decoder", "sd-vae-decoder"]:
        feed["latent_sample"] = np.ones([1, 4, 64, 64], dtype=np.float16)  # np.random.randn(1, 4, 64, 64).astype("float32")
    elif type in ["vae-decoder-fp32", "sd-vae-decoder-fp32"]:
        feed["latent_sample"] = np.ones([1, 4, 64, 64], dtype=np.float32)
    elif type == "sd-unet":
        feed["sample"] = np.random.randn(1, 4, 64, 64).astype("float16")
        feed["timestep"] = np.array([0]).astype("int64")
        feed["encoder_hidden_states"] = np.random.randn(1, 77, 768).astype("float16")
    elif type == "sd-unet-fp32":
        feed["sample"] = np.random.randn(1, 4, 64, 64).astype("float32")
        feed["timestep"] = np.array([999]).astype("int64")
        feed["encoder_hidden_states"] = np.random.randn(1, 77, 1024).astype("float32")
    elif type == "sd-text-encoder":
        feed["input_ids"] = np.array([[49406, 550, 9899, 40071, 525, 320, 7998, 267, 1400, 268, 3027, 267, 741, 16125, 267, 2870, 525, 7483,
                                        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                                        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                                        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
                                        49407, 49407, 49407, 49407, 49407]], dtype=np.int32)
    elif type == "sd-safety-checker":
        feed["clip_input"] = np.random.randn(1, 3, 224, 224).astype("float32")
        feed["images"] = np.random.randn(1, 512, 512, 3).astype("float32")
    elif type == "sd-turbo-unet":
        feed["sample"] = fill([1, 4, 64, 64], np.float32, 0.5)
        feed["timestep"] = fill([1], np.int64, 999)
        feed["encoder_hidden_states"] = fill([1, 77, 1024], np.float32, 0.5)
    elif type == "sd-turbo-vae":
        feed["latent_sample"] = fill([1, 4, 64, 64], np.float32, 0.5)
    elif type == "circle640":
        feed[name] = create_image([640, 640])
    elif type == "img64":
        feed[name] = fill([1, 3, 64, 64], meta[name].dtype)
    elif type == "whisper-encoder":
        feed["input_features"] = fill([1, 80, 3000], meta["input_features"].dtype)
    elif type == "whisper-turbo-encoder":
        feed["input_features"] = fill([1, 128, 3000], meta["input_features"].dtype)
    elif type in ["whisper-decoder", "hf-whisper-decoder", "hf-whisper-decoder-fp16"]:
        tokens = fill([1, 448], np.int64)
        transformer(feed, tokens, meta)
        feed["encoder_hidden_states"] = fill([1, 1500, 384], meta["encoder_hidden_states"].dtype)
    elif type in ["whisper-turbo-decoder"]:
        tokens = fill([1, 448], np.int64)
        transformer(feed, tokens, meta)
        feed["encoder_hidden_states"] = fill([1, 1500, 1280], meta["encoder_hidden_states"].dtype)
    elif type == "moonshine-encoder":
        feed["input_values"] = fill([1, 3000], meta["input_values"].dtype)
    elif type in ["moonshine-decoder"]:
        tokens = fill([1, 448], np.int64, 99.)
        transformer(feed, tokens, meta)
        feed["encoder_hidden_states"] = fill([1, 1500, 288], meta["encoder_hidden_states"].dtype)
    elif type == "silero-vad":
        feed["input"] = fill([1, 512], meta['input'].dtype, 1.)
        feed["sr"] = fill([1], meta['sr'].dtype, 16000)
        feed["state"] = fill([2, 1, 128], meta['state'].dtype, 1.)
    elif type == "img224":
        feed[name] = fill([1, 3, 224, 224], meta[name].dtype)
    elif type in ["img640x640", "img640"]:
        feed[name] = fill([1, 3, 640, 640], meta[name].dtype, 128)
    elif type == "img640x480":
        feed[name] = fill([1, 3, 640, 480], meta[name].dtype)
    elif type in ["img512x512", "img512"]:
        feed[name] = fill([1, 3, 512, 512], meta[name].dtype)
    elif type == "img224":
        feed[name] = fill([1, 3, 224, 224], meta[name].dtype)
    elif type == "img1024":
        feed[name] = fill([1, 3, 1024, 1024], meta[name].dtype)
    elif type == "img768":
        feed[name] = fill([1, 3, 768, 768], meta[name].dtype)
    elif type == "img4x224x224":
        feed[name] = fill([1, 4, 224, 224], meta[name].dtype)
    elif type == "img416x416":
        feed[name] = fill([1, 3, 416, 416], meta[name].dtype)
    elif type == "fp32x100k":
        feed[name] = ramp([1, 100 * 1024], 1, np.float32, 0)
        print(feed[name].shape)
    elif type in ["bert", "bert64"]:
        bert_shape = [1, 128]
        tokens = fill(bert_shape, meta[name].dtype)
        transformer(feed, tokens, meta)
    elif type in ["bart-large-12"]:
        bert_shape = [1, 128]
        tokens = fill(bert_shape, meta[name].dtype)
        transformer(feed, tokens, meta)
        feed["encoder_hidden_states"] = fill([1, 128, 768], np.float32, 1.)
    elif type in ["bart-cnn"]:
        bert_shape = [1, 128]
        tokens = fill(bert_shape, meta[name].dtype)
        transformer(feed, tokens, meta)
        feed["encoder_hidden_states"] = fill([1, 128, 1024], np.float32, 1.)
    elif type in ["512x512"]:
        feed[name] = fill([512, 512], np.float32, val=1)
    elif type in ["lama-onnx"]:
        feed['image'] = fill([1, 3, 512, 512], np.float32, val=1)
        feed['mask'] = fill([1, 1, 512, 512], np.float32, val=1)
    elif type in ["foo"]:
        feed['pre_x'] = create_image([2160, 3840])
        # feed['pre_x'] = fill([1, 3, 2160, 3840], "float32", 0.5)
        feed['orig_target_sizes'] = np.array([640, 640], dtype=np.int64).reshape(1, 2)
    else:
        raise NotImplementedError("don't know how to generate data for " + type)


class metadata:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = named_type_to_np(dtype)
        self.shape = shape


def gen_data(sess, type, verbose):
    feeds = {}
    if type and type.startswith("data/"):
        with open(type) as f:
            js = json.load(f)
            for name, item in js.items():
                shape = item['shape']
                t = np.array(item['data']).astype(item['dtype']).reshape(shape)
                feeds[name] = t
        return feeds

    meta = {}
    name = None
    for input_meta in sess.get_inputs():
        v = metadata(input_meta.name, input_meta.type, input_meta.shape)
        meta[v.name] = v
        if not name:
            name = input_meta.name
        for i, dim in enumerate(v.shape):
            if dim == "batch_size":
                v.shape[i] = 1
    gen_one_input(feeds, name, type, meta)
    return feeds


def save_protobuf(path, message):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "wb") as f:
        f.write(message.SerializeToString())


def save_feed(name, inputs, outputs):
    for i, data_key in enumerate(inputs):
        data = inputs[data_key]
        t = numpy_helper.from_array(data)
        t.name = data_key
        data_full_path = os.path.join(
            name, "test_data_set_0", "input_" + str(i) + ".pb")
        save_protobuf(data_full_path, t)

    for i, data_key in enumerate(outputs):
        data = outputs[data_key]
        t = numpy_helper.from_array(data)
        t.name = data_key
        data_full_path = os.path.join(
            name, "test_data_set_0", "output_" + str(i) + ".pb")
        save_protobuf(data_full_path, t)


def io_list_to_dict(sess, outputs):
    feed = {}
    output_names = list(map(lambda x: x.name, sess.get_outputs()))
    for i, s in enumerate(outputs):
        feed[output_names[i]] = s
    return feed


def dump_as_json(feed, fname, dumpkv=False, namelist=None):
    j = {}
    for k, s in feed.items():
        if namelist is None or k in namelist:
            if dumpkv is False and (k.startswith("past_key_values") or k.startswith("present")):
                continue
            print(f"dumping {k}")
            v = s.flatten().tolist()
            j[k] = {
                "dtype": str(s.dtype),
                "shape": list(s.shape),
                "output-avg": float(s.mean()),
                "data": list(v),
            }
    with open(fname, "w") as f:
        f.write(json.dumps(j, indent=2))


RTOL = 0.01
ATOL = 0.01


def validate_as_json(feed, fname, rtol=RTOL, atol=ATOL):
    with open(fname, "r") as f:
        j = json.load(f)

    for k, v in feed.items():
        v2 = np.array(j[k]['data']).astype(j[k]['dtype']).reshape(j[k]['shape'])
        ret = np.allclose(v, v2, rtol, atol)
        print(f"{k}: {ret}")
