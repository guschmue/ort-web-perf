import * as ort from 'onnxruntime-web';


function fillTensor(shape, dtype, val = 1) {
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });
    switch (dtype) {
        case "float32":
            return new ort.Tensor(Float32Array.from({ length: size }, () => val), shape);
        case "float16":
            return new ort.Tensor("float16", Uint16Array.from({ length: size }, () => val), shape);
        case "int32":
            return new ort.Tensor(Int32Array.from({ length: size }, () => val), shape);
        case "int64":
            if (typeof (val) != BigInt) {
                val = BigInt(val)
            }
            return new ort.Tensor(BigInt64Array.from({ length: size }, () => val), shape);
    }
    throw new Error(`Input tensor type ${dtype} is not supported`);
}

function randomTensor(shape, dtype) {
    let size = 1;
    shape.forEach(element => {
        if (element > 1) {
            size *= element;
        }
    });
    switch (dtype) {
        case "float32":
            return new ort.Tensor(Float32Array.from({ length: size }, () => Math.random()), shape);
    }
    throw new Error(`Input tensor type ${dtype} is not supported`);
}

function rampTensor(shape, dtype) {
    let size = 1;
    shape.forEach(element => {
        if (element > 1) {
            size *= element;
        }
    });
    switch (dtype) {
        case "float32":
            return new ort.Tensor(Float32Array.from({ length: size }, (_, i) => i), shape);
        case "int32":
            return new ort.Tensor(Int32Array.from({ length: size }, (_, i) => i), shape);
        case "int64":
            return new ort.Tensor(BigInt64Array.from({ length: size }, (_, i) => BigInt(i)), shape);
    }
    throw new Error(`Input tensor type ${dtype} is not supported`);
}

function dump_feed(feed) {
    for (const [name, t] of Object.entries(feed)) {
        log(`${name}: ${t.type} ${t.dims}`);
    }
}

async function data_from_json(feed, file) {
    const resp = await fetch(file);
    const json = await resp.json();
    for (const [name, item] of Object.entries(json)) {
        let t;
        switch (item.dtype) {
            case "float32":
                t = new ort.Tensor(new Float32Array(item.data), item.shape);
                break;
            case "float16":
                t = new ort.Tensor("float16", new Uint16Array(item.data), item.shape);
                break;
            case "int32":
                t = new ort.Tensor(new Int32Array(item.data), item.shape);
                break;
            case "int8":
                t = new ort.Tensor(new Int8Array(item.data), item.shape);
                break;
                case "int64":
                t = new ort.Tensor(new BigInt64Array(item.data.map(BigInt)), item.shape);
                break;
            default:
                throw new Error(`unsupported type ${item.dtype}`);
        }
        feed[name] = t;
    }
    return feed;
}

async function createImage(width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    // Calculate radius so that the circle occupies 25% of the image area
    const imageArea = width * height;
    const circleArea = imageArea * 0.25;
    const radius = Math.sqrt(circleArea / Math.PI);

    // Draw circle in the center
    ctx.beginPath();
    ctx.arc(width / 2, height / 2, radius, 0, 2 * Math.PI);
    ctx.fillStyle = 'black'; // or any color you want
    ctx.fill();
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const tensor = await ort.Tensor.fromImage(imageData);
    return tensor;
}

export async function make_inputs(gen, inputNames, config) {
    const seqlen = config.seqlen;
    const enc_seqlen = config.enc_seqlen;
    const feed = {};
    const name = inputNames[0];

    if (gen.startsWith("data/")) {
        return data_from_json(feed, gen);
    }

    if (gen == "gqa") {
        // GroupQueryAttention.float16  [1, 37, 2048], [1, 37, 512], [1, 37, 512], [1, 8, 0, 64], [1, 8, 0, 64], [1, 1], []
        // inp: Q, K, V, past_key (optional), past_value (optional), seqlens_k, total_sequence_length
        // out: output, present_key, present_value
    }
    if (gen == "gatherblock") {
        const data = [1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n, 9n, 10n, 11n, 12n, 13n, 14n];
        feed['X'] = new ort.Tensor(new BigInt64Array(data), [1, data.length], "int64");
        return feed;
    }
    if (gen == "avgpool") {
        const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        feed['X'] = new ort.Tensor(new Float32Array(data), [1, 1, 4, 4], "float32");
        return feed;
    }
    if (gen == "SmolVLM-Instruct") {
        const tokens = [101n];
        // feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['inputs_embeds'] = fillTensor([1, 1, 2048], "float32", 1.);
        const decoder_shape = [1, 32, 0, 64];
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                feed[v] = fillTensor(decoder_shape, "float32", 1);
            }
        }
        return feed;
    }

    if (gen == "nomic-embed") {
        const tokens = [101n, 3945n, 1035n, 23032n, 1024n, 2054n, 2003n, 24529n, 2638n, 1029n, 102n, 0n, 0n, 0n, 0n, 0n, 101n, 3945n, 1035n, 23032n, 1024n, 2040n, 2003n, 10294n, 2015n, 3158n, 4315n, 5003n, 3686n, 2078n, 1029n, 102n];
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['token_type_ids'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        return feed;
    }
    if (gen == "tiny-random-llama") {
        const decoder_shape = [1, 4, 0, 4];
        const tokens = [529n, 29989n, 5205n, 29989n, 29958n, 13n, 3492n, 526n, 263n, 19780n, 20255n, 29889n, 2n, 13n, 29966n,
            29989n, 1792n, 29989n, 29958n, 13n, 29911n, 514n, 592n, 1048n, 23194n, 1991n, 29889n, 2n, 13n, 29966n, 29989n,
            465n, 22137n, 29989n, 29958n, 13n];
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['position_ids'] = rampTensor([1, tokens.length], "int64");
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                feed[v] = fillTensor(decoder_shape, "float32", 1);
            }
        }
        return feed;
    }
    if (["tinyllama", "tinyllama-fp16", "stablelm",  "stablelm-fp16"].includes(gen)) {
        const dtype = (gen == "tinyllama-fp16" || gen == "stablelm-fp16") ? "float16" : "float32";
        const decoder_shape = (gen.startsWith("stablelm")) ?  [1, 32, 0, 64] :  [1, 4, 0, 64];
        const tokens = [529n, 29989n, 5205n, 29989n, 29958n, 13n, 3492n, 526n, 263n, 19780n, 20255n, 29889n, 2n, 13n, 29966n,
            29989n, 1792n, 29989n, 29958n, 13n, 29911n, 514n, 592n, 1048n, 23194n, 1991n, 29889n, 2n, 13n, 29966n, 29989n,
            465n, 22137n, 29989n, 29958n, 13n];
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['position_ids'] = rampTensor([1, tokens.length], "int64");

        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                feed[v] = fillTensor(decoder_shape, dtype, 1);
            }
        }
        return feed;
    }
    if (gen == "kokoro") {
        const tokens = [0, 50, 83, 54, 156, 57, 135, 4, 16, 81, 102, 61, 16, 156, 76, 158, 46, 102, 157, 57, 135, 16, 46, 147, 156, 86, 56, 85, 123, 157, 47, 102, 125, 177, 46, 16, 44, 43, 102, 16, 53, 83, 53, 156, 76, 158, 123, 57, 135, 5, 0];
        const style = [-0.24795426428318024, 0.1338161677122116, 0.06700889766216278, -0.09083234518766403, -0.289242684841156, 0.262271523475647, -0.06581351906061172, -0.20899277925491333, -0.19587525725364685, 0.11188706010580063, -0.15732786059379578, 0.07457935810089111, 0.2731393575668335, 0.1515142172574997, 0.0014903922565281391, -0.3734884560108185, 0.00624218862503767, -0.16857711970806122, 0.0058748903684318066, -0.2995207607746124, -0.17109881341457367, -0.1464318484067917, 0.0958787202835083, 0.010619520209729671, -0.041546229273080826, 0.024865610525012016, -0.15756352245807648, 0.16213154792785645, -0.01941429264843464, 0.1691819578409195, 0.22503052651882172, -0.18372882902622223, -0.11051906645298004, -0.012209123000502586, -0.040588945150375366, 0.014558388851583004, -0.15434794127941132, -0.03063286282122135, 0.011701852083206177, 0.13056758046150208, -0.05477725714445114, -0.0037457970902323723, -0.07555022835731506, 0.30040261149406433, -0.29972851276397705, -0.1868869513273239, 0.09964413195848465, 0.016701819375157356, 0.0882839784026146, 0.121697336435318, -0.06211097165942192, -0.013683944940567017, 0.23499691486358643, 0.1656334102153778, 0.017871351912617683, 0.16263000667095184, 0.23277032375335693, -0.05172961950302124, -0.1148824542760849, 0.06715935468673706, 0.12813922762870789, -0.0026368864346295595, -0.18799611926078796, -0.03666650131344795, 0.06646652519702911, 0.14488264918327332, -0.04604625329375267, 0.5179643630981445, -0.11372148245573044, -0.1621483862400055, 0.14382317662239075, -0.1109749972820282, 0.039128705859184265, -0.21671780943870544, -0.38694310188293457, -0.18338650465011597, 0.03619689866900444, -0.09438976645469666, 0.18678954243659973, -0.20850075781345367, 0.21362043917179108, 0.14895914494991302, -0.20318159461021423, 0.11670200526714325, 0.24071235954761505, 0.05267072468996048, -0.048550549894571304, -0.15825942158699036, -0.24454697966575623, 0.11479644477367401, -0.015535812824964523, -0.05869020149111748, -0.0229669027030468, 0.04387282207608223, -0.10104963183403015, -0.17126299440860748, 0.04762035980820656, 0.047435320913791656, -0.06177264451980591, -0.0465497151017189, -0.21667355298995972, -0.3662424683570862, 0.10997497290372849, -0.006136247422546148, 0.15906278789043427, 0.28938817977905273, 0.05473345145583153, 0.18217052519321442, 0.2625107765197754, 0.0010020756162703037, 0.03955361992120743, -0.10128777474164963, 0.056408245116472244, 0.05765051394701004, -0.21881981194019318, 0.10425154119729996, -0.20411954820156097, 0.07677004486322403, 0.08993423730134964, -0.1667548418045044, 0.027711395174264908, 0.22852228581905365, -0.1607397347688675, 0.24704760313034058, 0.08645891398191452, -0.22631043195724487, 0.1332206279039383, -0.26941365003585815, -0.12412530183792114, 0.2368716299533844, 0.2130863070487976, 0.16018088161945343, -0.3362795412540436, 0.05821998417377472, 0.44190144538879395, -0.0071663158014416695, -0.06984671205282211, -0.40545058250427246, 0.13538339734077454, -0.18319948017597198, 0.2524869740009308, -0.05552295595407486, -0.09272931516170502, -0.12234407663345337, -0.0025546224787831306, 0.17200878262519836, 0.1301029771566391, -0.2586514949798584, 0.20385591685771942, -0.11571730673313141, -0.05336975306272507, 0.3135416507720947, -0.18703360855579376, 0.011971375904977322, -0.006857029162347317, -0.13974881172180176, -0.27306312322616577, 0.16871687769889832, 0.17794503271579742, -0.11541739106178284, 0.1430143266916275, 0.1489647626876831, 0.0531061589717865, 0.4186581075191498, 0.41715171933174133, 0.2523469030857086, 0.06060232222080231, -0.025356024503707886, 0.012731916271150112, -0.0928533598780632, 0.1879071295261383, 0.2503766715526581, -0.25279220938682556, 0.2461116909980774, -0.20918074250221252, -0.5854116082191467, -0.30969947576522827, 0.07790165394544601, 0.1912315934896469, -0.09404738247394562, 0.1484833061695099, 0.18205595016479492, 0.07462356984615326, -0.017541563138365746, 0.5140320658683777, -0.2232881486415863, 0.01817263290286064, -0.10544224083423615, -0.13483119010925293, -0.25258776545524597, -0.17391861975193024, 0.11897670477628708, 0.17084628343582153, 0.3620717525482178, 0.03714647889137268, 0.014582810923457146, 0.24124981462955475, 0.19698452949523926, -0.01561416033655405, -0.0057082874700427055, 0.03306499496102333, -0.42797666788101196, -0.48292168974876404, 0.22070863842964172, 0.36470723152160645, 0.18103823065757751, 0.23546016216278076, 0.031840190291404724, 0.024735674262046814, 0.18023765087127686, -0.03965626284480095, -0.14427220821380615, -0.28584444522857666, -0.10464715957641602, 0.022148240357637405, -0.1133788600564003, 0.14148995280265808, -0.0038788672536611557, -0.04177098721265793, 0.3123432695865631, 0.17356684803962708, 0.18005062639713287, -0.2717491686344147, 0.10513272881507874, -0.2739059031009674, 0.10081014037132263, -0.031733717769384384, 0.029638178646564484, -0.07570099085569382, 0.23769965767860413, -0.09160297363996506, 0.17463311553001404, -0.11542663723230362, 0.3706754148006439, 0.18904975056648254, -0.29093438386917114, -0.01883920654654503, -0.29375627636909485, -0.08776254206895828, 0.22491639852523804, -0.14468906819820404, 0.08872667700052261, 0.09172000735998154, 0.4222434461116791, -0.21699510514736176, 0.06316738575696945, -0.004107693675905466, 0.3961399793624878, -0.0665103942155838, -0.3044799566268921, -0.21313011646270752, 0.5354771614074707, 0.07801689952611923, -0.359889954328537, -0.12797550857067108, 0.13279223442077637];
        feed['input_ids'] = new ort.Tensor('int64', tokens, [1, tokens.length],);
        feed['style'] = new ort.Tensor('float32', style, [1, style.length],);
        feed['speed'] = new ort.Tensor('float32', [1], [1],);
        return feed;
    }
    if (gen == "phi2") {
        const tokens = [24446n, 502n, 546n, 262n, 46371n, 286n, 27872n];
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['position_ids'] = rampTensor([1, tokens.length], "int64");
        const decoder_shape = [1, 32, 0, 80];
        for (var i = 0; i < 32; i++) {
            feed['past_key_values.' + i + '.key'] = fillTensor(decoder_shape, 'float16', 1);
            feed['past_key_values.' + i + '.value'] = fillTensor(decoder_shape, 'float16', 1);
        }
        return feed;
    }
    if (gen == "reader-1.5b") {
        const tokens = [13745n, 1784n, 2599n, 1784n, 71n, 16n, 79497n, 11n, 1879n, 18685n, 71n, 16n, 1472n, 2599n, 1472n, 1551n, 29n];
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['position_ids'] = rampTensor([1, tokens.length], "int64");
        const decoder_shape = [1, 2, 0, 128];
        for (var i = 0; i < 28; i++) {
            feed['past_key_values.' + i + '.key'] = fillTensor(decoder_shape, 'float16', 1);
            feed['past_key_values.' + i + '.value'] = fillTensor(decoder_shape, 'float16', 1);
        }
        return feed;
    }
    if (gen == "phi3") {
        const tokens = [1n, 24948n, 592n, 1048n, 23194n, 1991n, 29889n]
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['position_ids'] = rampTensor([1, tokens.length], "int64");
        const decoder_shape = [1, 32, 0, 96];
        for (var i = 0; i < 32; i++) {
            feed['past_key_values.' + i + '.key'] = fillTensor(decoder_shape, 'float16', 1);
            feed['past_key_values.' + i + '.value'] = fillTensor(decoder_shape, 'float16', 1);
        }
        return feed;
    }
    if (gen == "gemma-2-2b") {
        const tokens = [2n, 27445n, 682n, 1105n, 89615n, 235265n]
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);
        feed['position_ids'] = rampTensor([1, tokens.length], "int64");
        const decoder_shape = [1, 4, 0, 256];
        for (var i = 0; i < 26; i++) {
            feed['past_key_values.' + i + '.key'] = fillTensor(decoder_shape, 'float16', 1);
            feed['past_key_values.' + i + '.value'] = fillTensor(decoder_shape, 'float16', 1);
        }
        return feed;
    }
    if (gen == "bge-m3") {
        const tokens = [0n, 4865n, 83n, 335n, 11679n, 276n, 363n, 32n, 2n, 0n, 262n, 5983n, 2320n, 111n, 90017n, 2588n, 2n, 1n];
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['attention_mask'] = fillTensor([1, tokens.length], "int64", 1n);

    }
    if (gen == "phi3-v-vision") {
        const tokens = [336n, 336n];
        feed['image_sizes'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        feed['pixel_values'] = fillTensor([1, 17, 3, 336, 336], "float32", 1.);
        return feed;
    }
    if (gen == "sd-text-encoder") {
        const ids = [
            49406, 320, 25602, 2000, 539, 320, 1794, 1773, 7100, 3309, 550, 29616, 5175, 14222, 23116, 269, 49407,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        feed["input_ids"] = new ort.Tensor("int32", new Int32Array(ids), [1, 77]);
        return feed;
    }
    if (gen == "sd-turbo-unet") {
        feed["sample"] = fillTensor([1, 4, 64, 64], "float32");
        feed["timestep"] = fillTensor([1], "int64", 999n);
        feed["encoder_hidden_states"] = fillTensor([1, 77, 1024], "float32");
        return feed;
    }
    if (gen == "sd-turbo-vae") {
        feed["latent_sample"] = fillTensor([1, 4, 64, 64], "float32", 9);
        return feed;
    }
    if (["sd-unet-fp32", "sd-unet-fp16", "sd-unet-fp16-webml"].includes(gen)) {
        const dtype = (gen == "sd-unet-fp32") ? "float32" : "float16";
        const bs = (gen == "sd-unet-fp16-webml") ? 2 : 1;
        feed["sample"] = fillTensor([bs, 4, 64, 64], dtype);
        feed["timestep"] = (gen == "sd-unet-fp16-webml") ? fillTensor([2], "int64", 1n) : fillTensor([1], "int64", 1n);
        feed["encoder_hidden_states"] = fillTensor([bs, 77, 768], dtype);
        return feed;
    }
    if (["sd-vae-fp32", "sd-vae-fp16"].includes(gen)) {
        const dtype = (gen == "sd-vae-fp32") ? "float32" : "float16";
        feed["latent_sample"] = fillTensor([1, 4, 64, 64], dtype);
        return feed;
    }
    if (gen == "vit") {
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.includes("_values")) {
                feed[v] = rampTensor([1, 3, 224, 224], "float32");
            } else if (v == "attention_mask") {
                feed[v] = fillTensor([1, seqlen], "int64", 1n);
            } else {
                feed[v] = fillTensor([1, seqlen], "int64", 49407n);
            }
        }
        return feed;
    }
    if (gen == "vti") {
        feed["input"] = fillTensor([1, 3, 224, 224], "float32", 99);
        return feed;
    }
    if (gen == "clip" || gen == "clipseg") {
        feed["input_ids"] = fillTensor([1, 77], "int64", 49407n);
        feed["pixel_values"] = fillTensor([1, 3, 224, 224], "float32", 99);
        feed["attention_mask"] = fillTensor([1, 77], "int64", 1n);
        return feed;
    }
    if (gen == "detr") {
        feed["pixel_values"] = fillTensor([1, 3, 224, 224], "float32", 99);
        feed["pixel_mask"] = fillTensor([1, 64, 64], "int64", 1n);
        return feed;
    }
    if (gen == "detr-800") {
        feed["pixel_values"] = fillTensor([1, 3, 800, 800], "float32", 99);
        feed["pixel_mask"] = fillTensor([1, 64, 64], "int64", 1n);
        return feed;
    }
    if (gen == "mobilevit") {
        feed["pixel_values"] = fillTensor([1, 3, 224, 224], "float32", 99);
        return feed;
    }
    if (gen == "wav2vec") {
        feed["input_values"] = fillTensor([1, 512], "float32");
        return feed;
    }
    if (gen == "wav2vec2") {
        feed["input"] = fillTensor([1, 16000], "float32", 1.);
        return feed;
    }
    if (gen == "pyannote") {
        feed["input_values"] = fillTensor([1, 1, 1000], "float32");
        return feed;
    }
    if (gen == "whisper-turbo-encoder") {
        feed["input_features"] = fillTensor([1, 128, 3000], "float32");
        return feed;
    }
    if (gen == "whisper-turbo-decoder") {
        const dtype = "float32";
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 99n);
        feed["encoder_hidden_states"] = fillTensor([1, 1500, 1280], dtype);
        const decoder_shape = [1, 20, seqlen, 64];
        const encoder_shape = [1, 20, 1500, 64];
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                if (v.includes("decoder")) {
                    feed[v] = fillTensor(decoder_shape, dtype, 1);
                } else if (v.includes("encoder")) {
                    feed[v] = fillTensor(encoder_shape, dtype, 1);
                }
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        return feed;
    }
    if (gen == "silero-vad") {
        feed["input"] = fillTensor([1, 512], "float32", 1.);
        feed["sr"] = fillTensor([1], "int64", 16000n);
        feed["state"] = fillTensor([2, 1, 128], "float32", 1.);
        return feed;
    }
    if (gen == "moonshine-encoder") {
        feed["input_values"] = fillTensor([1, 3000], "float32");
        return feed;
    }
    if (gen == "moonshine-decoder") {
        const dtype = "float32";
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 99n);
        feed["encoder_hidden_states"] = fillTensor([1, 1500, 288], dtype);
        const decoder_shape = [1, 8, seqlen, 36];
        const encoder_shape = [1, 8, 1500, 36];
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                if (v.includes("decoder")) {
                    feed[v] = fillTensor(decoder_shape, dtype, 1);
                } else if (v.includes("encoder")) {
                    feed[v] = fillTensor(encoder_shape, dtype, 1);
                }
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        return feed;
    }
    if (gen == "whisper-encoder") {
        feed["input_features"] = fillTensor([1, 80, 3000], "float32");
        return feed;
    }
    if (gen == "whisper-decoder" || gen == "hf-whisper-decoder-fp16" || gen == "hf-whisper-decoder") {
        const dtype = (gen == "hf-whisper-decoder-fp16") ? "float16" : "float32";
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 99n);
        feed["encoder_hidden_states"] = fillTensor([1, 1500, 384], dtype);
        const decoder_shape = [1, 6, seqlen, 64];
        const encoder_shape = [1, 6, 1500, 64];
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                if (v.includes("decoder")) {
                    feed[v] = fillTensor(decoder_shape, dtype, 1);
                } else if (v.includes("encoder")) {
                    feed[v] = fillTensor(encoder_shape, dtype, 1);
                }
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        return feed;
    }
    if (gen == "distil-whisper-decoder") {
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 1n);
        feed["encoder_hidden_states"] = fillTensor([1, 1500, 768], "float32");
        const decoder_shape = [1, 12, seqlen, 64];
        const encoder_shape = [1, 12, 1500, 64];
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                if (v.includes("decoder")) {
                    feed[v] = fillTensor(decoder_shape, "float32", 1);
                } else if (v.includes("encoder")) {
                    feed[v] = fillTensor(encoder_shape, "float32", 1);
                }
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        return feed;
    }
    if (gen == "img512") {
        feed[name] = rampTensor([1, 3, 512, 512], "float32");
        return feed;
    }
    if (gen == "img320") {
        feed[name] = rampTensor([1, 3, 320, 320], "float32");
        return feed;
    }
    if (gen == "img1020x678") {
        feed[name] = rampTensor([1, 3, 1020, 678], "float32");
        return feed;
    }
    if (gen == "img1024") {
        feed[name] = rampTensor([1, 3, 1024, 1024], "float32");
        return feed;
    }
    if (gen == "img224-fp16") {
        feed[name] = fillTensor([1, 3, 224, 224], "float16");
        return feed;
    }
    if (gen == "img224") {
        feed[name] = rampTensor([1, 3, 224, 224], "float32");
        return feed;
    }
    if (gen == "2ximg224") {
        feed[name] = rampTensor([2, 3, 224, 224], "float32", 0.5);
        return feed;
    }
    if (gen == "img768") {
        feed[name] = rampTensor([1, 3, 768, 768], "float32");
        return feed;
    }
    if (gen == "foo") {
        // feed['pre_x'] = fillTensor([1, 3, 2160, 3840], "float32", 0.5);
        feed['pre_x'] = createImage(2160, 3840);
        feed['orig_target_sizes'] = new ort.Tensor("int64", new BigInt64Array([BigInt(640), BigInt(640)]), [1, 2]);
        return feed;
    }
    if (gen == "img4x224x224") {
        feed[name] = fillTensor([1, 4, 224, 224], "float32", 0.5);
        return feed;
    }
    if (gen == "img640x480") {
        feed[name] = randomTensor([1, 3, 640, 480], "float32");
        return feed;
    }
    if (gen == "img256x256") {
        feed[name] = fillTensor([1, 3, 256, 256], "float32", 0.5);
        return feed;
    }
    if (gen == "img416x416") {
        feed[name] = fillTensor([1, 3, 416, 416], "float32", 0.5);
        return feed;
    }
    if (gen == "img-plants") {
        // broken: fromImage() doesn't accept RGB
        img = document.getElementById("dummy-img");
        img.src = "plants.png";
        const t = await new Promise((resolve, reject) => {
            img.onload = () => resolve(ort.Tensor.fromImage(img, { resizedWidth: 224, resizedHeight: 224 }));
        });
        feed[name] = t;
        return feed;
    }
    if (gen == "img640" || gen == "img640x640") {
        feed[name] = randomTensor([1, 3, 640, 640], "float32");
        return feed;
    }
    if (gen == "img800") {
        feed[name] = randomTensor([1, 3, 800, 800], "float32");
        return feed;
    }
    if (gen == "img512x512") {
        feed[name] = randomTensor([1, 3, 512, 512], "float32");
        return feed;
    }
    if (gen == "img224nhwc") {
        feed[name] = rampTensor([1, 224, 224], "float32");
        return feed;
    }
    if (gen == "bert" || gen == "bert64") {
        const dtype = (gen == "bert") ? "int32" : "int64";
        const val = (gen == "bert") ? 99 : 99n;
        const one = (gen == "bert") ? 1 : 1n;

        for (var k in inputNames) {
            const v = inputNames[k];
            if (v === "input_ids") {
                feed[v] = fillTensor([1, seqlen], dtype, val);
            }
            if (v === "input_mask" || v === "attention_mask") {
                feed[v] = fillTensor([1, seqlen], dtype, one);
            }
            if (v === "token_type_ids" || v == "segment_ids") {
                feed[v] = fillTensor([1, seqlen], dtype, one);
            }
        }
        return feed;
    }
    if (gen == "llm-decoder") {
        // const ids = [40n, 2883n, 6155n, 351n, 616n, 13779n, 3290n, 11n];
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values")) {
                feed[v] = fillTensor([1, 12, 0, 64], "float32", .5);
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [false], [1]);
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 40n);
        feed["attention_mask"] = fillTensor([1, seqlen], "int64", 1n);
        return feed;
    }
    if (gen == "starcoder") {
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values")) {
                feed[v] = fillTensor([1, seqlen, 128], "float32", 1.);
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 99n);
        feed["attention_mask"] = fillTensor([1, seqlen], "int64", 1n);
        return feed;
    }
    if (gen == "bart-large" || gen == "bart-large-12") {
        const kvdim = (gen == "bart-large") ? 16 : 12;
        const hiddendim = (gen == "bart-large") ? 1024 : 768;
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values")) {
                feed[v] = fillTensor([1, kvdim, seqlen, 64], "float32", 1.);
            }
            if (v.startsWith("encoder_attention_mask")) {
                feed["encoder_attention_mask"] = fillTensor([1, enc_seqlen], "int64", 1n);
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 99n);
        feed["encoder_hidden_states"] = fillTensor([1, enc_seqlen, hiddendim], "float32", 1);
        return feed;
    }
    if (gen == "bart-cnn") {
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values")) {
                feed[v] = fillTensor([1, 16, seqlen, 64], "float32", 1.);
            }
            if (v == "encoder_attention_mask") {
                feed["encoder_attention_mask"] = fillTensor([1, enc_seqlen], "int64", 1n);
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        feed["input_ids"] = fillTensor([1, seqlen], "int64", 99n);
        feed["encoder_hidden_states"] = fillTensor([1, seqlen, 1024], "float32", 1);
        return feed;
    }

    if (gen == "sam-decoder") {
        feed["image_embeddings"] = fillTensor([1, 256, 64, 64], "float32", 0.5);
        feed["image_positional_embeddings"] = fillTensor([1, 256, 64, 64], "float32", 0.5)
        feed["input_points"] = new ort.Tensor(new Float32Array([327.1111, 426.875, 241.77777, 341.5, 398.22223, 498.02084]), [1, 1, 3, 2]);
        feed["input_labels"] = new ort.Tensor(new BigInt64Array([0n, 2n, 3n]), [1, 1, 3]);
        return feed;
    }
    if (gen == "slimsam-decoder") {
        feed["image_embeddings"] = rampTensor([1, 256, 64, 64], "float32");
        // feed["image_embeddings"] = fillTensor([1, 256, 64, 64], "float32", 0.5);
        feed["image_positional_embeddings"] = fillTensor([1, 256, 64, 64], "float32", 0.5);
        feed["input_labels"] = new ort.Tensor(new BigInt64Array([1n]), [1, 1, 1]);
        feed["input_points"] = new ort.Tensor(new Float32Array([400., 500.]), [1, 1, 1, 2]);;
        return feed;
    }

    if (gen == "sam-encoder" || gen == "slimsam-encoder") {
        feed["pixel_values"] = fillTensor([1, 3, 1024, 1024], "float32", 1.);
        return feed;
    }
    if (gen == "sam-mobile-encoder") {
        feed["input_image"] = fillTensor([1, 3, 1024, 1024], "float32", 1.);
        return feed;
    }
    if (gen == "t5-encoder") {
        feed['input_ids'] = fillTensor([1, seqlen], "int64", 99n);
        feed['attention_mask'] = fillTensor([1, seqlen], "int64", 1n);
        return feed;
    }
    if (["t5-decoder", "flan-t5-decoder", "t5-decoder-i32"].includes(gen)) {
        let type = "int64";
        if (gen.includes("-i32")) {
            gen = gen.replace("-i32", "")
            type = "int32"
        }
        feed['input_ids'] = fillTensor([1, seqlen], type, 99);
        feed['encoder_hidden_states'] = fillTensor([1, enc_seqlen, 512], "float32", 1);
        const encoder_shape = (gen == "t5-decoder") ? [1, 8, enc_seqlen, 64] : [1, 6, enc_seqlen, 64];
        const decoder_shape = (gen == "t5-decoder") ? [1, 8, 0, 64] : [1, 6, 0, 64];
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                if (v.includes("decoder")) {
                    feed[v] = fillTensor(decoder_shape, "float32", 1);
                } else if (v.includes("encoder")) {
                    feed[v] = fillTensor(encoder_shape, "float32", 1);
                }
            }
            if (v == "encoder_attention_mask") {
                feed['encoder_attention_mask'] = fillTensor([1, enc_seqlen], type, 1);
            }
        }
        feed['use_cache_branch'] = new ort.Tensor("bool", [false], [1]);
        return feed;
    }
    if (gen == "florence-embed") {
        const tokens = [0n, 35438n, 162n, 59n, 5n, 2274n, 2n];
        const dtype = "float32";
        feed['input_ids'] = new ort.Tensor(new BigInt64Array(tokens), [1, tokens.length]);
        return feed;
    }
    if (gen == "florence-decoder") {
        feed["inputs_embeds"] = fillTensor([1, 128, 768], "float32", 0.5); 
        feed["encoder_hidden_states"] = fillTensor([1, 128, 768], "float32", 0.5);
        feed["encoder_attention_mask"] = fillTensor([1, 128], "int64", 1n);
        feed['use_cache_branch'] = new ort.Tensor("bool", [0], [1]);
        const decoder_shape = [1, 12, 0, 64];
        const dtype = "float32";
        for (var k in inputNames) {
            const v = inputNames[k];
            if (v.startsWith("past_key_values.")) {
                feed[v] = fillTensor(decoder_shape, dtype, 1);
            }
        }
        return feed
    }
    if (gen == "florence-encoder") {
        feed["inputs_embeds"] = fillTensor([1, 128, 768], "float32", 0.5); 
        feed["attention_mask"] = fillTensor([1, 128], "int64", 1n);
        return feed
    }

    if (gen == "tiny-random-music-encoder") {
        const tokens = [0n, 64n, 64n, 66n, 23n, 0n, 21n, 22n, 22n, 84n, 84n, 80n, 39n, 39n, 39n, 1n, 45n, 45n, 96n, 95n, 95n, 27n, 36n, 36n];
        feed['audio_codes'] = new ort.Tensor(new BigInt64Array(tokens), [1, 1, 4, 6]);
        return feed;
    }
    if (gen == "nbitmatmul") {
        feed['X'] = webgpu_tensor_from_tensor(fillTensor([96 * 160, 3072], "float32", 1));
        // feed['X'] = fillTensor([96 * 160, 3072], "float32", 1);
        return feed;
    }

    throw new Error(`unknown gendata ${gen}`);
}


export default make_inputs;
