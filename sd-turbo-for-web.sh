
#
# script to create a stable diffusion turbo onnx model that is good for ort-web
# before running, change root to your needs and 
# pip install -U transformers diffusers optimum onnxruntime
#

root=/data/NN/sd
model=$root/onnx-sd-turbo
model_fp16=$root/onnx-sd-turbo-fp16
out=$root/sd-opt

org=stabilityai/sd-turbo

if [ ! -d $model ] ; then
   optimum-cli export onnx -m $org $model
fi
if [ ! -d $model_fp16 ] ; then
  optimum-cli export onnx --fp16 --device cuda -m $org $model_fp16
  python onnx-wrap-fp16.py --input $model_fp16/unet/model.onnx --output  $model_fp16/unet/model.onnx
  python onnx-wrap-fp16.py --input $model_fp16/vae_decoder/model.onnx --output  $model_fp16/vae_decoder/model.onnx
  python onnx-remove-const.py --input $model_fp16/text_encoder/model.onnx --output  $model_fp16/text_encoder/model.onnx
  python onnx-remove-double.py --input $model_fp16/text_encoder/model.onnx --output  $model_fp16/text_encoder/model.onnx
  python onnx-wrap-fp16.py --input $model_fp16/text_encoder/model.onnx --output  $model_fp16/text_encoder/model.onnx
fi

# [--inspect] [--disable_attention] [--disable_skip_layer_norm] [--disable_embed_layer_norm] [--disable_bias_skip_layer_norm] [--disable_bias_gelu] 
# [--disable_layer_norm] [--disable_gelu] [--enable_gelu_approximation] [--disable_shape_inference] [--enable_gemm_fast_gelu] [--use_mask_index] 
# [--use_raw_attention_mask] [--no_attention_mask] [--use_multi_head_attention] [--disable_group_norm] [--disable_skip_group_norm] [--disable_rotary_embeddings]
# [--disable_packed_kv] [--disable_packed_qkv] [--disable_bias_add] [--disable_bias_splitgelu] [--disable_nhwc_conv] [--use_group_norm_channels_first] 

# for now don't use custom ops
opt="--no_attention_mask --disable_skip_layer_norm --disable_attention --disable_nhwc_conv --disable_group_norm --disable_skip_group_norm --disable_embed_layer_norm --disable_bias_splitgelu --disable_bias_skip_layer_norm  --disable_bias_gelu --disable_packed_kv  --disable_packed_qkv"

python -m onnxruntime.transformers.models.stable_diffusion.optimize_pipeline -i $model -o $out --overwrite --float16 $opt

python onnx/onnx-wrap-fp16.py --input $out/unet/model.onnx --output  $out/unet/model.onnx
python onnx/onnx-wrap-fp16.py --input $out/vae_decoder/model.onnx --output  $out/vae_decoder/model.onnx
python onnx/onnx-wrap-fp16.py --input $out/text_encoder/model.onnx --output  $out/text_encoder/model.onnx

