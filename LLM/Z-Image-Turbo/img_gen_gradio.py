import os
import sys

# =================【关键修复】=================
# 1. 必须在 import torch 之前设置！
# 这样程序启动时，只会看到一张显卡（即物理上的 GPU 2）
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# ============================================

import torch
import gradio as gr
from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

print(f"正在加载模型: {MODEL_ID}...")
print("提示：已配置为仅使用 GPU 2 (Tesla V100)")

try:
    # 加载管道
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    # ================= 显存优化 =================
    # 2. 显式告诉 offload 使用 "cuda:0"
    # (因为上面屏蔽了其他卡，所以这里的 cuda:0 实际上就是物理 GPU 2)
    pipe.enable_model_cpu_offload(device="cuda:0")

    # 开启 VAE 切片 (防止解码大图爆显存)
    pipe.vae.enable_tiling()

    print("✅ 模型加载成功！显卡锁定正确。")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()


# ================= 生成函数 =================
def generate_image(prompt, width, height, guidance_scale, steps, seed):
    if seed == -1:
        seed = torch.randint(0, 2147483647, (1,)).item()

    # 这里 generator 指定 cuda 即可
    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    print(f"生成中... 尺寸: {width}x{height} | 种子: {seed}")

    try:
        image = pipe(
            prompt=prompt,
            width=int(width),
            height=int(height),
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator
        ).images[0]
        return image, f"Used Seed: {seed} | Size: {width}x{height}"

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return None, "❌ 显存不足 (OOM)！请尝试减小尺寸。"
        else:
            return None, f"出错: {e}"


# ================= 界面 =================
with gr.Blocks(title="Z-Image-Turbo") as demo:
    gr.Markdown("## 🚀 Z-Image-Turbo (GPU 2 专属)")

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="提示词",
                value="A majestic tiger sitting on a mountain peak, chinese painting style, 8k",
                lines=3
            )

            with gr.Row():
                width_slider = gr.Slider(256, 1280, value=768, step=64, label="宽度 (Width)")
                height_slider = gr.Slider(256, 1280, value=768, step=64, label="高度 (Height)")

            with gr.Row():
                steps_slider = gr.Slider(1, 20, value=8, step=1, label="步数")
                guidance_slider = gr.Slider(0, 5, value=1.0, step=0.1, label="引导系数")

            seed_input = gr.Number(value=-1, label="种子 (-1 随机)", precision=0)
            run_btn = gr.Button("生成", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="结果", type="pil")
            status_output = gr.Textbox(label="状态")

    run_btn.click(
        generate_image,
        [prompt_input, width_slider, height_slider, guidance_slider, steps_slider, seed_input],
        [output_image, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    