# app_colab.py
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc
import os

# --- Основные константы ---
MODEL_ID = "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers"
LORA_DIR = "loras"
MAX_SEED = np.iinfo(np.int32).max
FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 80
MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

# --- Загрузка модели с оптимизацией памяти ---
print("Загрузка основной модели с 8-битной оптимизацией...")
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder='transformer', torch_dtype=torch.bfloat16),
    transformer_2=WanTransformer3DModel.from_pretrained(MODEL_ID, subfolder='transformer_2', torch_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    variant="bf16",
    load_in_8bit=True  # <-- Ключевая оптимизация для экономии памяти
)
pipe.enable_model_cpu_offload() # <-- Дополнительная экономия памяти
print("✅ Модель успешно загружена.")


# --- Функции обработки (без изменений) ---
def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height: return image.resize((640, 640), Image.LANCZOS)
    aspect_ratio = width / height
    MAX_ASPECT_RATIO, MIN_ASPECT_RATIO = 832 / 480, 480 / 832
    if aspect_ratio > MAX_ASPECT_RATIO:
        target_w, target_h = 832, 480
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        target_w, target_h = 480, 832
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height: target_w, target_h = 832, int(round(832 / aspect_ratio))
        else: target_h, target_w = 832, int(round(832 * aspect_ratio))
    final_w = round(target_w / 16) * 16
    final_h = round(target_h / 16) * 16
    return image_to_resize.resize((max(480, min(832, final_w)), max(480, min(832, final_h))), Image.LANCZOS)

def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(int(round(duration_seconds * FIXED_FPS)), MIN_FRAMES_MODEL, MAX_FRAMES_MODEL))

# --- Основная функция генерации ---
def generate_video(
    input_image, prompt, lora_choice, steps=6, negative_prompt="", duration_seconds=3.5,
    guidance_scale=1.0, guidance_scale_2=1.0, seed=42, randomize_seed=True,
    progress=gr.Progress(track_tqdm=True),
):
    if input_image is None: raise gr.Error("Пожалуйста, загрузите изображение.")

    # Выгружаем старые LoRA перед загрузкой новых
    try: pipe.unload_lora_weights()
    except: pass
    
    # Загружаем выбранный LoRA
    if lora_choice != "None":
        print(f"Применяем LoRA: {lora_choice}")
        lora_path = os.path.join(LORA_DIR, lora_choice)
        if os.path.exists(lora_path):
            pipe.load_lora_weights(LORA_DIR, weight_name=lora_choice, adapter_name="chosen_lora")
            pipe.set_adapters(["chosen_lora"], adapter_weights=[1.0])
            pipe.fuse_lora()
        else:
            print(f"⚠️ Внимание: Файл LoRA {lora_choice} не найден!")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image, prompt=prompt, negative_prompt=negative_prompt,
        height=resized_image.height, width=resized_image.width, num_frames=num_frames,
        guidance_scale=float(guidance_scale), guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

    gc.collect()
    torch.cuda.empty_cache()

    return video_path, current_seed

# --- Создание интерфейса Gradio ---
lora_files = ["None"] + [f for f in os.listdir(LORA_DIR) if f.endswith('.safetensors')] if os.path.exists(LORA_DIR) else ["None"]

with gr.Blocks() as demo:
    gr.Markdown("# Wan 2.2 I2V (14B) - Оптимизировано для Google Colab")
    with gr.Row():
        with gr.Column():
            input_image_component = gr.Image(type="pil", label="Входное изображение")
            prompt_input = gr.Textbox(label="Промпт", value="cinematic motion, smooth animation")
            lora_dropdown = gr.Dropdown(lora_files, label="Выберите LoRA (опционально)", value="None")
            duration_seconds_input = gr.Slider(minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5, label="Длительность (сек)")
            with gr.Accordion("Расширенные настройки", open=False):
                negative_prompt_input = gr.Textbox(label="Негативный промпт", value="static, still image, ugly, disfigured")
                seed_input = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
                randomize_seed_checkbox = gr.Checkbox(label="Случайный seed", value=True)
                steps_slider = gr.Slider(minimum=1, maximum=30, step=1, value=6, label="Шаги")
                guidance_scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 1")
                guidance_scale_2_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=1, label="Guidance Scale 2")
            generate_button = gr.Button("Сгенерировать видео", variant="primary")
        with gr.Column():
            video_output = gr.Video(label="Результат", autoplay=True, interactive=False)
    
    ui_inputs = [
        input_image_component, prompt_input, lora_dropdown, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input, seed_input, randomize_seed_checkbox
    ]
    generate_button.click(fn=generate_video, inputs=ui_inputs, outputs=[video_output, seed_input])

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
