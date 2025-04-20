# GAZ_Final.py
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from diffusers import DiffusionPipeline, LCMScheduler
from typing import Union
from PIL import Image
import time
import logging

# Yapılandırma
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "quantization": {
        "text": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        ),
        "image": None,
        "video": None
    },
    "models": {
        "text": "CerebrumTech/cere-llama-3-8b-tr",
        "image": "segmind/SSD-1B",
        "video": "damo-vilab/text-to-video-ms-1.7b",
        "music": "facebook/musicgen-small"
    },
    "cache_dir": "./model_cache",
    "max_memory": "8GB"  # Sistem limiti
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GAZ")


class GelecekAhenkZekası:
    def __init__(self):
        self.loaded_models = {}
        self.current_mode = None

        # Türkçe tokenizer ön yükleme
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["models"]["text"],
            use_fast=False,
            cache_dir=CONFIG["cache_dir"]
        )

    def load_model(self, model_type: str):
        if model_type == self.current_mode:
            return

        # Önceki modeli boşalt
        if self.current_mode:
            del self.loaded_models[self.current_mode]
            torch.cuda.empty_cache()

        logger.info(f"{model_type} modeli yükleniyor...")

        if model_type == "text":
            self.loaded_models["text"] = AutoModelForCausalLM.from_pretrained(
                CONFIG["models"]["text"],
                quantization_config=CONFIG["quantization"]["text"],
                device_map="auto",
                cache_dir=CONFIG["cache_dir"]
            )

        elif model_type == "image":
            self.loaded_models["image"] = DiffusionPipeline.from_pretrained(
                CONFIG["models"]["image"],
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                safety_checker=None
            ).to(CONFIG["device"])
            self.loaded_models["image"].scheduler = LCMScheduler.from_config(
                self.loaded_models["image"].scheduler.config
            )

        elif model_type == "video":
            self.loaded_models["video"] = pipeline(
                "text-to-video",
                model=CONFIG["models"]["video"],
                device=CONFIG["device"],
                torch_dtype=torch.float16
            )

        self.current_mode = model_type

    def konus(self, prompt: str, max_tokens: int = 256) -> str:
        self.load_model("text")
        inputs = self.tokenizer(
            f"<s>[KULLANICI]{prompt}[/YAPAYZEKA]",
            return_tensors="pt"
        ).to(CONFIG["device"])

        outputs = self.loaded_models["text"].generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def resim_olustur(self, prompt: str) -> Image.Image:
        self.load_model("image")
        return self.loaded_models["image"](
            prompt,
            num_inference_steps=6,
            guidance_scale=1.3
        ).images[0]

    def video_olustur(self, prompt: str) -> str:
        self.load_model("video")
        output = self.loaded_models["video"](prompt, num_frames=24, fps=8)
        video_path = f"output_{int(time.time())}.mp4"
        output.save(video_path)
        return video_path


# Kullanım Örneği
if __name__ == "__main__":
    gaz = GelecekAhenkZekası()

    # Metin Üretimi
    print("### Konuşma Testi ###")
    cevap = gaz.konus("Türk yapay zekası için neden önemliyiz?")
    print(cevap)


    # Video Üretimi (GPU gerektirir)