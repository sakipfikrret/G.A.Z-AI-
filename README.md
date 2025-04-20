# Gelecek Ahenk Zekası (GAZ) - Multimodal AI Framework

Türkçe destekli çoklu ortam üretimi yapabilen yapay zeka sistemi


## Özellikler
- 🇹🇷 Türkçe Doğal Dil İşleme (CerebrumTech'in özel eğitilmiş LLM modeli)
- 🖼️ Metinden Görsel Üretme (Segmind SSD-1B modeli)
- 🎥 Metinden Video Üretme (DAMO-VILab'ın text-to-video modeli)
- 🎶 Metinden Müzik Üretme (Facebook MusicGen - geliştirme aşamasında)
- 🚀 Dinamik Model Yükleme ve Bellek Optimizasyonu
- 🔋 4-bit Quantization Desteği

## Teknik Gereksinimler
- Python 3.9+
- NVIDIA GPU (RTX 20xx ve üzeri önerilir)
- CUDA 11.8
- 16GB+ RAM (24GB önerilir)
- 8GB+ VRAM

## Gerekli paketler
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt


## Sınırlamalar ve Bilinen Sorunlar

Video üretimi için en az 12GB VRAM gerekmektedir

Türkçe model yanıt kalitesi İngilizce'ye göre daha düşük olabilir

Müzik üretim modülü şu anda aktif değil

Yüksek çözünürlüklü görseller için ek optimizasyon gerekli


## Referanslar ve Altyapı

Hugging Face Transformers

CerebrumTech Türkçe LLM

Segmind SSD-1B Model

DAMO-VILab Video Model

BitsAndBytes Quantization
