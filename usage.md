# Pocket-TTS: Comprehensive Usage & Deployment Guide
# دليل التثبيت، التشغيل، والاستخدام الشامل (محلياً وعبر GitHub)

دليل شامل ومرجع عملي خطوة بخطوة لتثبيت وتشغيل مشروع **Pocket-TTS** المتخصص في توليد الصوت وتحويل النص إلى كلام (TTS) باللغة البرتغالية واللغات العالمية، مع دعم كامل للتشغيل على المعالج (**CPU**) وبطاقات الرسوميات (**NVIDIA GPU / CUDA**)، واستنساخ الأصوات (Voice Cloning)، وواجهة الويب التفاعلية (Web UI).

---

## 📑 فهرس المحتويات (Table of Contents)
1. [متطلبات النظام والتشغيل (Prerequisites)](#1-متطلبات-النظام-والتشغيل-prerequisites)
2. [خطوات التثبيت خطوة بخطوة لأول مرة (Installation & Setup)](#2-خطوات-التثبيت-خطوة-بخطوة-لأول-مرة-installation--setup)
3. [التحقق السريع من صحة التثبيت (Verification)](#3-التحقق-السريع-من-صحة-التثبيت-verification)
4. [أوامر التوليد عبر سطر الأوامر (CLI Usage)](#4-أوامر-التوليد-عبر-سطر-الأوامر-cli-usage)
5. [تشغيل واجهة الويب التفاعلية (Multi-Language Web UI)](#5-تشغيل-واجهة-الويب-التفاعلية-multi-language-web-ui)
6. [استنساخ الأصوات الخارجية ومصادقة Hugging Face](#6-استنساخ-الأصوات-الخارجية-ومصادقة-hugging-face)
7. [استخدام المكتبة برمجياً داخل كود Python (API)](#7-استخدام-المكتبة-برمجياً-داخل-كود-python-api)
8. [هيكل ملفات المشروع (Repository Structure)](#8-هيكل-ملفات-المشروع-repository-structure)
9. [دليل استكشاف الأخطاء الشائعة وحلها (Troubleshooting)](#9-دليل-استكشاف-الأخطاء-الشائعة-وحلها-troubleshooting)

---

## 1. متطلبات النظام والتشغيل (Prerequisites)

قبل البدء، تأكد من توفر المتطلبات التالية على جهازك:

| المتطلب | المواصفات المدعومة / الموصى بها | ملاحظات |
| :--- | :--- | :--- |
| **نظام التشغيل** | Windows 10/11, Ubuntu/Debian Linux, macOS | يعمل على جميع المنصات |
| **إصدار بايثون** | **Python 3.10** أو **3.11** أو **3.12** *(الموصى به: 3.11)* | بايثون 3.13 غير مدعوم حالياً في بعض حزم الذكاء الاصطناعي |
| **مدير الحزم / البيئات** | **Miniconda / Anaconda** أو بيئة `venv` الافتراضية | يفضل Conda لسهولة إدارة حزم CUDA |
| **أداة Git** | مثبتة ومسجلة في الـ PATH | لتنزيل واستنساخ المشروع |
| **كارت الشاشة (اختياري)** | NVIDIA GPU (RTX / GTX) مع تعريفات CUDA 11.8 أو 12.x | اختياري؛ النموذج مصمم ليعمل بسرعة فائقة على الـ **CPU** فقط |
| **أداة FFmpeg** | مثبتة في النظام | هامة جداً لمعالجة وتصدير الصوت بصيغة **MP3** |

> 💡 **طريقة تثبيت FFmpeg بسهولة إذا لم تكن مثبتة:**
> - **Windows**: افتح PowerShell كمسؤول ونفّذ: `winget install Gyan.FFmpeg`
> - **Ubuntu / Debian**: `sudo apt update && sudo apt install ffmpeg -y`
> - **macOS**: `brew install ffmpeg`

---

## 2. خطوات التثبيت خطوة بخطوة لأول مرة (Installation & Setup)

اتبع هذه الخطوات بالترتيب لتجهيز بيئة العمل وتشغيل المشروع:

### 🔹 الخطوة 1: استنساخ المستودع من GitHub
افتح الطرفية (PowerShell أو Terminal) ونفّذ الأوامر التالية:

```bash
# استنساخ المشروع مع المستودعات الفرعية
git clone --recurse-submodules https://github.com/<your-username>/<repo-name>.git

# الدخول إلى مجلد المشروع
cd <repo-name>
```

> ⚠️ **ملاحظة:** إذا قمت باستنساخ المشروع بدون خيار `--recurse-submodules`، نفّذ الأمر التالي لتنزيل كود `pocket-tts-repo`:
> ```bash
> git submodule update --init --recursive
> ```

---

### 🔹 الخطوة 2: إنشاء وتفعيل بيئة العمل (Virtual Environment)

اختر إحدى الطريقتين التاليتين:

#### الخيار (أ) - باستخدام Conda (الموصى به):
```bash
# إنشاء بيئة جديدة باسم pocket-tts
conda create -n pocket-tts python=3.11 -y

# تفعيل البيئة
conda activate pocket-tts
```

#### الخيار (ب) - باستخدام Python venv الافتراضي:
```bash
# إنشاء البيئة
python -m venv venv

# تفعيل البيئة على Windows (PowerShell):
venv\Scripts\Activate.ps1

# أو على Windows (CMD):
venv\Scripts\activate.bat

# أو على Linux / macOS:
source venv/bin/activate
```

---

### 🔹 الخطوة 3: تثبيت مكتبة PyTorch المناسبة لجهازك

#### إذا كان لديك كارت شاشة NVIDIA (GPU مع تسريع CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
*(إذا كانت بطاقتك تدعم CUDA 11.8 استخدم: `https://download.pytorch.org/whl/cu118`)*

#### إذا كنت تريد التشغيل على المعالج المركزي فقط (CPU):
```bash
pip install torch torchvision torchaudio
```

---

### 🔹 الخطوة 4: تثبيت اعتماديات ومكتبة Pocket-TTS
نفّذ الأمر التالي لتثبيت كافة الحزم والمكتبات دفعة واحدة:

```bash
pip install -r requirements.txt
```

*(أو يدوياً في سطرين)*:
```bash
pip install -e ./pocket-tts-repo
pip install soundfile scipy requests huggingface_hub fastapi uvicorn python-multipart
```

---

## 3. التحقق السريع من صحة التثبيت (Verification)

للتأكد من أن جميع الحزم تم تثبيتها بنجاح وأن النموذج قادر على التوليد على جهازك، شغّل سكربت الاختبار المرفق:

```bash
python test_pt_tts.py
```

سيقوم السكربت بفحص:
1. إصدار بايثون و PyTorch وتوفر CUDA.
2. تحميل النموذج وتوليد عينة صوتية على **CPU**.
3. تحميل النموذج وتوليد عينة صوتية على **GPU** (في حال توفر كارت شاشة متوافق).
4. حفظ عينات اختبارية للتأكد من سلامة الصوت.

---

## 4. أوامر التوليد عبر سطر الأوامر (CLI Usage)

بعد تفعيل البيئة، يمكنك تشغيل أوامر التوليد مباشرة من داخل مجلد المشروع دون الحاجة لكتابة أي مسارات مطلقة:

### 🔹 4.1 التوليد على المعالج المركزي (CPU)
أداة مخصصة وخفيفة جداً تعمل على أي حاسوب دون الحاجة لكارت شاشة:

```bash
python tts_portuguese_cpu.py --text "Olá! Este é um teste de síntese de voz em português executado no CPU." --output "output_cpu.mp3"
```

**الخيارات المتاحة:**
* `--text` أو `-t`: النص البرتغالي المراد نطقه.
* `--output` أو `-o`: مسار واسم ملف الصوت الناتج (`.mp3` أو `.wav`).
* `--voice` أو `-v`: اسم الصوت (الافتراضي: `rafael`، أو مسار ملف صوت خارجي).
* `--24l`: تفعيل نموذج الـ 24 طبقة عالي الدقة (افتراضياً يعمل النموذج القياسي فائق السرعة).
* `--temperature`: درجة حرارة التوليد وضبط التنوع (مثلاً `0.7`).

---

### 🔹 4.2 التوليد مع تسريع كارت الشاشة (GPU / CUDA)
أقصى سرعة ممكنة للتوليد اللحظي باستغلال أنوية Tensor Cores:

```bash
python tts_portuguese_gpu.py --text "Olá! Este é um teste com aceleração total em GPU NVIDIA usando Pocket TTS." --output "output_gpu.mp3"
```

---

### 🔹 4.3 السكربت الموحد الشامل (`tts_portuguese_unified.py`)

أداة سطر أوامر ذكية تدعم التحديد التلقائي للعتاد، البث اللحظي، وضغط الذاكرة:

#### أ) التوليد التلقائي (يكتشف GPU تلقائياً وإلا يعمل على CPU):
```bash
python tts_portuguese_unified.py --device auto --text "Pocket TTS é leve, rápido e funciona perfeitamente em português." --output "resultado.mp3"
```

#### ب) التوليد بنمط البث المباشر (Streaming Mode):
```bash
python tts_portuguese_unified.py --device cuda --stream --text "Este é um teste de transmissão em tempo real de áudio em português." --output "stream_pt.mp3"
```

#### ج) التوليد مع تقليل استهلاك الذاكرة (INT8 Quantization):
```bash
python tts_portuguese_unified.py --quantize --text "Texto em português com modelo quantizado em int8." --output "quantized_pt.mp3"
```

#### د) استنساخ صوت خارجي (Voice Cloning):
قم بتمرير ملف صوتي `.wav` (مدة 3 إلى 10 ثوانٍ) لأي متحدث:
```bash
python tts_portuguese_unified.py --voice "sample_voice.wav" --text "Olá, agora estou falando com a voz clonada do arquivo de áudio fornecido." --output "voz_clonada.mp3"
```

---

## 5. تشغيل واجهة الويب التفاعلية (Multi-Language Web UI)

يوفر المشروع واجهة ويب رسومية عصرية تدعم 6 لغات عالمية، وتتيح تجربة كافة الأصوات، الاستماع المباشر، وتسجيل الصوت لاستنساخه فوراً.

### 🌐 أمر تشغيل خادم الويب:

#### للتشغيل على المعالج (CPU):
```bash
python -m pocket_tts serve --language portuguese --device cpu
```

#### للتشغيل مع تسريع كارت الشاشة (GPU):
```bash
python -m pocket_tts serve --language portuguese --device cuda
```

بمجرد تشغيل السيرفر، افتح المتصفح على:
👉 **`http://localhost:8000`**

### ✨ مميزات واجهة الويب:
1. **شريط تبديل اللغات (Language Switcher)**: التبديل بنقرة زر واحدة بين:
   - 🇵🇹 البرتغالية (Portuguese)
   - 🇺🇸 الإنجليزية (English)
   - 🇪🇸 الإسبانية (Spanish)
   - 🇫🇷 الفرنسية (French)
   - 🇩🇪 الألمانية (German)
   - 🇮🇹 الإيطالية (Italian)
2. **كتالوج الأصوات الذكي**: الوصول لـ 26 صوتاً مدمجاً عالي الجودة مع تعيين الصوت الافتراضي الأنسب لكل لغة تلقائياً.
3. **أمثلة نصوص جاهزة (Quick Prompts)** مخصصة لكل لغة لتجربة فورية.
4. **تسجيل مباشر أو رفع ملفات**: إمكانية تسجيل صوتك من المايكروفون مباشرة واستنساخ نبرتك في ثوانٍ.
5. **تنزيل بصيغتي MP3 و WAV** مع دعم التشغيل الفوري (Real-Time Audio Streaming).

---

## 6. استنساخ الأصوات الخارجية ومصادقة Hugging Face

### 💡 كيف تعمل أوزان النموذج؟
* **الأصوات المدمجة (بدون تسجيل دخول)**:
  تعمل مباشرة دون الحاجة لأي توكن أو حساب، وتشمل 26 صوتاً احترافياً (مثل الصوت البرتغالي الافتراضي **`rafael`**، والأصوات `alba`, `giovanni`, `lola`, `juergen`, `estelle`... إلخ).
* **استنساخ أي صوت خارجي مخصص (Voice Cloning)**:
  يتطلب تحميل أوزان نموذج الـ Voice Conditioning المحمي بترخيص Kyutai على Hugging Face.

### 🔑 خطوات تفعيل استنساخ الأصوات (لمرة واحدة فقط):
1. سجّل دخولك على Hugging Face وافتح رابط المستودع: **[https://huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts)**.
2. اضغط على زر **Agree and access repository** للموافقة على الشروط.
3. أنشئ توكن قراءة (Read Token) من: **[Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens)**.
4. سجّل دخولك في الطرفية داخل بيئة العمل:
   ```bash
   huggingface-cli login
   ```
   *(ألصق التوكن واضغط Enter)*.

   *أو عبر تعيين متغير البيئة مباشرة في جلستك:*
   - **Windows PowerShell**:
     ```powershell
     $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     ```
   - **Linux / macOS / Bash**:
     ```bash
     export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
     ```

---

## 7. استخدام المكتبة برمجياً داخل كود Python (API)

يمكنك دمج النموذج بسهولة داخل تطبيقاتك الخاصة:

### 🔸 مثال 1: توليد بسيط وحفظ بصيغة WAV على الـ CPU
```python
import scipy.io.wavfile
from pocket_tts import TTSModel

# 1. تحميل النموذج للغة البرتغالية
model = TTSModel.load_model(language="portuguese")

# 2. تحديد الصوت الافتراضي rafael
voice_state = model.get_state_for_audio_prompt("rafael")

# 3. توليد الصوت
text = "Olá! Este é um teste de síntese de voz em português com Pocket TTS."
audio = model.generate_audio(voice_state, text)

# 4. حفظ ملف الصوت
scipy.io.wavfile.write("output.wav", model.sample_rate, audio.numpy())
print("Saved to output.wav successfully!")
```

---

### 🔸 مثال 2: توليد مسرّع على كارت الشاشة (CUDA) وحفظ MP3
```python
import soundfile as sf
from pocket_tts import TTSModel

model = TTSModel.load_model(language="portuguese")
model.to("cuda")

voice_state = model.get_state_for_audio_prompt("rafael")
text = "Síntese de voz ultra rápida acelerada por GPU NVIDIA."
audio = model.generate_audio(voice_state, text)

# نقل التنسور للمعالج وحفظه كـ MP3
audio_np = audio.detach().cpu().numpy()
sf.write("output.mp3", audio_np, model.sample_rate, format="MP3")
print("Saved to output.mp3")
```

---

### 🔸 مثال 3: بث الصوت لحظياً (Real-Time Audio Streaming)
```python
from pocket_tts import TTSModel

model = TTSModel.load_model(language="portuguese")
voice_state = model.get_state_for_audio_prompt("rafael")

# استقبال أجزاء الصوت فور توليدها
for chunk in model.generate_audio_stream(voice_state, "Este é um texto longo que será transmitido em tempo real."):
    chunk_np = chunk.detach().cpu().numpy()
    # يمكن إرسال chunk_np فوراً عبر WebSocket أو تشغيله على بطاقة الصوت
    print(f"Received chunk with {chunk_np.shape[0]} samples")
```

---

## 8. هيكل ملفات المشروع (Repository Structure)

| الملف / المجلد | الوصف |
| :--- | :--- |
| `tts_portuguese_cpu.py` | سكربت سطر الأوامر المخصص للتوليد فائق السرعة على المعالج (CPU). |
| `tts_portuguese_gpu.py` | سكربت التوليد المخصص للعمل على كروت شاشة NVIDIA (CUDA). |
| `tts_portuguese_unified.py` | الأداة الشاملة (اكتشاف تلقائي للعتاد، بث مباشر Streaming، واستنساخ صوت). |
| `test_pt_tts.py` | سكربت التحقق السريع من صحة التثبيت وفحص دعم الـ CPU و GPU. |
| `requirements.txt` | قائمة الاعتماديات والمكتبات اللازمة لتثبيت المشروع بأمر واحد. |
| `.gitignore` | ملف استبعاد الملفات المؤقتة وعينات الصوت الكبيرة من مستودع Git. |
| `pocket-tts-repo/` | المستودع المصدري لمكتبة Pocket-TTS ونظام خادم الويب (FastAPI). |
| `train_portuguese_guide.md` | دليل تدريب وتخصيص نماذج برتغالية جديدة من الصفر. |
| `usage.md` | هذا الدليل المرجعي الشامل للاستخدام والتشغيل. |

---

## 9. دليل استكشاف الأخطاء الشائعة وحلها (Troubleshooting)

### ❓ خطأ: `ModuleNotFoundError: No module named 'pocket_tts'`
* **السبب**: لم يتم تثبيت حزمة `pocket-tts-repo` داخل بيئة العمل الحالية.
* **الحل**: تأكد من تفعيل البيئة (`conda activate pocket-tts` أو تفعيل الـ `venv`) ثم نفّذ:
  ```bash
  pip install -e ./pocket-tts-repo
  ```

---

### ❓ خطأ: `RuntimeError: CUDA out of memory`
* **السبب**: ذاكرة كارت الشاشة (VRAM) غير كافية للنص الطويل دفعة واحدة.
* **الحل**:
  1. استخدم خيار تقليل الذاكرة: `--quantize` في السكربت الموحد.
  2. أو قسّم النصوص الطويلة إلى فقرات أصغر.
  3. أو شغّل على المعالج: `--device cpu` حيث لا تقتيدك ذاكرة الـ VRAM.

---

### ❓ خطأ في تصدير ملفات MP3 أو في مكتبة `soundfile`
* **السبب**: عدم توفر مكتبة FFmpeg على النظام.
* **الحل**: ثبّت FFmpeg عبر `winget install Gyan.FFmpeg` (على Windows) أو `sudo apt install ffmpeg` (على Linux) وأعد تشغيل الطرفية.

---

### ❓ خطأ: `ValueError: We could not download the weights for the model with voice cloning...`
* **السبب**: محاولة استنساخ صوت خارجي دون الموافقة على ترخيص النموذج على Hugging Face أو عدم تسجيل الـ Token.
* **الحل**: اتبع خطوات [قسم مصادقة Hugging Face](#6-استنساخ-الأصوات-الخارجية-ومصادقة-hugging-face) للموافقة على الشروط وتسجيل الدخول عبر `huggingface-cli login`.

---

### ❓ خطأ في PowerShell: `File ... cannot be loaded because running scripts is disabled`
* **السبب**: سياسة أمان PowerShell تمنع تفعيل البيئات الافتراضية افتراضياً.
* **الحل**: افتح PowerShell ونفّذ الأمر التالي:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  ثم أعد محاولة تفعيل البيئة.
