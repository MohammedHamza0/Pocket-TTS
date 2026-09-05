# دليل بناء وتدريب واستخدام نموذج Pocket-TTS باللغة البرتغالية (CPU & GPU)

يوفر هذا الدليل شرحاً تفصيلياً لكيفية تشغيل، استخدام، وتدريب نماذج **Pocket-TTS** باللغة البرتغالية على كل من المعالج المركزي (CPU) وكارت الشاشة (GPU NVIDIA CUDA).

---

## 1. نظرة عامة على معمارية Pocket-TTS
- **Flow Matching Transformer (FlowLM)**: نموذج خفيف وسريع يعتمد على Flow Matching لتوليد التضمينات الصوتية (Audio Latents) من النصوص والرموز (Text Tokens).
- **Mimi Neural Audio Codec**: معمارية تشفير وفك تشفير صوتي تعمل بتردد 24kHz بمعدل ضغط عالي واستجابة لحظية.
- **Voice Conditioning**: استنساخ الصوت عبر تحويل عينة صوتية (أو ملف `.safetensors`) إلى فضاء حالة النموذج لضمان تطابق نبرة المتحدث وأسلوبه.

---

## 2. النماذج والأصوات البرتغالية المدمجة
تدعم Pocket-TTS مستويين من النماذج للغة البرتغالية:
1. `portuguese` (النموذج القياسي السريع - 6 طبقات): فائق السرعة وخفيف جداً على الـ CPU.
2. `portuguese_24l` (النموذج عالي الجودة - 24 طبقة): يقدم جودة صوتية أعلى ووضوح نبرة أدق.
3. **الصوت البرتغالي الافتراضي**: `rafael` (مدمج وجاهز للاستخدام مباشرة دون الحاجة لملفات خارجية)، بالإضافة لإمكانية تمرير أي ملف `.wav` برتغالي لاستنساخ الصوت فورياً.

---

## 3. التشغيل السريع من سطر الأوامر (CLI)

### تشغيل على الـ CPU:
```bash
python tts_portuguese_cpu.py --text "Olá! Este é um exemplo de síntese de voz em português no CPU." --output pt_cpu.wav
```

### تشغيل على الـ GPU (NVIDIA RTX CUDA):
```bash
python tts_portuguese_gpu.py --text "Olá! Este é um exemplo de síntese de voz em português acelerado por GPU." --output pt_gpu.wav
```

### تشغيل عبر الأداة الموحدة الشاملة (مع البث المباشر Streaming):
```bash
# توليد فوري على GPU
python tts_portuguese_unified.py --device cuda --text "Pocket TTS é incrível e muito rápido." --output resultado.wav

# توليد مع تفعيل الـ Streaming واستخدام النموذج الأكبر 24-layer
python tts_portuguese_unified.py --device cuda --model-size 24l --stream --text "Este é um teste de transmissão em tempo real."
```

### تشغيل واجهة الويب التفاعلية (Web UI Server):
```bash
python -m pocket_tts serve --language portuguese --device cpu
```
ثم فتح المتصفح على: `http://localhost:8000`

---

## 4. استخدام مكتبة Python البرمجية

### كود Python للتوليد على CPU:
```python
import scipy.io.wavfile
from pocket_tts import TTSModel

# 1. تحميل النموذج للغة البرتغالية على CPU
model = TTSModel.load_model(language="portuguese")

# 2. تحديد الصوت (rafael مدمج أو مسار ملف صوتي)
voice_state = model.get_state_for_audio_prompt("rafael")

# 3. توليد الصوت
text = "Olá! Síntese de voz em português com Pocket TTS."
audio = model.generate_audio(voice_state, text)

# 4. حفظ الملف الصوتي (24kHz)
scipy.io.wavfile.write("pt_cpu.wav", model.sample_rate, audio.numpy())
```

### كود Python للتوليد مع تسريع GPU (CUDA):
```python
import scipy.io.wavfile
import torch
from pocket_tts import TTSModel

# 1. تحميل النموذج ونقله لكارت الشاشة
model = TTSModel.load_model(language="portuguese")
model.to("cuda")

# 2. استخراج حالة الصوت على GPU
voice_state = model.get_state_for_audio_prompt("rafael")

# 3. توليد الصوت على CUDA
text = "Olá! Geração de áudio acelerada por GPU NVIDIA."
audio = model.generate_audio(voice_state, text)

# 4. نقل النتيجة إلى CPU وحفظها
audio_np = audio.detach().cpu().numpy()
scipy.io.wavfile.write("pt_gpu.wav", model.sample_rate, audio_np)
```

---

## 5. تدريب نموذج برتغالي مخصص من الصفر (Training Pipeline)

إذا كنت ترغب في تدريب نموذج TTS مخصص للغة البرتغالية أو لهجة معينة:

### الخطوة 1: جمع البيانات (Dataset)
- تحتاج إلى تسجيلات صوتية مقترنة بالنصوص (Speech-Text pairs) بجودة 24kHz أو 44.1kHz.
- مصادر مقترحة:
  - **Mozilla Common Voice (Portuguese)**
  - **MLS (Multilingual LibriSpeech - Portuguese)**
  - تسجيلات استوديو مخصصة (100 ساعة كحد أدنى، 1000+ ساعة لأعلى جودة).

### الخطوة 2: المحاذاة الإجبارية (Forced Alignment)
تتطلب Pocket-TTS معرفة توقيت بداية ونهاية كل كلمة بدقة عبر Wav2Vec2 CTC:
```bash
# استخدام نموذج Wav2Vec2 مخصص للغة البرتغالية
python training/scripts/align_data.py --model facebook/wav2vec2-large-xlsr-53-portuguese --audio_dir ./data/pt_audio --transcripts ./data/pt_transcripts.json
```

### الخطوة 3: تدريب الـ Tokenizer (SentencePiece)
```bash
python training/scripts/train_tokenizer.py --input ./data/pt_corpus.txt --vocab_size 4000 --model_prefix ./portuguese_tokenizer
```

### الخطوة 4: ملف الإعدادات والتدريب (Training Config)
إنشاء ملف إعدادات `training/configs/portuguese_custom.yaml`:
```yaml
flow_lm:
  transformer:
    d_model: 1024
    num_heads: 16
    num_layers: 6 # أو 24 للنموذج الأكبر
  lookup_table:
    tokenizer_path: ./portuguese_tokenizer.model
    n_bins: 4000

dataset:
  train_manifest: ./data/train_pt_manifest.jsonl
  valid_manifest: ./data/valid_pt_manifest.jsonl
  batch_size: 16
  grad_accum_steps: 4

training:
  lr: 0.0002
  max_steps: 100000
  checkpoint_interval: 5000
```

### الخطوة 5: بدء التدريب
```powershell
& "G:\miniconda\installation\envs\midade\python.exe" e:\TTS\pocket-tts-repo\training\train.py training/configs/portuguese_custom.yaml
```

### الخطوة 6: توليد الصوت من الـ Checkpoint المدرب
```powershell
& "G:\miniconda\installation\envs\midade\python.exe" -m pocket_tts generate `
    --config e:\TTS\pocket-tts-repo\pocket_tts\config\portuguese.yaml `
    --checkpoint runs/portuguese_custom/checkpoint_00050000.pt `
    --voice "rafael" `
    --text "Olá! Este modelo foi treinado do zero com dados em português."
```
