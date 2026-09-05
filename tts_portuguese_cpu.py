"""
Portuguese Text-to-Speech (TTS) using Pocket-TTS on CPU.
Language: Portuguese (pt)
Default Voice: rafael
Supports MP3 and WAV output formats.
"""

import argparse
import os
import time
import torch
import soundfile as sf
import scipy.io.wavfile
from pocket_tts import TTSModel

def save_audio_file(audio_tensor: torch.Tensor, sample_rate: int, output_path: str):
    """Save audio tensor to MP3 or WAV based on file extension."""
    audio_np = audio_tensor.detach().cpu().numpy()
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp3":
        # soundfile expects (samples, channels) or (samples,)
        data = audio_np.T if audio_np.ndim > 1 else audio_np
        sf.write(output_path, data, sample_rate, format="MP3")
    else:
        scipy.io.wavfile.write(output_path, sample_rate, audio_np)

def generate_tts_cpu(
    text: str,
    voice: str = "rafael",
    output_path: str = "portuguese_cpu.mp3",
    use_24l: bool = False,
    temperature: float | None = None
):
    print("=" * 60)
    print(" Pocket-TTS: Portuguese Speech Synthesis (CPU)")
    print("=" * 60)
    
    language_key = "portuguese_24l" if use_24l else "portuguese"
    print(f"[*] Language Model : {language_key}")
    print(f"[*] Voice           : {voice}")
    print(f"[*] Execution Device: CPU ({torch.get_num_threads()} threads)")
    print(f"[*] Output Format   : {os.path.splitext(output_path)[1].upper().replace('.', '')}")
    print(f"[*] Input Text      : \"{text}\"")
    print("-" * 60)

    # 1. Load the Model on CPU
    print("[1/3] Loading Pocket-TTS Portuguese model...")
    t0 = time.time()
    model = TTSModel.load_model(
        language=language_key,
        temp=temperature
    )
    load_time = time.time() - t0
    print(f"      Model loaded in {load_time:.2f} seconds.")

    # 2. Extract / Load Voice State
    print(f"[2/3] Conditioning on voice '{voice}'...")
    t0 = time.time()
    voice_state = model.get_state_for_audio_prompt(voice)
    voice_time = time.time() - t0
    print(f"      Voice state initialized in {voice_time:.2f} seconds.")

    # 3. Generate Audio
    print("[3/3] Synthesizing speech...")
    t0 = time.time()
    audio = model.generate_audio(
        model_state=voice_state,
        text_to_generate=text,
        copy_state=True
    )
    gen_time = time.time() - t0

    # Save to disk (MP3 or WAV)
    save_audio_file(audio, model.sample_rate, output_path)
    
    audio_duration = audio.shape[-1] / model.sample_rate
    rtf = audio_duration / gen_time if gen_time > 0 else 0
    file_size_kb = os.path.getsize(output_path) / 1024

    print("-" * 60)
    print(f"[+] Audio Duration  : {audio_duration:.2f} seconds")
    print(f"[+] Generation Time : {gen_time:.2f} seconds")
    print(f"[+] Speed Factor    : {rtf:.2f}x Real-Time (Faster than real-time: {'Yes' if rtf > 1 else 'No'})")
    print(f"[+] Saved to File   : {os.path.abspath(output_path)} ({file_size_kb:.1f} KB)")
    print("=" * 60)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Portuguese TTS on CPU using Pocket-TTS")
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="Olá! Bem-vindo ao Pocket TTS, um modelo leve e rápido executado na CPU.",
        help="Portuguese text to synthesize."
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="rafael",
        help="Voice name ('rafael') or path to a .wav audio prompt for voice cloning."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="portuguese_cpu.mp3",
        help="Path to save the generated audio file (.mp3 or .wav)."
    )
    parser.add_argument(
        "--24l",
        action="store_true",
        dest="use_24l",
        help="Use higher quality 24-layer Portuguese model (portuguese_24l)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (e.g., 0.7)."
    )

    args = parser.parse_args()
    generate_tts_cpu(
        text=args.text,
        voice=args.voice,
        output_path=args.output,
        use_24l=args.use_24l,
        temperature=args.temperature
    )
