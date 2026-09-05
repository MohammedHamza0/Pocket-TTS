"""
Unified Portuguese Text-to-Speech (TTS) CLI Tool using Pocket-TTS.
Supports CPU and GPU (CUDA), streaming, voice cloning, INT8 quantization, and MP3/WAV export.
"""

import argparse
import os
import sys
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
        data = audio_np.T if audio_np.ndim > 1 else audio_np
        sf.write(output_path, data, sample_rate, format="MP3")
    else:
        scipy.io.wavfile.write(output_path, sample_rate, audio_np)

def main():
    parser = argparse.ArgumentParser(
        description="Unified Portuguese Pocket-TTS Generator (CPU / GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to execute inference on ('cpu', 'cuda', or 'auto')."
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="O Pocket TTS é um sistema de síntese de voz de última geração que funciona perfeitamente em CPU e GPU.",
        help="Text to synthesize in Portuguese."
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default="rafael",
        help="Predefined voice name ('rafael') or local path / URL to a .wav audio file or .safetensors embedding."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_portuguese.mp3",
        help="Output audio file path (.mp3 or .wav)."
    )
    parser.add_argument(
        "--model-size", "-m",
        choices=["standard", "24l"],
        default="standard",
        help="Model variant: 'standard' (6-layer, fast) or '24l' (24-layer, higher quality)."
    )
    parser.add_argument(
        "--quantize", "-q",
        action="store_true",
        help="Enable dynamic INT8 quantization for smaller memory footprint."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default uses config temperature)."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable real-time audio chunk streaming mode."
    )

    args = parser.parse_args()

    # Determine execution device
    if args.device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = args.device

    if target_device == "cuda" and not torch.cuda.is_available():
        print("[!] Warning: CUDA requested but not available. Falling back to CPU.")
        target_device = "cpu"

    language_name = "portuguese_24l" if args.model_size == "24l" else "portuguese"
    fmt = os.path.splitext(args.output)[1].upper().replace('.', '') or "MP3"

    print("=" * 65)
    print("  Pocket-TTS - Unified Portuguese Text-to-Speech Engine")
    print("=" * 65)
    print(f" Language Model: {language_name}")
    print(f" Target Device : {target_device.upper()} " + (f"({torch.cuda.get_device_name(0)})" if target_device == "cuda" else ""))
    print(f" Voice Prompt  : {args.voice}")
    print(f" Output Format : {fmt}")
    print(f" Quantization  : {'Enabled (INT8)' if args.quantize else 'Disabled (FP32)'}")
    print(f" Text Input    : \"{args.text}\"")
    print("-" * 65)

    # 1. Load Model
    print(f"[1/3] Loading '{language_name}' model...")
    t_start = time.time()
    model = TTSModel.load_model(
        language=language_name,
        quantize=args.quantize,
        temp=args.temperature
    )
    if target_device == "cuda":
        model.to("cuda")
    print(f"      Loaded on {model.device} in {time.time() - t_start:.2f}s")

    # 2. Voice State Conditioning
    print(f"[2/3] Preparing voice conditioning for '{args.voice}'...")
    t_voice = time.time()
    voice_state = model.get_state_for_audio_prompt(args.voice)
    print(f"      Voice state initialized in {time.time() - t_voice:.2f}s")

    # 3. Generation
    print("[3/3] Generating Portuguese speech...")
    if target_device == "cuda":
        torch.cuda.synchronize()
    t_gen = time.time()

    if args.stream:
        print("      [Streaming Mode Enabled]")
        chunks = []
        chunk_idx = 0
        for chunk in model.generate_audio_stream(
            model_state=voice_state,
            text_to_generate=args.text,
            copy_state=True
        ):
            chunk_idx += 1
            chunks.append(chunk)
            chunk_dur = chunk.shape[-1] / model.sample_rate
            print(f"      -> Received chunk #{chunk_idx}: {chunk_dur*1000:.0f} ms ({chunk.shape[-1]} samples)")
        audio = torch.cat(chunks, dim=0)
    else:
        audio = model.generate_audio(
            model_state=voice_state,
            text_to_generate=args.text,
            copy_state=True
        )

    if target_device == "cuda":
        torch.cuda.synchronize()
    total_gen_time = time.time() - t_gen

    # Save audio (MP3 or WAV)
    save_audio_file(audio, model.sample_rate, args.output)

    audio_duration = audio.shape[-1] / model.sample_rate
    rtf = audio_duration / total_gen_time if total_gen_time > 0 else 0
    file_size_kb = os.path.getsize(args.output) / 1024

    print("-" * 65)
    print(f" Results Summary:")
    print(f" - Generated Audio Length : {audio_duration:.2f} s")
    print(f" - Total Synthesis Time   : {total_gen_time:.2f} s")
    print(f" - Real-Time Factor (RTF) : {rtf:.2f}x Real-Time")
    print(f" - Sample Rate            : {model.sample_rate} Hz")
    print(f" - Output File Saved To   : {os.path.abspath(args.output)} ({file_size_kb:.1f} KB)")
    print("=" * 65)

if __name__ == "__main__":
    main()
