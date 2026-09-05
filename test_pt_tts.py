import os
import sys
import time
import torch
import scipy.io.wavfile
from pocket_tts import TTSModel

def test_cpu():
    print("=== Testing Portuguese TTS on CPU ===")
    start_load = time.time()
    model = TTSModel.load_model(language="portuguese")
    print(f"Model loaded in {time.time() - start_load:.2f}s on {model.device}")
    
    text = "Olá! Este é um teste de síntese de voz em português executado no processador CPU."
    print("Encoding voice prompt 'rafael'...")
    voice_state = model.get_state_for_audio_prompt("rafael")
    
    print(f"Generating audio for text: '{text}'...")
    start_gen = time.time()
    audio = model.generate_audio(voice_state, text)
    gen_time = time.time() - start_gen
    
    audio_duration = audio.shape[-1] / model.sample_rate
    print(f"Generated {audio_duration:.2f}s of audio in {gen_time:.2f}s (RTF: {audio_duration/gen_time:.2f}x)")
    
    output_path = "output_cpu_pt.wav"
    audio_np = audio.detach().cpu().numpy()
    scipy.io.wavfile.write(output_path, model.sample_rate, audio_np)
    print(f"Saved audio to {output_path} (File size: {os.path.getsize(output_path)} bytes)")
    return audio_duration, gen_time

def test_gpu():
    print("\n=== Testing Portuguese TTS on GPU (CUDA) ===")
    if not torch.cuda.is_available():
        print("CUDA is not available on this system!")
        return None, None
        
    device = torch.device("cuda")
    print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    
    start_load = time.time()
    model = TTSModel.load_model(language="portuguese")
    model.to(device)
    print(f"Model loaded and moved to {model.device} in {time.time() - start_load:.2f}s")
    
    text = "Olá! Este é um teste de síntese de voz em português acelerado por GPU NVIDIA."
    print("Encoding voice prompt 'rafael'...")
    voice_state = model.get_state_for_audio_prompt("rafael")
    
    print(f"Generating audio for text: '{text}'...")
    start_gen = time.time()
    audio = model.generate_audio(voice_state, text)
    torch.cuda.synchronize()
    gen_time = time.time() - start_gen
    
    audio_duration = audio.shape[-1] / model.sample_rate
    print(f"Generated {audio_duration:.2f}s of audio in {gen_time:.2f}s (RTF: {audio_duration/gen_time:.2f}x)")
    
    output_path = "output_gpu_pt.wav"
    audio_np = audio.detach().cpu().numpy()
    scipy.io.wavfile.write(output_path, model.sample_rate, audio_np)
    print(f"Saved audio to {output_path} (File size: {os.path.getsize(output_path)} bytes)")
    return audio_duration, gen_time

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    test_cpu()
    test_gpu()
