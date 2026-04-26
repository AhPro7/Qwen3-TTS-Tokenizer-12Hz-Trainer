# evaluate_kanade_colab.py
# Run this in Google Colab after uploading your audio files.
#
# First, install dependencies in a Colab cell:
# !pip install git+https://github.com/frothywater/kanade-tokenizer soundfile numpy torch torchaudio

import os
import time
import glob
import torch
import soundfile as sf
import numpy as np

# Import Kanade components
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode

def run_evaluation(model_id, audio_dir, output_dir):
    print(f"\n{'='*60}\nEvaluating Kanade Model: {model_id}\n{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    recon_dir = os.path.join(output_dir, "reconstruction")
    vc_dir = os.path.join(output_dir, "voice_conversion")
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(vc_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Model
    print("Loading model...")
    model = KanadeModel.from_pretrained(model_id)
    model = model.eval().to(device)

    # 2. Load Vocoder
    print(f"Loading vocoder: {model.config.vocoder_name}...")
    vocoder = load_vocoder(model.config.vocoder_name).to(device)

    # 3. Find Audio Files
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    audio_files = sorted(audio_files)

    if not audio_files:
        print(f"❌ No audio files found in {audio_dir}!")
        return

    print(f"Found {len(audio_files)} audio files.")
    
    # Store features for VC
    features_dict = {}
    
    print("\n[RECONSTRUCTION & RTF BENCHMARK]")
    total_audio_s = 0.0
    total_compute_s = 0.0

    for file_path in audio_files:
        file_name = os.path.basename(file_path)
        print(f"  Processing: {file_name}")
        
        # Load audio using Kanade's utility (handles resampling automatically)
        waveform = load_audio(file_path, sample_rate=model.config.sample_rate).to(device)
        input_dur = waveform.shape[-1] / model.config.sample_rate
        total_audio_s += input_dur
        
        # Benchmark timing
        t0 = time.perf_counter()
        
        with torch.inference_mode():
            # Encode
            features = model.encode(waveform)
            
            # Decode
            mel_spectrogram = model.decode(
                content_token_indices=features.content_token_indices,
                global_embedding=features.global_embedding,
            )
            
            # Vocode back to waveform
            resynthesized = vocode(vocoder, mel_spectrogram.unsqueeze(0)).squeeze().cpu().numpy()
        
        compute_dur = time.perf_counter() - t0
        total_compute_s += compute_dur
        rtf = compute_dur / input_dur
        
        print(f"    Input length: {input_dur:.2f}s | Compute time: {compute_dur:.4f}s | RTF: {rtf:.3f}x")
        
        # Save reconstruction
        out_path = os.path.join(recon_dir, f"recon_{file_name}.wav")
        sf.write(out_path, resynthesized, model.config.sample_rate)
        
        # Save features for VC
        features_dict[file_name] = features

    avg_rtf = total_compute_s / total_audio_s if total_audio_s > 0 else 0
    print(f"\nOverall RTF: {avg_rtf:.3f}x")

    print("\n[VOICE CONVERSION]")
    # Do Voice Conversion between the first 3 files (Content A + Speaker B)
    vc_files = audio_files[:3]
    for i in range(len(vc_files)):
        for j in range(len(vc_files)):
            if i == j: continue
            
            src_name = os.path.basename(vc_files[i])
            tgt_name = os.path.basename(vc_files[j])
            
            print(f"  Content: {src_name} → Speaker: {tgt_name}")
            
            feat_src = features_dict[src_name]
            feat_tgt = features_dict[tgt_name]
            
            t0 = time.perf_counter()
            with torch.inference_mode():
                # Swap: use SRC content and TGT speaker
                mel_vc = model.decode(
                    content_token_indices=feat_src.content_token_indices,
                    global_embedding=feat_tgt.global_embedding,
                )
                wav_vc = vocode(vocoder, mel_vc.unsqueeze(0)).squeeze().cpu().numpy()
            
            vc_time = time.perf_counter() - t0
            print(f"    VC Time: {vc_time:.4f}s")
            
            # Save VC
            out_name = f"vc_{os.path.splitext(src_name)[0]}_TO_{os.path.splitext(tgt_name)[0]}.wav"
            sf.write(os.path.join(vc_dir, out_name), wav_vc, model.config.sample_rate)

    print(f"\nDone! Outputs saved in '{output_dir}'.")

if __name__ == "__main__":
    # In Colab, you can upload your files to a folder, e.g., '/content/audios'
    # Change 'audio_dir' to point to that folder.
    
    AUDIO_FOLDER = "./audios"  # Replace with your Colab audio folder path
    
    # Evaluate 25Hz Clean
    run_evaluation(
        model_id="frothywater/kanade-25hz-clean", 
        audio_dir=AUDIO_FOLDER, 
        output_dir="./kanade_eval_output/25hz_clean"
    )
    
    # Evaluate 12.5Hz
    run_evaluation(
        model_id="frothywater/kanade-12.5hz", 
        audio_dir=AUDIO_FOLDER, 
        output_dir="./kanade_eval_output/12.5hz"
    )
    
    # Evaluate 25Hz Standard
    run_evaluation(
        model_id="frothywater/kanade-25hz", 
        audio_dir=AUDIO_FOLDER, 
        output_dir="./kanade_eval_output/25hz"
    )
