"""
Voice Recording Script for TTS Dataset
=======================================
Records 100 sentences one by one, saves as WAV files.
Press ENTER to start recording each sentence.
Press ENTER again to stop and move to next.

Requirements:
    pip install sounddevice scipy numpy
"""

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import time

# ─── Config ───────────────────────────────────────────────
SAMPLE_RATE = 22050       # required by XTTS v2
CHANNELS    = 1           # mono
OUTPUT_DIR  = "wavs"
META_FILE   = "metadata.csv"
# ──────────────────────────────────────────────────────────

SENTENCES = [
    # Group 1 — Short and clear
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck.",
    "I love building things with Python.",
    "Artificial intelligence is absolutely fascinating.",
    "The sun sets beautifully in the west.",
    "My voice is unique and truly powerful.",
    "Deep learning changes absolutely everything.",
    "I enjoy solving complex and challenging problems.",
    "Technology continuously shapes our future.",

    # Group 2 — Medium sentences
    "PyTorch makes building neural networks simple and intuitive.",
    "I am training my own voice model completely from scratch.",
    "The transformer architecture revolutionized natural language processing forever.",
    "Every morning I wake up excited to write Python code.",
    "Machine learning models learn patterns from large amounts of data.",
    "I believe anyone can learn deep learning with consistent practice.",
    "Gradient descent helps the model minimize its loss function efficiently.",
    "Attention mechanisms allow transformers to focus on the most important tokens.",
    "I want to build products that help millions of people worldwide.",
    "The embedding layer converts tokens into meaningful vector representations.",
    "Fine tuning a pretrained model saves enormous amounts of training time.",
    "My goal is to become a world class machine learning engineer.",
    "Neural networks are loosely inspired by the human brain structure.",
    "I push my code to GitHub every single day without fail.",
    "Kaggle provides free GPU resources for training deep learning models.",

    # Group 3 — Conversational
    "Hey, how are you doing today?",
    "That is a really interesting question, let me think about it carefully.",
    "I was wondering if you could help me with something important.",
    "Sure, I would be absolutely happy to explain that to you.",
    "What do you think about the latest developments in artificial intelligence?",
    "Let me walk you through this process step by step.",
    "That makes a lot of sense when you really think about it.",
    "I appreciate you taking the time to listen to what I have to say.",
    "Could you please repeat that one more time for me?",
    "I think we should approach this problem from a completely different angle.",
    "Let me know if you have any questions along the way.",
    "I will do my best to explain this as clearly as possible.",
    "That is actually a much simpler concept than it first appears.",
    "Thank you so much for your patience and understanding.",
    "I am really excited to show you what I have been working on.",

    # Group 4 — Technical
    "The model achieved ninety two percent accuracy on the validation set.",
    "Backpropagation computes gradients by applying the chain rule repeatedly.",
    "Convolutional neural networks are excellent at processing image data efficiently.",
    "The learning rate controls how fast the model updates its weights.",
    "Batch normalization helps stabilize training by normalizing layer inputs.",
    "Dropout randomly deactivates neurons to prevent the model from overfitting.",
    "The softmax function converts raw logits into probability distributions.",
    "Residual connections allow gradients to flow more easily through deep networks.",
    "Transfer learning leverages pretrained models to solve new tasks faster.",
    "The attention score is computed by taking the dot product of queries and keys.",

    # Group 5 — Numbers and mixed
    "In twenty twenty four, large language models became incredibly powerful.",
    "The dataset contains over one million labeled training examples.",
    "We trained the model for fifty epochs with a batch size of thirty two.",
    "The learning rate was set to zero point zero zero one initially.",
    "After three hours of training, the loss dropped from two point four to zero point three.",
    "The model has approximately seven billion trainable parameters in total.",
    "We used eight attention heads with a head dimension of sixty four.",
    "The vocabulary size for this tokenizer is fifty thousand unique tokens.",
    "Training on four GPUs reduced the total time from twelve hours to three.",
    "The embedding dimension was set to five hundred and twelve for this experiment.",

    # Group 6 — Storytelling style
    "Once upon a time, a young engineer decided to build his own AI assistant.",
    "He started with a simple idea and slowly turned it into something incredible.",
    "Every day he learned something new and applied it to his growing project.",
    "The first time the model spoke in his voice, he could not believe his ears.",
    "He realized that with enough dedication, anyone can build amazing things.",
    "The journey of a thousand miles begins with a single line of code.",
    "Failures taught him more than any success ever could have.",
    "He documented everything so that others could follow the same path.",
    "One day, his creation would help people all around the world.",
    "And that was only just the very beginning of his incredible story.",

    # Group 7 — Questions and exclamations
    "Are you ready to start building something truly amazing today?",
    "What is the best way to learn machine learning from scratch?",
    "Have you ever wondered how voice assistants actually understand what you say?",
    "Why do transformers perform so much better than recurrent neural networks?",
    "Can a small model trained on limited data still produce great results?",
    "Wow, I cannot believe how far artificial intelligence has come in just five years!",
    "This is absolutely incredible, the model sounds exactly like a real human being!",
    "I never thought I would be able to build something this powerful on my own!",
    "Look at how fast the loss is dropping during the training process!",
    "We did it, the voice model is finally working exactly as we planned!",

    # Group 8 — Daily life
    "I usually start my day with a strong cup of chai and some coding.",
    "After lunch, I review the code I wrote in the morning for any mistakes.",
    "In the evening, I like to read research papers to stay updated with the field.",
    "On weekends, I work on personal projects that excite and challenge me.",
    "I always make sure to push my code before going to sleep at night.",
    "Learning something new every day is the most important habit I have built.",
    "I keep a notebook where I write down ideas that come to me randomly.",
    "Collaborating with other developers has taught me so much about writing clean code.",
    "The best feeling is when your model finally trains without any errors.",
    "I am proud of how far I have come in just thirty days of dedicated practice.",

    # Group 9 — Philosophical
    "The best investment you can ever make is in your own skills and knowledge.",
    "Consistency over a long period of time beats talent almost every single time.",
    "The people who change the world are the ones who refuse to give up.",
    "Every expert was once an absolute beginner who chose not to quit.",
    "The gap between where you are and where you want to be is called work.",
    "Small daily improvements lead to truly remarkable results over time.",
    "Do not compare your beginning to someone else's middle or end.",
    "The most dangerous phrase in any language is we have always done it this way.",
    "Innovation comes from people who are crazy enough to think they can change things.",
    "Build things that matter, learn things that last, and share what you discover.",

    # Group 10 — Closing sentences
    "This voice model was built entirely by me using open source tools and PyTorch.",
    "I hope this project inspires others to build their own voice AI systems.",
    "Thank you for listening to my voice throughout this entire recording session.",
    "Every sentence I recorded brought this project one step closer to completion.",
    "I am Jay, and this is my custom trained text to speech voice model.",
    "With enough data and the right model, anyone can clone their own voice.",
    "This is what happens when curiosity meets consistency and hard work every day.",
    "The future of AI is being built by people just like you and me right now.",
    "Keep learning, keep building, and never stop asking why and how things work.",
    "This is the end of the recording session. Thank you and have a wonderful day.",
]


def record_sentence(sentence_num, sentence_text):
    """Record one sentence and return the audio array."""
    print("\n" + "="*60)
    print(f"  Sentence {sentence_num}/100")
    print("="*60)
    print(f"\n  📖  {sentence_text}\n")
    input("  Press ENTER to start recording... ")

    print("  🔴  Recording... (press ENTER to stop)\n")

    frames = []
    recording = True

    def callback(indata, frame_count, time_info, status):
        if recording:
            frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype='float32', callback=callback):
        input()  # wait for enter to stop

    recording = False
    audio = np.concatenate(frames, axis=0)
    return audio


def save_audio(audio, filename):
    """Save audio array as WAV file."""
    audio_int16 = (audio * 32767).astype(np.int16)
    write(filename, SAMPLE_RATE, audio_int16)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("   🎙️  VOICE RECORDING SCRIPT — TTS Dataset")
    print("="*60)
    print(f"\n  Total sentences : 100")
    print(f"  Sample rate     : {SAMPLE_RATE} Hz")
    print(f"  Output folder   : {OUTPUT_DIR}/")
    print(f"  Metadata file   : {META_FILE}")
    print("\n  TIPS:")
    print("  ✅  Quiet room, close doors and windows")
    print("  ✅  Sit 20-30cm from microphone")
    print("  ✅  Speak naturally, do not rush")
    print("  ✅  Drink water if your throat gets dry")
    print("  ❌  No fan or AC running in background")
    print("\n" + "="*60)

    # check if resuming
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    start_idx = len(existing)

    if start_idx > 0:
        print(f"\n  ⚡  Found {start_idx} existing recordings.")
        resume = input(f"  Resume from sentence {start_idx + 1}? (y/n): ").strip().lower()
        if resume != 'y':
            start_idx = 0

    input("\n  Ready? Press ENTER to begin recording...\n")

    metadata_lines = []

    # load existing metadata if resuming
    if start_idx > 0 and os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            metadata_lines = f.read().splitlines()

    for i in range(start_idx, len(SENTENCES)):
        sentence  = SENTENCES[i]
        filename  = f"audio_{i+1:03d}.wav"
        filepath  = os.path.join(OUTPUT_DIR, filename)

        while True:
            audio = record_sentence(i + 1, sentence)

            duration = len(audio) / SAMPLE_RATE
            print(f"\n  ✅  Recorded {duration:.1f} seconds")

            action = input("  Save (s) | Re-record (r): ").strip().lower()
            if action == 's':
                save_audio(audio, filepath)
                metadata_lines.append(f"{filename}|{sentence}")
                print(f"  💾  Saved → {filepath}")
                break
            else:
                print("  🔄  Re-recording...")

        # save metadata after every sentence
        with open(META_FILE, 'w') as f:
            f.write('\n'.join(metadata_lines))

        # progress
        done = i + 1
        remaining = 100 - done
        print(f"\n  Progress: {done}/100 done | {remaining} remaining")

        if done % 30 == 0 and done < 100:
            print("\n  ☕  Take a short break! Rest your voice for 2-3 minutes.")
            input("  Press ENTER when ready to continue...\n")

    print("\n" + "="*60)
    print("  🎉  ALL 100 SENTENCES RECORDED SUCCESSFULLY!")
    print("="*60)
    print(f"\n  📁  WAV files  → {OUTPUT_DIR}/")
    print(f"  📄  Metadata   → {META_FILE}")
    print("\n  Next step: push to GitHub → fine-tune on Kaggle!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()