from pydub import AudioSegment
import os

def split_audio(audio_file, segment_length=5000, output_dir="data/split_audio/"):
    audio = AudioSegment.from_wav(audio_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    segments = []
    for i in range(0, len(audio), segment_length):
        segment = audio[i:i+segment_length]
        segment_path = os.path.join(output_dir, f"{os.path.basename(audio_file)}_segment_{i}.wav")
        segment.export(segment_path, format="wav")
        segments.append(segment_path)
    return segments
