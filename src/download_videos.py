from pytube import YouTube
from moviepy.editor import VideoFileClip
import os

def clean_url(url):
    # Strip any URL parameters after "?"
    return url.split('?')[0]

def download_video(video_url, output_path="data/audio/"):
    video_url = clean_url(video_url)  # Clean the URL
    yt = YouTube(video_url)
    
    # Get the highest-quality audio stream
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    video_path = os.path.join(output_path, yt.title + ".mp4")
    audio_stream.download(output_path=output_path)
    
    return video_path

def extract_audio(video_path, output_audio_path="data/audio/"):
    video = VideoFileClip(video_path)
    audio_path = os.path.join(output_audio_path, os.path.basename(video_path).replace(".mp4", ".wav"))
    video.audio.write_audiofile(audio_path)
    return audio_path

def download_and_extract_multiple_videos(video_urls, output_path="data/audio/"):
    audio_paths = []
    for url in video_urls:
        try:
            video_path = download_video(url, output_path)
            audio_path = extract_audio(video_path, output_path)
            print(f"Downloaded and extracted audio from: {url}")
            audio_paths.append(audio_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            print("Detailed error information:", e.__class__.__name__, e)
    
    return audio_paths


if __name__ == "__main__":
    video_urls = [
        # "https://www.youtube.com/watch?v=fWnGTCSHT3A",
        # "https://youtu.be/jyGJobYf_0k?si=QqZkuORCnn8X7wdE",
        # "https://youtu.be/N5Qu-tZcJtQ?si=eS4U6jB7dJegyjLe",
        # "https://youtu.be/G-nFH55duU0?si=hgwL7pp5KGGI5K36",
        # "https://youtu.be/FupZSRp__kQ?si=SORroHEYQRULUf_x",
        # "https://youtu.be/C1g7jf7HcjU?si=L-MB10huHob9TlBW",
        # "https://youtu.be/meWWSndp94o?si=X1jF4JhF0ICERe9u",
        # "https://youtu.be/DoGpaygdZ_U?si=mJrcDDX1Pxe7_27E",
        # "https://youtu.be/DUI2kaRqgZ4?si=IgwGjxt-RfoLMexl",
        # "https://youtu.be/Hrx9gCG1nEo?si=MU9unkE9bp92W1hy",
        # "https://youtu.be/H4IrUf3G2oU?si=eEr5g-PwJaqt8bUb",
        # "https://youtu.be/_hJFLPTddIw?si=tVsKSCmTFo44A-mt",
        # "https://youtu.be/7BalldUG9Us?si=7qgXL33CZlJ3fNp3",
        # "https://youtu.be/Un17BLxRLRU?si=4yW2Om7hLmIL2kEE",
        # "https://youtu.be/Gku1TE19lGc?si=9Uekxzghu3Cdk797"
    ]
    
    audio_files = download_and_extract_multiple_videos(video_urls)
    print("Downloaded and extracted audio files:", audio_files)
