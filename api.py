from fastapi import FastAPI
from datetime import datetime
from graph import build_graph
from google.cloud import storage
import os

app = FastAPI()
BUCKET_NAME = "kids-content-generator-outputs"

def upload_to_gcs(local_path: str, destination_blob_name: str) -> str:
    """Upload a file to GCS and return its public URL."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{destination_blob_name}"

# Health check
@app.get("/")
def home():
    return {"status": "Kids Content Generator is running!"}

# Main endpoint
@app.post("/run")
def run_pipeline(topic: str = ""):
    graph = build_graph()
    story_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    initial_state = {
        "trending_topic":    topic,
        "character_descriptions": "",
        "scene_prompts":     [], 
        "chosen_value":      "",
        "outline":           "",
        "story":             "",
        "safety_score":      0.0,
        "educational_score": 0.0,
        "engagement_score":  0.0,
        "critic_feedback":   "",    
        "retry_count":       0,
        "passed_quality":    False,
        "title":             "",
        "description":       "",
        "hashtags":          [],
        "thumbnail_concept": "",
        "audio_path":        "",
        "video_path":        "",
        "story_id":          story_id,
    }
    result = graph.invoke(initial_state)

    cwd = os.getcwd()
    print(f"[GCS] Current working directory: {cwd}")

    # Upload files to GCS
    audio_gcs_url = None
    video_gcs_url = None
    audio_path = result.get("audio_path")
    video_path = result.get("video_path")

    print(f"[GCS] audio_path from result: {audio_path}")
    print(f"[GCS] video_path from result: {video_path}")

    if audio_path and os.path.exists(audio_path):
        print(f"[GCS] Uploading audio...")
        audio_gcs_url = upload_to_gcs(audio_path, f"{story_id}/audio.mp3")
        print(f"[GCS] Audio uploaded: {audio_gcs_url}")
    else:
        print(f"[GCS] Audio not found at: {audio_path}, cwd: {cwd}")
        # Try absolute path
        abs_audio = os.path.join(cwd, audio_path) if audio_path else None
        print(f"[GCS] Trying absolute path: {abs_audio}, exists: {os.path.exists(abs_audio) if abs_audio else False}")
        if abs_audio and os.path.exists(abs_audio):
            audio_gcs_url = upload_to_gcs(abs_audio, f"{story_id}/audio.mp3")
            print(f"[GCS] Audio uploaded via abs path: {audio_gcs_url}")

    if video_path and os.path.exists(video_path):
        print(f"[GCS] Uploading video...")
        video_gcs_url = upload_to_gcs(video_path, f"{story_id}/video.mp4")
        print(f"[GCS] Video uploaded: {video_gcs_url}")
    else:
        print(f"[GCS] Video not found at: {video_path}, cwd: {cwd}")
        # Try absolute path
        abs_video = os.path.join(cwd, video_path) if video_path else None
        print(f"[GCS] Trying absolute path: {abs_video}, exists: {os.path.exists(abs_video) if abs_video else False}")
        if abs_video and os.path.exists(abs_video):
            video_gcs_url = upload_to_gcs(abs_video, f"{story_id}/video.mp4")
            print(f"[GCS] Video uploaded via abs path: {video_gcs_url}")

    return {
        "status":    "done",
        "title":     result.get("title"),
        "topic":     result.get("trending_topic"),
        "passed":    result.get("passed_quality"),
        "retries":   result.get("retry_count"),
        "hashtags":  result.get("hashtags"),
        "audio":     audio_gcs_url,
        "video":     video_gcs_url,
    }