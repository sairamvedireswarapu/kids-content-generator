from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from graph import build_graph
from google.cloud import storage
from nodes import get_duration_config, update_status, AGENT_LABELS
import os
import threading
import json

import logging
logger = logging.getLogger(__name__)

# ── LangSmith observability (2 lines) ────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

app = FastAPI()
BUCKET_NAME = "kids-content-generator-outputs"

def upload_to_gcs(local_path: str, destination_blob_name: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{destination_blob_name}"


def run_pipeline_bg(story_id: str, initial_state: dict):
    """Run the full pipeline in a background thread."""
    try:
        graph = build_graph()
        result = graph.invoke(initial_state)
        logger.info(f"[DEBUG] result keys: title={result.get('title')} audio={result.get('audio_path')} video={result.get('video_path')} story_id={result.get('story_id')}")

        cwd = os.getcwd()
        audio_path = result.get("audio_path")
        video_path = result.get("video_path")

        audio_gcs_url = None
        video_gcs_url = None

        if audio_path and os.path.exists(audio_path):
            audio_gcs_url = upload_to_gcs(audio_path, f"{story_id}/audio.mp3")
        elif audio_path:
            abs_audio = os.path.join(cwd, audio_path)
            if os.path.exists(abs_audio):
                audio_gcs_url = upload_to_gcs(abs_audio, f"{story_id}/audio.mp3")

        if video_path and os.path.exists(video_path):
            video_gcs_url = upload_to_gcs(video_path, f"{story_id}/video.mp4")
        elif video_path:
            abs_video = os.path.join(cwd, video_path)
            if os.path.exists(abs_video):
                video_gcs_url = upload_to_gcs(abs_video, f"{story_id}/video.mp4")

        # Write final status
        update_status(
            story_id=story_id,
            current_agent=None,
            completed=list(AGENT_LABELS.keys()),
            status="done",
            title=result.get("title"),
            hashtags=result.get("hashtags"),
            audio_url=audio_gcs_url,
            video_url=video_gcs_url,
        )

    except Exception as e:
        update_status(
            story_id=story_id,
            current_agent=None,
            completed=[],
            status="error",
            error=str(e)
        )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

# ── Main endpoint — async ─────────────────────────────────────────────────────
@app.post("/run")
def run_pipeline(topic: str = "", duration: int = 90):
    story_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get word count + paragraph count from duration
    config = get_duration_config(duration)

    initial_state = {
        "trending_topic":         topic,
        "character_descriptions": "",
        "scene_prompts":          [],
        "chosen_value":           "",
        "outline":                "",
        "story":                  "",
        "safety_score":           0.0,
        "educational_score":      0.0,
        "engagement_score":       0.0,
        "critic_feedback":        "",
        "retry_count":            0,
        "passed_quality":         False,
        "title":                  "",
        "description":            "",
        "hashtags":               [],
        "thumbnail_concept":      "",
        "audio_path":             "",
        "video_path":             "",
        "story_id":               story_id,
        "target_words":           config["words"],
        "target_paragraphs":      config["paragraphs"],
    }

    # Write initial status immediately
    update_status(story_id, "check_topic", [], "running")

    # Start pipeline in background thread
    thread = threading.Thread(
        target=run_pipeline_bg,
        args=(story_id, initial_state),
        daemon=True
    )
    thread.start()

    return {
        "story_id": story_id,
        "status":   "started",
        "duration": duration,
        "config":   config,
    }


# ── Status endpoint ───────────────────────────────────────────────────────────
@app.get("/status/{story_id}")
def get_status(story_id: str):
    status_path = f"stories/story_{story_id}/status.json"
    if not os.path.exists(status_path):
        return {"status": "not_found"}
    with open(status_path, encoding="utf-8") as f:
        return json.load(f)