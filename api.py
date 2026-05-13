from fastapi import FastAPI
from datetime import datetime
from graph import build_graph

app = FastAPI()

# Health check — to confirm app is running
@app.get("/")
def home():
    return {"status": "Kids Content Generator is running!"}

# Main endpoint — triggers your pipeline
@app.post("/run")
def run_pipeline(topic: str = ""):
    graph = build_graph()

    initial_state = {
        "trending_topic":    topic,
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
        "story_id":          datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    result = graph.invoke(initial_state)

    return {
        "status":    "done",
        "title":     result.get("title"),
        "topic":     result.get("trending_topic"),
        "passed":    result.get("passed_quality"),
        "retries":   result.get("retry_count"),
        "hashtags":  result.get("hashtags"),
        "audio":     result.get("audio_path"),
        "video":     result.get("video_path"),
    }
