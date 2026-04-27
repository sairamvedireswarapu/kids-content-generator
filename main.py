from graph import build_graph

def main():
    print("=== Kids Content Generator ===\n")
    app = build_graph()

    initial_state = {
        "trending_topic": "",
        "chosen_value": "",
        "outline": "",
        "story": "",
        "safety_score": 0.0,
        "educational_score": 0.0,
        "engagement_score": 0.0,
        "critic_feedback": "",
        "retry_count": 0,
        "passed_quality": False,
        "title": "",
        "description": "",
        "hashtags": [],
        "thumbnail_concept": "",
        "audio_path": "",       # ← add
        "video_path": "",       # ← add
        "story_id": ""
    }

    result = app.invoke(initial_state)

    print("\n=== Final Output ===")
    print(f"Title     : {result.get('title')}")
    print(f"Topic     : {result.get('trending_topic')}")
    print(f"Value     : {result.get('chosen_value')}")
    print(f"Passed    : {result.get('passed_quality')}")
    print(f"Retries   : {result.get('retry_count')}")
    print(f"Hashtags  : {result.get('hashtags')}")
    print(f"Thumbnail : {result.get('thumbnail_concept')}")
    print(f"Audio     : {result.get('audio_path')}")   # ← add
    print(f"Video     : {result.get('video_path')}")   # ← add

if __name__ == "__main__":
    main()