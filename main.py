from graph import build_graph
from datetime import datetime

def main():
    print("=== Kids Content Generator ===\n")
    print("Enter a topic (or press Enter to auto-research trending topics):")
    print("Examples: Animals, Rama and Sita, Space Adventure, Dinosaurs, Ocean\n")
    user_topic = input("Topic: ").strip()

    app = build_graph()

    initial_state = {
        "trending_topic":    user_topic,
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

    if user_topic:
        print(f"\n✅ Using your topic: '{user_topic}'\n")
    else:
        print("\n🔍 No topic given — researching trending kids topics...\n")

    result = app.invoke(initial_state)

    print("\n=== Final Output ===")
    print(f"Title     : {result.get('title')}")
    print(f"Topic     : {result.get('trending_topic')}")
    print(f"Value     : {result.get('chosen_value')}")
    print(f"Passed    : {result.get('passed_quality')}")
    print(f"Retries   : {result.get('retry_count')}")
    print(f"Hashtags  : {result.get('hashtags')}")
    print(f"Thumbnail : {result.get('thumbnail_concept')}")
    print(f"Audio     : {result.get('audio_path')}")
    print(f"Video     : {result.get('video_path')}")

if __name__ == "__main__":
    main()