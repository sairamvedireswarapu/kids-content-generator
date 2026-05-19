from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from tavily import TavilyClient
from dotenv import load_dotenv
import logging
import os
from datetime import datetime
import json
import random

USED_TOPICS_FILE = "used_topics.json"

def load_used_topics() -> list:
    if os.path.exists(USED_TOPICS_FILE):
        with open(USED_TOPICS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []

def save_used_topic(topic: str):
    topics = load_used_topics()
    topics.append(topic)
    topics = topics[-20:]
    with open(USED_TOPICS_FILE, "w", encoding="utf-8") as f:
        json.dump(topics, f)

load_dotenv()

month = datetime.now().strftime("%B")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("story_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.9
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

VALUES = ["kindness", "courage", "honesty", "friendship", "patience", "gratitude", "perseverance"]
WORDS_PER_MINUTE = 140

# ── Duration presets ──────────────────────────────────────────────────────────
DURATION_PRESETS = {
    30:  {"words": 70,  "paragraphs": 2},   # Test Pipeline
    60:  {"words": 140, "paragraphs": 4},   # Short
    90:  {"words": 210, "paragraphs": 6},   # Medium
    120: {"words": 280, "paragraphs": 8},   # Long
}

def get_duration_config(duration_seconds: int) -> dict:
    """Get word count and paragraph count for a given duration."""
    if duration_seconds in DURATION_PRESETS:
        return DURATION_PRESETS[duration_seconds]
    # Custom duration — calculate dynamically
    words = int((duration_seconds / 60) * WORDS_PER_MINUTE)
    paragraphs = max(2, min(8, round(duration_seconds / 20)))
    return {"words": words, "paragraphs": paragraphs}


# ── Status tracking ───────────────────────────────────────────────────────────
def update_status(story_id: str, current_agent: str, completed: list, status: str, **kwargs):
    """Write current pipeline status to a JSON file for UI polling."""
    folder = f"stories/story_{story_id}"
    os.makedirs(folder, exist_ok=True)
    path = f"{folder}/status.json"
    data = {
        "current_agent": current_agent,
        "completed": completed,
        "status": status,
        "updated_at": datetime.now().isoformat(),
        **kwargs
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Agent order for status tracking ──────────────────────────────────────────
AGENT_ORDER = [
    "trend_researcher",
    "outline_agent",
    "story_writer",
    "quality_critic",
    "scene_prompt_agent",
    "metadata_agent",
    "audio_generator",
    "video_generator",
    "save_content",
]

AGENT_LABELS = {
    "trend_researcher":  "🔍 Researching trending topics",
    "outline_agent":     "📝 Creating story outline",
    "story_writer":      "✍️  Writing the story",
    "quality_critic":    "🎯 Quality check",
    "scene_prompt_agent":"🎨 Generating scene descriptions",
    "metadata_agent":    "🏷️  Creating metadata",
    "audio_generator":   "🎙️  Generating audio narration",
    "video_generator":   "🎬 Generating images & video",
    "save_content":      "💾 Saving & uploading",
}


class StoryState(TypedDict):
    trending_topic: str
    chosen_value: str
    outline: str
    character_descriptions: str
    scene_prompts: list
    story: str
    safety_score: float
    educational_score: float
    engagement_score: float
    critic_feedback: str
    retry_count: int
    passed_quality: bool
    title: str
    description: str
    hashtags: list
    thumbnail_concept: str
    audio_path: str
    story_id: str
    video_path: str
    target_words: int       # ← new: passed from api.py based on duration
    target_paragraphs: int  # ← new: passed from api.py based on duration


# ── Node 1: Trend Researcher ──────────────────────────────────────────────────
def trend_researcher(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    completed = []
    update_status(story_id, "trend_researcher", completed, "running")

    logger.info("[Trend Researcher] Searching for trending kids topics...")

    used_topics = load_used_topics()
    avoid_str = ", ".join(used_topics[-10:]) if used_topics else "none"

    search_queries = [
        f"trending kids activities hobbies {month} 2026",
        f"popular kids story themes {month} 2026",
        f"what are children interested in {month} 2026",
        f"kids learning topics educational {month} 2026",
        f"popular kids shows themes {month} 2026",
    ]
    query = random.choice(search_queries)
    logger.info(f"[Trend Researcher] Search query: {query}")

    results = tavily.search(query=query, max_results=7)
    snippets = "\n".join(
        f"- {r['title']}: {r['content'][:150]}"
        for r in results.get("results", [])
    )

    topic_prompt = f"""You are a kids content strategist.
Based on these trending topics, pick ONE specific, engaging story topic for children aged 4-8.
Avoid these recently used topics: {avoid_str}

Trending topics:
{snippets}

Reply with ONLY the topic name, nothing else. Example: "Friendly Forest Animals" or "Space Adventure with a Robot"."""

    response = llm.invoke(topic_prompt)
    topic = response.content.strip()
    logger.info(f"[Trend Researcher] Topic selected: {topic}")
    save_used_topic(topic)

    update_status(story_id, "outline_agent", ["trend_researcher"], "running")
    return {"trending_topic": topic}


# ── Node 2: Outline Agent ─────────────────────────────────────────────────────
def outline_agent(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    completed = ["trend_researcher"] if not state.get("trending_topic_user") else []
    update_status(story_id, "outline_agent", completed, "running")

    chosen_value = random.choice(VALUES)

    outline_prompt = f"""You are a children's story planner.
Topic: {state['trending_topic']}
Value to teach: {chosen_value}

Write a short story outline with these sections:
- Characters (2-3 max)
- Problem
- Journey (2-3 steps)
- Resolution
- Lesson (one sentence)

Keep it simple and suitable for ages 4-8."""

    outline_response = llm.invoke(outline_prompt)
    outline = outline_response.content.strip()

    character_prompt = f"""You are an animation art director for a children's cartoon show.

Based on this story outline, write a compact visual description for each character.
These will be used as image generation prompts — keyword style, not prose.

For each character include:
- Species, age, gender
- Hair/fur color and style
- Eye color
- Clothing (color + type)
- 1 unique accessory
- Art style: cel-shaded cartoon, bold outlines, bright colors

Format EXACTLY like this:
CHARACTER: <name>
DESCRIPTION: <name>: <species/age/gender>, <hair>, <eye color>, <clothing>, <accessory>. Cel-shaded cartoon style, bold clean outlines, bright colors.

Example:
CHARACTER: Emma
DESCRIPTION: Emma: shy 6-year-old girl, curly brown hair, big brown eyes, light blue floral dress, white socks, pink hair clip. Cel-shaded cartoon style, bold clean outlines, bright colors.

Story outline:
{outline}"""

    character_response = llm.invoke(character_prompt)
    character_descriptions = character_response.content.strip()

    logger.info(f"[Outline Agent] Value chosen: {chosen_value}")
    logger.info(f"[Outline Agent] Character descriptions generated")

    update_status(story_id, "story_writer", ["trend_researcher", "outline_agent"], "running")
    return {
        "chosen_value": chosen_value,
        "outline": outline,
        "character_descriptions": character_descriptions,
    }


# ── Node 3: Story Writer ──────────────────────────────────────────────────────
def story_writer(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    update_status(story_id, "story_writer",
                  ["trend_researcher", "outline_agent"], "running")

    critic_feedback = state.get("critic_feedback") or ""
    feedback_section = (
        f"\n\nPrevious critic feedback to address:\n{critic_feedback}"
        if critic_feedback else ""
    )

    target_words = state.get("target_words", 210)
    target_paragraphs = state.get("target_paragraphs", 6)

    prompt = f"""You are a children's story writer.
Write a fun, engaging story for kids aged 4-8 based on this outline:

{state['outline']}

Requirements:
- Simple language a 5-year-old understands
- Short sentences
- Warm and positive tone
- Naturally teach the value of {state['chosen_value']}
- Length: {target_words - 20} to {target_words} words (be strict, do not exceed)
- Write EXACTLY {target_paragraphs} paragraphs
- Each paragraph = one scene (different moment in the story)
- Separate each paragraph with a blank line{feedback_section}"""

    response = llm.invoke(prompt)
    logger.info(f"[Story Writer] Story written (retry #{state.get('retry_count', 0)})")

    update_status(story_id, "quality_critic",
                  ["trend_researcher", "outline_agent", "story_writer"], "running")
    return {"story": response.content.strip()}


# ── Node 3.5: Scene Prompt Agent ──────────────────────────────────────────────
def scene_prompt_agent(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    update_status(story_id, "scene_prompt_agent",
                  ["trend_researcher", "outline_agent", "story_writer", "quality_critic"], "running")

    logger.info("[Scene Prompt Agent] Generating visual scene descriptions...")

    story = state["story"]
    character_descriptions = state.get("character_descriptions", "")

    paragraphs = [p.strip() for p in story.split("\n") if p.strip()]

    character_names = []
    for line in character_descriptions.splitlines():
        if line.startswith("CHARACTER:"):
            name = line.replace("CHARACTER:", "").strip()
            character_names.append(name)

    names_str = ", ".join(character_names)
    paragraphs_str = "\n".join(
        f"PARAGRAPH {i+1}: {p}" for i, p in enumerate(paragraphs)
    )

    prompt = f"""You are a visual director for a children's animated storybook.

Convert each story paragraph into a visual scene description for an image generator.
Also identify which characters from the list appear in each scene.

Rules:
- Describe only what is VISIBLE — setting, characters present, actions, expressions, objects
- No dialogue, no narrative, no emotions told — only what a camera would see
- Keep each scene description under 50 words
- Only include characters who are actually present in that paragraph

Available characters: {names_str}

{paragraphs_str}

Reply ONLY in this exact format, one block per paragraph:
SCENE 1:
DESCRIPTION: <visual scene description>
CHARACTERS: <comma-separated character names present, or NONE>

SCENE 2:
DESCRIPTION: <visual scene description>
CHARACTERS: <comma-separated character names present, or NONE>

(continue for all paragraphs)"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    scene_prompts = []
    blocks = raw.split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        scene_desc = ""
        characters = []
        for line in lines:
            if line.startswith("DESCRIPTION:"):
                scene_desc = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("CHARACTERS:"):
                chars_raw = line.replace("CHARACTERS:", "").strip()
                if chars_raw.upper() != "NONE":
                    characters = [c.strip() for c in chars_raw.split(",")]
        if scene_desc:
            scene_prompts.append({"scene": scene_desc, "characters": characters})

    if len(scene_prompts) != len(paragraphs):
        logger.warning(f"[Scene Prompt Agent] Parsed {len(scene_prompts)} scenes but expected {len(paragraphs)} — using fallback")
        scene_prompts = [
            {"scene": p, "characters": character_names}
            for p in paragraphs
        ]

    logger.info(f"[Scene Prompt Agent] Generated {len(scene_prompts)} scene prompts")

    update_status(story_id, "metadata_agent",
                  ["trend_researcher", "outline_agent", "story_writer",
                   "quality_critic", "scene_prompt_agent"], "running")
    return {"scene_prompts": scene_prompts}


# ── Node 4: Quality Critic ────────────────────────────────────────────────────
def quality_critic(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    update_status(story_id, "quality_critic",
                  ["trend_researcher", "outline_agent", "story_writer"], "running")

    prompt = f"""You are a children's content quality critic.
Evaluate this story on three dimensions. Reply ONLY in this exact format:

SAFETY_SCORE: <number 0-10>
SAFETY_REASON: <one sentence>
EDUCATIONAL_SCORE: <number 0-10>
EDUCATIONAL_REASON: <one sentence>
ENGAGEMENT_SCORE: <number 0-10>
ENGAGEMENT_REASON: <one sentence>
FEEDBACK: <specific actionable feedback for the writer if any score is below 7>

Story to evaluate:
{state['story']}

Outline it was based on:
{state['outline']}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    def extract(label):
        for line in raw.splitlines():
            if line.startswith(label + ":"):
                return line.split(":", 1)[1].strip()
        return ""

    safety = float(extract("SAFETY_SCORE") or 0)
    educational = float(extract("EDUCATIONAL_SCORE") or 0)
    engagement = float(extract("ENGAGEMENT_SCORE") or 0)
    feedback = extract("FEEDBACK")

    overall = (safety + educational + engagement) / 3
    passed = overall >= 8.0

    logger.info(f"[Quality Critic] Safety={safety} Educational={educational} Engagement={engagement} Overall={overall:.1f} Passed={passed}")

    return {
        "safety_score": safety,
        "educational_score": educational,
        "engagement_score": engagement,
        "critic_feedback": feedback,
        "passed_quality": passed,
        "retry_count": state.get("retry_count", 0) + 1
    }


# ── Node 5: Metadata Agent ────────────────────────────────────────────────────
def metadata_agent(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    update_status(story_id, "metadata_agent",
                  ["trend_researcher", "outline_agent", "story_writer",
                   "quality_critic", "scene_prompt_agent"], "running")

    prompt = f"""You are a kids YouTube content strategist.
Based on this story about {state['chosen_value']}, generate metadata.
Reply ONLY in this exact format:

TITLE: <catchy YouTube title for kids, max 60 chars>
DESCRIPTION: <2-3 sentence YouTube description>
HASHTAGS: <5 hashtags separated by commas, no spaces>
THUMBNAIL: <one sentence describing the thumbnail image concept>

Story:
{state['story']}"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    def extract(label):
        for line in raw.splitlines():
            if line.startswith(label + ":"):
                return line.split(":", 1)[1].strip()
        return ""

    hashtags_raw = extract("HASHTAGS")
    hashtags = [h.strip() for h in hashtags_raw.split(",")]

    logger.info("[Metadata Agent] Metadata generated")

    update_status(story_id, "audio_generator",
                  ["trend_researcher", "outline_agent", "story_writer",
                   "quality_critic", "scene_prompt_agent", "metadata_agent"], "running")
    return {
        "title": extract("TITLE"),
        "description": extract("DESCRIPTION"),
        "hashtags": hashtags,
        "thumbnail_concept": extract("THUMBNAIL")
    }


# ── Node 6: Audio Generator ───────────────────────────────────────────────────
def audio_generator(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    update_status(story_id, "audio_generator",
                  ["trend_researcher", "outline_agent", "story_writer",
                   "quality_critic", "scene_prompt_agent", "metadata_agent"], "running")

    logger.info("[Audio Generator] Converting story to audio...")

    import asyncio
    import edge_tts
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    story_id = state.get("story_id") or timestamp

    folder = f"stories/story_{story_id}"
    os.makedirs(folder, exist_ok=True)
    audio_path = f"{folder}/audio.mp3"

    async def generate():
        communicate = edge_tts.Communicate(
            text=state["story"],
            voice="en-AU-NatashaNeural",
            rate="-10%",
            pitch="+5Hz"
        )
        await communicate.save(audio_path)

    asyncio.run(generate())

    logger.info(f"[Audio Generator] Audio saved to {audio_path}")

    update_status(story_id, "video_generator",
                  ["trend_researcher", "outline_agent", "story_writer",
                   "quality_critic", "scene_prompt_agent", "metadata_agent",
                   "audio_generator"], "running")
    return {"audio_path": audio_path, "story_id": story_id}


# ── Node 7: Save Content ──────────────────────────────────────────────────────
def save_content(state: StoryState) -> dict:
    story_id = state.get("story_id", "")
    update_status(story_id, "save_content",
                  ["trend_researcher", "outline_agent", "story_writer",
                   "quality_critic", "scene_prompt_agent", "metadata_agent",
                   "audio_generator", "video_generator"], "running")

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import asyncio

    folder = f"stories/story_{story_id}"

    output = {
        "topic": state.get("trending_topic"),
        "value": state.get("chosen_value"),
        "outline": state.get("outline"),
        "story": state.get("story"),
        "scores": {
            "safety": state.get("safety_score"),
            "educational": state.get("educational_score"),
            "engagement": state.get("engagement_score"),
        },
        "passed_quality": state.get("passed_quality"),
        "retries": state.get("retry_count"),
        "metadata": {
            "title": state.get("title"),
            "description": state.get("description"),
            "hashtags": state.get("hashtags"),
            "thumbnail_concept": state.get("thumbnail_concept"),
        },
        "audio_path": state.get("audio_path"),
        "video_path": state.get("video_path"),
    }

    async def run_mcp():
        server_params = StdioServerParameters(
            command="python",
            args=["mcp_server.py"],
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.call_tool("save_local", {
                    "folder": folder,
                    "filename": "story.json",
                    "content": output
                })
                logger.info("[MCP Client] Local save done")
                await session.call_tool("upload_to_drive", {
                    "file_path": f"{folder}/story.json",
                    "drive_folder_name": "Kids Stories"
                })
                logger.info("[MCP Client] Drive upload done")

    asyncio.run(run_mcp())
    return {}


# ── Node 0: Check Topic ───────────────────────────────────────────────────────
def check_topic_node(state: StoryState) -> dict:
    return {}

def route_entry(state: StoryState) -> str:
    if state.get("trending_topic", "").strip():
        logger.info(f"[Router] User-provided topic: '{state['trending_topic']}' — skipping researcher")
        return "outline_agent"
    logger.info("[Router] No topic provided — running trend researcher")
    return "trend_researcher"