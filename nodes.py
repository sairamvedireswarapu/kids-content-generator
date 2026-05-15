from typing import TypedDict, Optional
from langchain_groq import ChatGroq
from tavily import TavilyClient
from dotenv import load_dotenv
import logging
import os
from datetime import datetime
from elevenlabs.client import ElevenLabs

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
    topics = topics[-20:]   # keep last 20 only
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
    temperature= 0.9 
)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

VALUES = ["kindness", "courage", "honesty", "friendship", "patience", "gratitude", "perseverance"]
TARGET_DURATION_SECONDS = 90
WORDS_PER_MINUTE = 140
TARGET_WORD_COUNT = int((TARGET_DURATION_SECONDS / 60) * WORDS_PER_MINUTE)




class StoryState(TypedDict):
    trending_topic: str
    chosen_value: str
    outline: str
    character_descriptions: str
    scene_prompts: list        # ← new: list of {"scene": str, "characters": list[str]}
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

# ── Node 1: Trend Researcher ──────────────────────────────────────────────────
def trend_researcher(state: StoryState) -> dict:
    logger.info("[Trend Researcher] Searching for trending kids topics...")

    used_topics = load_used_topics()
    avoid_str = ", ".join(used_topics[-10:]) if used_topics else "none"

    # Rotate queries so we don't always hit the same search angle
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

    prompt = f"""You are a children's story writer.
    Write a fun, engaging story for kids aged 4-8 based on this outline:

    {state['outline']}

    Requirements:
    - Simple language a 5-year-old understands
    - Short sentences
    - Warm and positive tone
    - Naturally teach the value of {state['chosen_value']}
    - Length: {TARGET_WORD_COUNT - 20} to {TARGET_WORD_COUNT} words (be strict, do not exceed)
    - Each paragraph should be a separate scene
    - Separate each paragraph with a blank line
    - Each paragraph = one scene (different moment in the story){feedback_section}"""

    response = llm.invoke(prompt)
    topic = response.content.strip()
    logger.info(f"[Trend Researcher] Topic selected: {topic}")

    save_used_topic(topic)   # ← record it immediately
    return {"trending_topic": topic}


# ── Node 2: Outline Agent ─────────────────────────────────────────────────────
def outline_agent(state: StoryState) -> dict:
    import random
    chosen_value = random.choice(VALUES)

    # ── Step 1: Generate story outline ───────────────────────────────────────
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

    # ── Step 2: Extract detailed visual character descriptions ────────────────
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

    return {
        "chosen_value": chosen_value,
        "outline": outline,
        "character_descriptions": character_descriptions,   # ← new
    }


# ── Node 3: Story Writer ──────────────────────────────────────────────────────
def story_writer(state: StoryState) -> dict:
    critic_feedback = state.get("critic_feedback") or ""

    feedback_section = (
        f"\n\nPrevious critic feedback to address:\n{critic_feedback}"
        if critic_feedback else ""
    )

    prompt = f"""You are a children's story writer.
Write a fun, engaging story for kids aged 4-8 based on this outline:

{state['outline']}

Requirements:
- Simple language a 5-year-old understands
- Short sentences
- Warm and positive tone
- Naturally teach the value of {state['chosen_value']}
  - Length: {TARGET_WORD_COUNT - 20} to {TARGET_WORD_COUNT} words (be strict, do not exceed){feedback_section}"""
    
    response = llm.invoke(prompt)
    logger.info(f"[Story Writer] Story written (retry #{state.get('retry_count', 0)})")
    return {"story": response.content.strip()}

# ── Node 3.5: Scene Prompt Agent ──────────────────────────────────────────────
def scene_prompt_agent(state: StoryState) -> dict:
    """Convert each story paragraph into a visual scene description for Imagen."""
    logger.info("[Scene Prompt Agent] Generating visual scene descriptions...")

    story = state["story"]
    character_descriptions = state.get("character_descriptions", "")

    # Split into paragraphs same way video_generator does
    paragraphs = [p.strip() for p in story.split("\n") if p.strip()]

    # Extract character names from descriptions
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

    # Parse the response
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
            scene_prompts.append({
                "scene": scene_desc,
                "characters": characters
            })

    # Fallback — if parsing fails, use raw paragraphs
    if len(scene_prompts) != len(paragraphs):
        logger.warning(f"[Scene Prompt Agent] Parsed {len(scene_prompts)} scenes but expected {len(paragraphs)} — using fallback")
        scene_prompts = [
            {"scene": p, "characters": character_names}
            for p in paragraphs
        ]

    logger.info(f"[Scene Prompt Agent] Generated {len(scene_prompts)} scene prompts")
    return {"scene_prompts": scene_prompts}


# ── Node 4: Quality Critic ────────────────────────────────────────────────────
def quality_critic(state: StoryState) -> dict:
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
    return {
        "title": extract("TITLE"),
        "description": extract("DESCRIPTION"),
        "hashtags": hashtags,
        "thumbnail_concept": extract("THUMBNAIL")
    }



def audio_generator(state: StoryState) -> dict:
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
            voice="en-AU-NatashaNeural",  # warm, friendly female voice
            rate="-10%",   # slightly slower for kids
            pitch="+5Hz"   # slightly higher pitch, friendlier
        )
        await communicate.save(audio_path)

    asyncio.run(generate())

    logger.info(f"[Audio Generator] Audio saved to {audio_path}")
    return {"audio_path": audio_path, "story_id": story_id}

# ── Node 6: Save Content ──────────────────────────────────────────────────────


# def save_content(state: StoryState) -> dict:
#     import json
#     from datetime import datetime

#     try:
#         story_id = state.get("story_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
#         folder = f"stories/story_{story_id}"
#         os.makedirs(folder, exist_ok=True)
#         filename = f"{folder}/story.json"

#         output = {
#             "topic": state.get("trending_topic"),
#             "value": state.get("chosen_value"),
#             "outline": state.get("outline"),
#             "story": state.get("story"),
#             "scores": {
#                 "safety": state.get("safety_score"),
#                 "educational": state.get("educational_score"),
#                 "engagement": state.get("engagement_score"),
#             },
#             "passed_quality": state.get("passed_quality"),
#             "retries": state.get("retry_count"),
#             "metadata": {
#                 "title": state.get("title"),
#                 "description": state.get("description"),
#                 "hashtags": state.get("hashtags"),
#                 "thumbnail_concept": state.get("thumbnail_concept"),
#             },
#             "audio_path": state.get("audio_path"),    # ← add
#             "video_path": state.get("video_path"),    # ← add
#         }

#         with open(filename, "w", encoding="utf-8") as f:   # ← encoding fix
#             json.dump(output, f, indent=2, ensure_ascii=False)

#         logger.info(f"[Save] Story saved to {filename}")

#     except Exception as e:
#         logger.error(f"[Save] FAILED to save JSON: {e}")   # ← catch silent crash

#     return {}



def save_content(state: StoryState) -> dict:
    import json
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import asyncio

    story_id = state.get("story_id")
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

                # Tool 1 — save locally
                await session.call_tool("save_local", {
                    "folder": folder,
                    "filename": "story.json",
                    "content": output
                })
                logger.info("[MCP Client] Local save done")

                # Tool 2 — upload story.json to Drive
                await session.call_tool("upload_to_drive", {
                    "file_path": f"{folder}/story.json",
                    "drive_folder_name": "Kids Stories"
                })
                logger.info("[MCP Client] Drive upload done")

    asyncio.run(run_mcp())
    return {}
# ── Node 0: Check Topic (entry point) ────────────────────────────────────────
def check_topic_node(state: StoryState) -> dict:
    """No-op node — just routes based on whether topic is pre-filled."""
    return {}

def route_entry(state: StoryState) -> str:
    if state.get("trending_topic", "").strip():
        logger.info(f"[Router] User-provided topic: '{state['trending_topic']}' — skipping researcher")
        return "outline_agent"
    logger.info("[Router] No topic provided — running trend researcher")
    return "trend_researcher"