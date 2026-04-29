import os
import re
import time
import requests
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
)



logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# HF_API_KEY   = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VIDEO_W, VIDEO_H = 1280, 720          # 720p landscape — good for YouTube
FONT_SIZE    = 36                      # subtitle font size
MAX_CHARS    = 60                      # max chars per subtitle line
SUBTITLE_BG  = (0, 0, 0, 160)         # semi-transparent black
SUBTITLE_FG  = "white"


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_into_paragraphs(story: str) -> list[str]:
    """Split story text into clean non-empty paragraphs."""
    paragraphs = [p.strip() for p in story.split("\n") if p.strip()]
    # Merge very short lines (fewer than 30 chars) with the next paragraph
    merged = []
    buffer = ""
    for p in paragraphs:
        if len(p) < 30 and buffer:
            buffer += " " + p
        else:
            if buffer:
                merged.append(buffer)
            buffer = p
    if buffer:
        merged.append(buffer)
    return merged if merged else paragraphs


def build_image_prompt(paragraph: str, title: str, value: str) -> str:
    return (
        f"Children's storybook illustration, comic panel style, "
        f"bright vibrant colors, Pixar/Disney movie quality rendering, "
        f"expressive cute cartoon characters with big eyes, "
        f"bold clean outlines, cel shaded art style, "
        f"soft warm cinematic lighting, "
        f"safe for kids aged 4-10, cheerful and positive mood, "
        f"highly detailed background, "
        f"no text, no words, no letters, no watermark in image. "
        f"Scene: {paragraph[:250]}. "
        f"Moral theme: {value}. "
        f"Art direction: Saturday morning cartoon meets Pixar, "
        f"character design similar to Bluey or Peppa Pig style."
        f"no text, no words, no letters, no watermark, no signs, no captions in image. "
    )


def generate_image_hf(prompt: str, save_path: str, retries: int = 3) -> bool:
    """Generate image using Pollinations.AI — free, no API key needed."""
    import urllib.parse
    encoded_prompt = urllib.parse.quote(prompt)
    url = (
    f"https://image.pollinations.ai/prompt/{encoded_prompt}"
    f"?width={VIDEO_W}&height={VIDEO_H}&model=flux-pro&nologo=true&safe=true&enhance=true"
    )

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[Image Gen] Attempt {attempt} — generating image...")
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"[Image Gen] Saved to {save_path}")
                return True
            logger.error(f"[Image Gen] HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"[Image Gen] Attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)

    return False


def wrap_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Wrap long text to multiple lines for subtitles."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def get_audio_duration(audio_path: str) -> float:
    """Get mp3 duration in seconds using mutagen."""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        return audio.info.length
    except Exception:
        # Fallback: use moviepy (slower but always works)
        clip = AudioFileClip(audio_path)
        dur = clip.duration
        clip.close()
        return dur


# def make_subtitle_clip(text: str, video_w: int, video_h: int, duration: float):
#     """Create a subtitle TextClip positioned at the bottom of the frame."""
#     wrapped = wrap_text(text)
#     txt_clip = (
#     TextClip(
#         text=wrapped,
#         font_size=FONT_SIZE,
#         color=SUBTITLE_FG,
#         bg_color=(0, 0, 0, 160),   # ← RGBA tuple, not CSS string
#         method="caption",
#         size=(video_w - 80, None),
#     )
#     .with_duration(duration)
#     .with_position(("center", video_h - 160))
#     )
#     return txt_clip

def burn_subtitle_onto_image(image_path: str, text: str, output_path: str):
    """Burn subtitle text directly onto image using Pillow — no moviepy TextClip needed."""
    img = Image.open(image_path).convert("RGB").resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()

    wrapped = wrap_text(text)

    # Measure text block size
    lines = wrapped.split("\n")
    line_height = FONT_SIZE + 8
    block_height = line_height * len(lines) + 20
    block_y = VIDEO_H - block_height - 30

    # Draw semi-transparent background
    draw.rectangle(
        [(0, block_y - 10), (VIDEO_W, VIDEO_H - 20)],
        fill=(0, 0, 0, 160)
    )

    # Draw each line of text
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (VIDEO_W - text_w) // 2
        y = block_y + i * line_height
        draw.text((x, y), line, font=font, fill="white")

    img.save(output_path)


# ── Main Node ─────────────────────────────────────────────────────────────────

def video_generator(state: dict) -> dict:
    """
    LangGraph node: generates one image per story paragraph via Hugging Face,
    then assembles them with the existing audio.mp3 into a video.mp4.

    Reads from state:
        story, title, chosen_value, audio_path, story_id

    Writes to state:
        video_path
    """
    logger.info("[Video Generator] Starting video assembly...")

    story      = state["story"]
    title      = state.get("title", "Kids Story")
    value      = state.get("chosen_value", "kindness")
    audio_path = state["audio_path"]
    story_id   = state["story_id"]
    folder     = f"stories/story_{story_id}"

    # ── 1. Split story into paragraphs ────────────────────────────────────────
    paragraphs = split_into_paragraphs(story)
    logger.info(f"[Video Generator] {len(paragraphs)} paragraphs found")

    # ── 2. Get total audio duration & divide per paragraph ───────────────────
    total_duration = get_audio_duration(audio_path)
    per_paragraph  = total_duration / len(paragraphs)
    logger.info(f"[Video Generator] Audio duration: {total_duration:.1f}s → {per_paragraph:.1f}s per paragraph")

    # ── 3. Generate one image per paragraph ──────────────────────────────────
    image_paths = []
    for i, para in enumerate(paragraphs):
        img_path = f"{folder}/frame_{i:02d}.png"

        if os.path.exists(img_path):
            logger.info(f"[Video Generator] Frame {i} already exists, skipping generation")
            image_paths.append(img_path)
            continue

        prompt  = build_image_prompt(para, title, value)
        success = generate_image_hf(prompt, img_path)

        if not success:
            logger.warning(f"[Video Generator] Frame {i} failed — using fallback solid color image")
            # Fallback: plain colored image so the pipeline doesn't crash
            fallback = Image.new("RGB", (VIDEO_W, VIDEO_H), color=(70, 130, 180))
            draw = ImageDraw.Draw(fallback)
            draw.text((VIDEO_W // 2, VIDEO_H // 2), f"Scene {i+1}", fill="white", anchor="mm")
            fallback.save(img_path)

        image_paths.append(img_path)

        # Be polite to the free API — small delay between requests
        if i < len(paragraphs) - 1:
            time.sleep(2)

    # ── 4. Build video clips ──────────────────────────────────────────────────
    video_path = f"{folder}/video.mp4"

    try:
        clips = []
        for i, (img_path, para) in enumerate(zip(image_paths, paragraphs)):
            # Burn subtitle directly onto image — no TextClip
            burned_path = f"{folder}/frame_{i:02d}_sub.png"
            burn_subtitle_onto_image(img_path, para, burned_path)

            img_clip = ImageClip(burned_path).with_duration(per_paragraph)
            clips.append(img_clip)

        # ── 5. Concatenate ────────────────────────────────────────────────────
        final_video = concatenate_videoclips(clips, method="compose")

        # ── 6. Attach audio ───────────────────────────────────────────────────
        audio_clip = AudioFileClip(audio_path)
        final_video = final_video.with_audio(audio_clip)

        # ── 7. Export MP4 ─────────────────────────────────────────────────────
        final_video.write_videofile(
            video_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )
        final_video.close()
        audio_clip.close()
        logger.info(f"[Video Generator] Video saved to {video_path}")

    except Exception as e:
        logger.error(f"[Video Generator] Video assembly failed: {e}")
        video_path = ""   # mark as failed but continue

    # ── 8. Save JSON always ───────────────────────────────────────────────────
    import json
    json_path = f"{folder}/story.json"
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
        "audio_path": audio_path,
        "video_path": video_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"[Video Generator] JSON saved to {json_path}")

    return {"video_path": video_path}