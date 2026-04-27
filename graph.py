from langgraph.graph import StateGraph, END
from nodes import (
    StoryState,
    trend_researcher,
    outline_agent,
    story_writer,
    quality_critic,
    metadata_agent,
    audio_generator,
    save_content,
    logger,
)
from video_generator import video_generator   # ← new import


def route_after_critic(state: StoryState) -> str:
    if state["passed_quality"]:
        return "metadata_agent"
    elif state["retry_count"] >= 3:
        logger.info("[Router] Max retries reached — saving anyway")
        return "save_content"
    else:
        logger.info(f"[Router] Quality failed — retrying (attempt {state['retry_count']})")
        return "story_writer"


def build_graph():
    graph = StateGraph(StoryState)

    graph.add_node("trend_researcher",  trend_researcher)
    graph.add_node("outline_agent",     outline_agent)
    graph.add_node("story_writer",      story_writer)
    graph.add_node("quality_critic",    quality_critic)
    graph.add_node("metadata_agent",    metadata_agent)
    graph.add_node("audio_generator",   audio_generator)
    graph.add_node("video_generator",   video_generator)   # ← new node
    graph.add_node("save_content",      save_content)

    graph.set_entry_point("trend_researcher")

    graph.add_edge("trend_researcher",  "outline_agent")
    graph.add_edge("outline_agent",     "story_writer")
    graph.add_edge("story_writer",      "quality_critic")
    graph.add_conditional_edges("quality_critic", route_after_critic)
    graph.add_edge("metadata_agent",    "audio_generator")
    graph.add_edge("audio_generator",   "video_generator")   # ← new edge
    graph.add_edge("video_generator",   "save_content")      # ← updated
    graph.add_edge("save_content",      END)

    return graph.compile()