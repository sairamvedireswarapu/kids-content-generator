from nodes import (
    check_topic_node, route_entry,
    trend_researcher, outline_agent, story_writer,
    quality_critic, metadata_agent, audio_generator, save_content
)
from video_generator import video_generator
from langgraph.graph import StateGraph, END
from nodes import StoryState

def build_graph():
    graph = StateGraph(StoryState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("check_topic",      check_topic_node)
    graph.add_node("trend_researcher", trend_researcher)
    graph.add_node("outline_agent",    outline_agent)
    graph.add_node("story_writer",     story_writer)
    graph.add_node("quality_critic",   quality_critic)
    graph.add_node("metadata_agent",   metadata_agent)
    graph.add_node("audio_generator",  audio_generator)
    graph.add_node("video_generator",  video_generator)
    graph.add_node("save_content",     save_content)

    # ── Entry ──────────────────────────────────────────────────────────────────
    graph.set_entry_point("check_topic")
    graph.add_conditional_edges("check_topic", route_entry, {
        "trend_researcher": "trend_researcher",
        "outline_agent":    "outline_agent",
    })

    # ── Linear edges ───────────────────────────────────────────────────────────
    graph.add_edge("trend_researcher", "outline_agent")
    graph.add_edge("outline_agent",    "story_writer")
    graph.add_edge("story_writer",     "quality_critic")

    # ── Quality gate ──────────────────────────────────────────────────────────
    def route_after_critic(state: StoryState) -> str:
        if state["passed_quality"]:
            return "metadata_agent"
        if state.get("retry_count", 0) >= 3:
            return "save_content"
        return "story_writer"

    graph.add_conditional_edges("quality_critic", route_after_critic, {
        "metadata_agent": "metadata_agent",
        "story_writer":   "story_writer",
        "save_content":   "save_content",
    })

    graph.add_edge("metadata_agent",  "audio_generator")
    graph.add_edge("audio_generator", "video_generator")
    graph.add_edge("video_generator", "save_content")
    graph.add_edge("save_content",    END)

    return graph.compile()