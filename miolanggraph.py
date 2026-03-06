
import os
import time
from typing import Annotated, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# TOOL 
@tool
def transcribe_audio(audio_id: str) -> str:
    """
    Trascrive un file audio identificato da audio_id .
    Restituisce la trascrizione testuale e la lingua rilevata.
    """
    time.sleep(0.20)
    db = {
        "AUDIO-001": ("it", "Riunione 3 marzo: discussione KPI Q1 e roadmap prodotto."),
        "AUDIO-002": ("en", "Customer support call: order delayed, escalated to team."),
        "AUDIO-003": ("it", "Intervista tecnica: architettura microservizi e deployment."),
    }
    lang, text = db.get(audio_id.upper(), ("?", "file non trovato"))
    return f"[ASR] {audio_id} | lingua={lang} | testo: \"{text}\""


@tool
def classify_video_scenes(video_id: str) -> str:
    """
    Classifica le scene di un file video identificato da video_id .
    Restituisce l'elenco delle scene rilevate e la durata totale.
    """
    time.sleep(0.20)
    db = {
        "VIDEO-001": (300, ["intro_slides", "live_demo", "Q&A"]),
        "VIDEO-002": (180, ["outdoor_interview", "b-roll"]),
        "VIDEO-003": (480, ["tutorial_intro", "coding_session", "review", "outro"]),
    }
    duration, scenes = db.get(video_id.upper(), (0, ["non trovato"]))
    return f"[SceneClassifier] {video_id} | durata={duration}s | scene={scenes}"


@tool
def speaker_diarization(audio_id: str) -> str:
    """
    Identifica i parlanti presenti in un file audio .
    Restituisce nome e ruolo di ogni parlante rilevato.
    """
    time.sleep(0.20)
    db = {
        "AUDIO-001": [("Marco R.", "manager"), ("Giulia T.", "engineer")],
        "AUDIO-002": [("Marco R.", "manager"), ("Unknown", "guest")],
        "AUDIO-003": [("Giulia T.", "engineer"), ("Unknown", "guest")],
    }
    speakers = db.get(audio_id.upper(), [("?", "non trovato")])
    lines = [f"  - {name} ({role})" for name, role in speakers]
    return f"[Diarization] {audio_id} | {len(speakers)} parlanti:\n" + "\n".join(lines)


TOOLS = [transcribe_audio, classify_video_scenes, speaker_diarization]
TOOLS_MAP = {t.name: t for t in TOOLS}

# STATE

class AgentState(TypedDict):
    messages: Annotated[list, operator.add] 


# GRAFO

SYSTEM_PROMPT = """Sei un agente AI specializzato in analisi audio e video.
Hai accesso a questi tool:
- transcribe_audio(audio_id): trascrive un audio (AUDIO-001, AUDIO-002, AUDIO-003)
- classify_video_scenes(video_id): classifica scene di un video (VIDEO-001, VIDEO-002, VIDEO-003)
- speaker_diarization(audio_id): identifica i parlanti in un audio
Rispondi dando in output un riassunto chiaro dei risultati ottenuti."""


def build_agent(model: str = "llama-3.3-70b-versatile") -> object:
    llm = ChatGroq(model=model, temperature=0).bind_tools(TOOLS)

    def llm_node(state: AgentState) -> dict:
        history = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in history):
            history = [SystemMessage(content=SYSTEM_PROMPT)] + history

        response = llm.invoke(history)
        return {"messages": [response]}

    def _make_tool_node(tool_name: str):
        def node(state: AgentState) -> dict:
            t_start = time.perf_counter()
            print(f" {tool_name} START  t={t_start:.3f}")
            last_msg = state["messages"][-1]
            for tc in last_msg.tool_calls:
                if tc["name"] == tool_name:
                    result = TOOLS_MAP[tool_name].invoke(tc["args"])
                    t_end = time.perf_counter()
                    print(f" {tool_name} END    t={t_end:.3f}  (durata {round((t_end-t_start)*1000)}ms)")
                    return {"messages": [ToolMessage(content=str(result), tool_call_id=tc["id"])]}
            return {"messages": []}
        node.__name__ = f"node_{tool_name}"
        return node

    def should_continue(state: AgentState) -> list[str]:
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return ["end"]
        nodi = []
        for tc in last.tool_calls:
            if tc["name"] == "transcribe_audio":
                nodi.append("node_transcribe")
            elif tc["name"] == "classify_video_scenes":
                nodi.append("node_classify")
            elif tc["name"] == "speaker_diarization":
                nodi.append("node_diarize")
        return nodi if nodi else ["end"]

    builder = StateGraph(AgentState)
    builder.add_node("llm",            llm_node)
    builder.add_node("node_transcribe", _make_tool_node("transcribe_audio"))
    builder.add_node("node_classify",   _make_tool_node("classify_video_scenes"))
    builder.add_node("node_diarize",    _make_tool_node("speaker_diarization"))

    builder.set_entry_point("llm")
    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "node_transcribe": "node_transcribe",
            "node_classify":   "node_classify",
            "node_diarize":    "node_diarize",
            "end":             END,
        },
    )
    builder.add_edge("node_transcribe", "llm")
    builder.add_edge("node_classify",   "llm")
    builder.add_edge("node_diarize",    "llm")

    return builder.compile(checkpointer=MemorySaver())


# DEMO

def demo() -> None:
    graph = build_agent()

    tests = [
        ("T1", "Trascrivi l'audio AUDIO-001", "th-1"),
        ("T2", "Analizza le scene del video VIDEO-002", "th-2"),
        ("T3", "Trascrivi AUDIO-003 e dimmi chi parla", "th-3"),
        ("T4", "Trascrivi AUDIO-002e analizza le scene di VIDEO-001",    "th-4"),
    ]
    for label, task, tid in tests:
        print(f"\n{'─' * 72}")
        print(f"{label} | {task}")
        t0 = time.perf_counter()
        result = graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"configurable": {"thread_id": tid}},
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        final = result["messages"][-1].content
        print(f"\nRisposta agente:\n{final}")
        print(f"\nTempo totale: {elapsed} ms")

if __name__ == "__main__":
    demo()
