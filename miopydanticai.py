
import asyncio
import time

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

load_dotenv()


# MOCK 

_AUDIO_DB: dict[str, tuple[str, str]] = {
    "AUDIO-001": ("it", "Riunione 3 marzo: discussione KPI Q1 e roadmap prodotto."),
    "AUDIO-002": ("en", "Customer support call: order delayed, escalated to team."),
    "AUDIO-003": ("it", "Intervista tecnica: architettura microservizi e deployment."),
}

_VIDEO_DB: dict[str, tuple[int, list[str]]] = {
    "VIDEO-001": (300, ["intro_slides", "live_demo", "Q&A"]),
    "VIDEO-002": (180, ["outdoor_interview", "b-roll"]),
    "VIDEO-003": (480, ["tutorial_intro", "coding_session", "review", "outro"]),
}

_DIARIZATION_DB: dict[str, list[tuple[str, str]]] = {
    "AUDIO-001": [("Marco R.", "manager"), ("Giulia T.", "engineer")],
    "AUDIO-002": [("Marco R.", "manager"), ("Unknown", "guest")],
    "AUDIO-003": [("Giulia T.", "engineer"), ("Unknown", "guest")],
}


#  AGENTE 


SYSTEM_PROMPT = (
    "Sei un agente AI specializzato in analisi audio e video. "
    "Hai accesso a questi tool:\n"
    "- transcribe_audio(audio_id): trascrive un audio (AUDIO-001, AUDIO-002, AUDIO-003)\n"
    "- classify_video_scenes(video_id): classifica scene di un video (VIDEO-001, VIDEO-002, VIDEO-003)\n"
    "- speaker_diarization(audio_id): identifica i parlanti in un audio\n"
    "Rispondi dando in output un riassunto chiaro dei risultati ottenuti."
)

model = GroqModel("llama-3.3-70b-versatile")

agent: Agent[None, str] = Agent(model, system_prompt=SYSTEM_PROMPT)


#  TOOL DEFINITIONS 

@agent.tool_plain
async def transcribe_audio(audio_id: str) -> str:
    """Trascrive un file audio identificato da audio_id.
    Restituisce la trascrizione testuale e la lingua rilevata.
    """
    t_start = time.perf_counter()
    print(f"  transcribe_audio START  t={t_start:.3f}")
    await asyncio.sleep(0.20)  
    lang, text = _AUDIO_DB.get(audio_id.upper(), ("?", "file non trovato"))
    t_end = time.perf_counter()
    print(f"  transcribe_audio END    t={t_end:.3f}  ({round((t_end - t_start) * 1000)}ms)")
    return f'[ASR] {audio_id} | lingua={lang} | testo: "{text}"'


@agent.tool_plain
async def classify_video_scenes(video_id: str) -> str:
    """Classifica le scene di un file video identificato da video_id.
    Restituisce l'elenco delle scene rilevate e la durata totale.
    """
    t_start = time.perf_counter()
    print(f"  classify_video_scenes START  t={t_start:.3f}")
    await asyncio.sleep(0.20)  # non-blocking
    duration, scenes = _VIDEO_DB.get(video_id.upper(), (0, ["non trovato"]))
    t_end = time.perf_counter()
    print(f"  classify_video_scenes END    t={t_end:.3f}  ({round((t_end - t_start) * 1000)}ms)")
    return f"[SceneClassifier] {video_id} | durata={duration}s | scene={scenes}"


@agent.tool_plain
async def speaker_diarization(audio_id: str) -> str:
    """Identifica i parlanti presenti in un file audio.
    Restituisce nome e ruolo di ogni parlante rilevato.
    """
    t_start = time.perf_counter()
    print(f"  speaker_diarization START  t={t_start:.3f}")
    await asyncio.sleep(0.20)  # non-blocking
    speakers = _DIARIZATION_DB.get(audio_id.upper(), [("?", "non trovato")])
    lines = [f"  - {name} ({role})" for name, role in speakers]
    t_end = time.perf_counter()
    print(f"  speaker_diarization END    t={t_end:.3f}  ({round((t_end - t_start) * 1000)}ms)")
    return f"[Diarization] {audio_id} | {len(speakers)} parlanti:\n" + "\n".join(lines)


#  DEMO 

def demo() -> None:
    tests = [
        ("T1", "Trascrivi l'audio AUDIO-001"),
        ("T2", "Analizza le scene del video VIDEO-002"),
        ("T3", "Trascrivi AUDIO-003 e dimmi chi parla"),
        ("T4", "Trascrivi AUDIO-002 e analizza le scene di VIDEO-001"),
    ]

    for label, task in tests:
        print(f"{label} | {task}")
        t0 = time.perf_counter()
        result = agent.run_sync(task)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        print(f"\nRisposta agente:\n{result.output}")
        print(f"\nTempo totale: {elapsed} ms")


if __name__ == "__main__":
    demo()
