# LangGraph vs PydanticAI — Guida Approfondita
### Per la tesi magistrale: implementazione A2A con MCP

---

## Indice

1. [Filosofia dei due framework](#1-filosofia-dei-due-framework)
2. [LangGraph — Come funziona davvero](#2-langgraph--come-funziona-davvero)
   - 2.1 [Il modello mentale: tutto è un grafo](#21-il-modello-mentale-tutto-è-un-grafo)
   - 2.2 [Lo Stato](#22-lo-stato)
   - 2.3 [I Nodi](#23-i-nodi)
   - 2.4 [Gli Archi e il routing condizionale](#24-gli-archi-e-il-routing-condizionale)
   - 2.5 [Il Checkpointer e la memoria](#25-il-checkpointer-e-la-memoria)
   - 2.6 [Il loop agent in LangGraph](#26-il-loop-agent-in-langgraph)
   - 2.7 [Come gestisce i tool call paralleli](#27-come-gestisce-i-tool-call-paralleli)
3. [PydanticAI — Come funziona davvero](#3-pydanticai--come-funziona-davvero)
   - 3.1 [Il modello mentale: tutto è un Agent](#31-il-modello-mentale-tutto-è-un-agent)
   - 3.2 [I Generici: Agent[DepsType, OutputType]](#32-i-generici-agentdepstype-outputtype)
   - 3.3 [Il loop interno (pydantic-graph)](#33-il-loop-interno-pydantic-graph)
   - 3.4 [I Tool: tool_plain vs tool](#34-i-tool-tool_plain-vs-tool)
   - 3.5 [RunContext e Dependency Injection](#35-runcontext-e-dependency-injection)
   - 3.6 [Type Safety: cosa significa concretamente](#36-type-safety-cosa-significa-concretamente)
   - 3.7 [Come gestisce i tool call paralleli](#37-come-gestisce-i-tool-call-paralleli)
   - 3.8 [Output strutturato](#38-output-strutturato)
   - 3.9 [Memoria e multi-turn](#39-memoria-e-multi-turn)
4. [Confronto diretto: lo stesso agente nei due framework](#4-confronto-diretto-lo-stesso-agente-nei-due-framework)
5. [Differenze architetturali profonde](#5-differenze-architetturali-profonde)
6. [Rilevanza per A2A + MCP](#6-rilevanza-per-a2a--mcp)

---

## 1. Filosofia dei due framework

### LangGraph

LangGraph nasce da un'idea precisa: **un agente è un programma, e un programma è un grafo**.
Il controllo del flusso è **esplicito e visibile**. Il programmatore disegna ogni possibile percorso di esecuzione come archi tra nodi. Nulla avviene "per magia" — se vuoi che dopo un tool call il modello LLM venga reinvocato, devi aggiungere un arco che lo dica.

L'ispirazione è quella dei **workflow engine** e delle **state machine**. LangGraph è figlio di LangChain, una libreria nata per comporre catene di LLM, e ne porta l'eredità: molte astrazioni (Message, Tool, runnable) arrivano da LangChain.

**Motto implicito:** _"Mostrami il grafo e ti dirò cosa fa l'agente."_

### PydanticAI

PydanticAI nasce da un'idea diversa: **un agente è una funzione con effetti collaterali (i tool), e il loop è un dettaglio implementativo**.
Il controllo del flusso è **implicito e gestito dal framework**. Il programmatore dichiara cosa l'agente sa fare (i tool), cosa deve produrre (output type), e di cosa ha bisogno (deps). Il resto — il loop, il parallelismo, la validazione — è compito del framework.

L'ispirazione è quella di **FastAPI e Pydantic**: type hints come contratto, validazione automatica, developer experience come priorità.

**Motto implicito:** _"Dimmi cosa sai fare e cosa vuoi produrre, ci penso io al resto."_

---

## 2. LangGraph — Come funziona davvero

### 2.1 Il modello mentale: tutto è un grafo

In LangGraph ogni applicazione è modellata come un **grafo orientato** dove:
- I **nodi** sono unità di computazione (funzioni Python)
- Gli **archi** sono le transizioni tra nodi
- Lo **stato** è l'oggetto che fluisce attraverso il grafo

```
START → [llm_node] → [tool_node_A] → [llm_node] → END
                   ↘ [tool_node_B] ↗
```

Questo non è solo una metafora: LangGraph costruisce letteralmente un grafo in memoria e lo esegue usando un motore di scheduling. Quando compili il grafo con `.compile()`, ottieni un oggetto `CompiledStateGraph` che sa come eseguire ogni nodo nell'ordine corretto.

### 2.2 Lo Stato

Lo stato è il **cuore** di LangGraph. È un `TypedDict` che viene passato da nodo a nodo. Ogni nodo riceve lo stato corrente, produce un aggiornamento parziale (`dict`), e LangGraph fonde l'aggiornamento nello stato globale.

```python
from typing import Annotated, TypedDict
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #         Il tipo Annotated dice a LangGraph COME fare il merge:
    #         operator.add = concatena le liste invece di sovrascrivere
```

Il meccanismo `Annotated[list, operator.add]` è fondamentale: senza di esso, se due nodi paralleli restituiscono entrambi `{"messages": [...]}`, LangGraph non saprebbe quale valore tenere. Con `operator.add`, li concatena.

Un nodo può restituire:
- `{"messages": [nuovo_messaggio]}` → aggiunge il messaggio alla lista
- `{}` → nessun cambiamento allo stato
- `{"messages": []}` → aggiunge una lista vuota (no-op)

### 2.3 I Nodi

Un nodo è una **funzione Python** che riceve lo stato e restituisce un aggiornamento parziale.

```python
def llm_node(state: AgentState) -> dict:
    # Legge lo stato
    history = state["messages"]
    
    # Fa qualcosa (invoca il modello)
    response = llm.invoke(history)
    
    # Restituisce l'aggiornamento (NON lo stato completo, solo il delta)
    return {"messages": [response]}
```

Cosa importante: il nodo **non modifica** lo stato direttamente. Restituisce un dizionario con i campi da aggiornare. LangGraph applica l'aggiornamento secondo le regole definite in `AgentState` (es. `operator.add` per le liste).

I nodi possono essere:
- Funzioni Python normali (sincrone)
- Funzioni async
- Istanze di classi con `__call__`

### 2.4 Gli Archi e il routing condizionale

Gli archi definiscono **chi viene eseguito dopo chi**. Ci sono due tipi:

**Archi fissi:**
```python
builder.add_edge("node_transcribe", "llm")
# → dopo node_transcribe, esegui sempre llm
```

**Archi condizionali:**
```python
builder.add_conditional_edges(
    "llm",           # nodo sorgente
    should_continue, # funzione che decide
    {
        "node_transcribe": "node_transcribe",
        "node_classify":   "node_classify",
        "end":             END,
    }
)
```

La funzione `should_continue` riceve lo stato e restituisce **uno o più** nodi destinazione:

```python
def should_continue(state: AgentState) -> list[str]:
    last = state["messages"][-1]
    
    # Nessun tool call → termina
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return ["end"]
    
    # Fan-out: restituisce TUTTI i nodi corrispondenti ai tool richiesti
    nodi = []
    for tc in last.tool_calls:
        if tc["name"] == "transcribe_audio":
            nodi.append("node_transcribe")
        elif tc["name"] == "classify_video_scenes":
            nodi.append("node_classify")
    return nodi  # → esegui ENTRAMBI in parallelo!
```

Quando `should_continue` restituisce `["node_transcribe", "node_classify"]`, LangGraph esegue entrambi i nodi **contemporaneamente** (fan-out), poi aspetta che finiscano entrambi (fan-in) prima di procedere.

### 2.5 Il Checkpointer e la memoria

Il `MemorySaver` è un **checkpointer** in-memory che salva lo stato del grafo dopo ogni nodo. Questo permette di:

1. **Riprendere** una conversazione passando lo stesso `thread_id`
2. **Ispezionare** lo stato a qualsiasi punto dell'esecuzione
3. **Fault tolerance**: se un nodo fallisce, potenzialmente si può riprendere

```python
graph = builder.compile(checkpointer=MemorySaver())

# Thread 1 — prima chiamata
result = graph.invoke(
    {"messages": [HumanMessage(content="Trascrivi AUDIO-001")]},
    config={"configurable": {"thread_id": "conv-1"}}
)

# Thread 1 — seconda chiamata (stessa conversazione!)
result = graph.invoke(
    {"messages": [HumanMessage(content="E chi parla in quell'audio?")]},
    config={"configurable": {"thread_id": "conv-1"}}
    # LangGraph recupera i messaggi precedenti dal MemorySaver
)
```

Il `thread_id` è la chiave di accesso alla conversazione. Ogni thread ha il proprio stato indipendente.

### 2.6 Il loop agent in LangGraph

In LangGraph il **loop agente** (il ciclo LLM → tool → LLM → tool → ...) è implementato **esplicitamente dal programmatore** tramite gli archi:

```
START
  ↓
[llm_node]
  ↓ (should_continue)
  ├─→ [node_transcribe] ─→ [llm_node] ← fan-in
  ├─→ [node_classify]   ─→ [llm_node] ← fan-in
  └─→ END
```

Il grafo **non termina automaticamente** dopo un tool call. Il programmatore deve:
1. Aggiungere archi da ogni nodo tool → llm_node (fan-in)
2. Nella funzione di routing, controllare se ci sono ancora tool call
3. Aggiungere un arco verso `END` quando non ci sono più tool call

Se dimentichi un arco, il grafo si blocca. Se aggiungi un arco sbagliato, il grafo va in loop infinito. Questa esplicitezza è sia un vantaggio (controllo totale) che uno svantaggio (verbosità, possibilità di errori).

### 2.7 Come gestisce i tool call paralleli

LangGraph usa un **ThreadPoolExecutor** per eseguire i nodi in parallelo quando il routing restituisce più nodi:

```python
# Internamente LangGraph fa qualcosa di simile a:
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(node_func, state) for node_func in parallel_nodes]
    results = [f.result() for f in futures]

# Poi fonde i risultati nello stato con operator.add
merged_state = reduce(merge_with_reducer, results, current_state)
```

Questo significa che i **tool sincroni con `time.sleep` funzionano perfettamente in parallelo** in LangGraph: ogni nodo gira in un thread separato, quindi `time.sleep(0.20)` in un thread non blocca gli altri.

---

## 3. PydanticAI — Come funziona davvero

### 3.1 Il modello mentale: tutto è un Agent

In PydanticAI l'unica astrazione che il programmatore vede è l'`Agent`. Non ci sono nodi, archi, o stato esplicito. L'agente è pensato come un'entità **stateless** (senza stato proprio) che può essere instanziata una volta e riusata globalmente — esattamente come una `FastAPI` app o un router.

```python
agent: Agent[None, str] = Agent(
    model,
    system_prompt="..."
)
```

Il loop interno (LLM → tool → LLM → ...) è implementato dentro `pydantic-graph`, una libreria separata inclusa nel progetto. Il programmatore non lo vede mai direttamente a meno che non usi l'API avanzata `agent.iter()`.

### 3.2 I Generici: Agent[DepsType, OutputType]

`Agent` è una classe generica Python con due parametri di tipo:

```python
Agent[DepsType, OutputType]
```

- **DepsType**: il tipo dell'oggetto dipendenze che viene iniettato a runtime. `None` significa nessuna dipendenza.
- **OutputType**: il tipo dell'output finale dell'agente. Può essere `str`, un `BaseModel` Pydantic, `bool`, `list[str]`, ecc.

Questi due parametri non sono solo annotazioni: vengono usati **attivamente** da Pydantic per validare l'output a runtime e da Pyright/mypy per il type checking statico.

```python
agent_1: Agent[None, str]           # no deps, output testuale
agent_2: Agent[MyDeps, UserProfile] # deps tipate, output strutturato
agent_3: Agent[None, bool]          # no deps, output booleano
```

### 3.3 Il loop interno (pydantic-graph)

Sotto il cofano, PydanticAI usa `pydantic-graph` per implementare il loop. Il sorgente in `_agent_graph.py` mostra tre nodi fondamentali:

```
UserPromptNode → ModelRequestNode → CallToolsNode ─→ ModelRequestNode (loop)
                                                   └→ End
```

- **`UserPromptNode`**: processa il prompt dell'utente, costruisce la lista messaggi
- **`ModelRequestNode`**: invia la richiesta al modello LLM, riceve la risposta
- **`CallToolsNode`**: esegue tutti i tool call richiesti dal modello (in parallelo), aggiunge i risultati alla lista messaggi, torna a `ModelRequestNode`

Il loop continua finché il modello non produce una risposta finale (senza tool call). A differenza di LangGraph, questo grafo è **fisso e non configurabile** — il programmatore non può aggiungere nodi personalizzati a questo livello (a meno di usare `agent.iter()` per casi avanzati).

### 3.4 I Tool: tool_plain vs tool

Ci sono due decorator per registrare tool:

**`@agent.tool_plain`** — per tool che non hanno bisogno di contesto:
```python
@agent.tool_plain
async def transcribe_audio(audio_id: str) -> str:
    """Trascrive un file audio. Il docstring diventa la descrizione per il modello."""
    result = await some_asr_service(audio_id)
    return result
```

**`@agent.tool`** — per tool che necessitano di accedere al `RunContext`:
```python
@agent.tool
async def transcribe_audio(ctx: RunContext[MyDeps], audio_id: str) -> str:
    """Trascrive un file audio usando il client HTTP iniettato."""
    response = await ctx.deps.http_client.get(f"/asr/{audio_id}")
    return response.text
```

La differenza cruciale: `@agent.tool` riceve `RunContext[DepsType]` come **primo argomento**. PydanticAI lo riconosce automaticamente dalla firma della funzione e non lo include nei parametri che il modello LLM deve fornire. Il modello vede solo `audio_id`.

I tool possono essere sia `def` (sincroni) che `async def` (asincroni). Per i tool sincroni, PydanticAI li wrappa automaticamente in `asyncio.to_thread` per non bloccare l'event loop — ma solo se il tool è registrato come `def`, non come `async def`.

> ⚠️ **Attenzione**: se scrivi `async def` e usi `time.sleep()` dentro, stai bloccando l'event loop. Usa sempre `await asyncio.sleep()` nei tool async, o `time.sleep()` nei tool sincroni (`def`).

### 3.5 RunContext e Dependency Injection

`RunContext[DepsType]` è l'oggetto che porta le dipendenze ai tool. Ha questi attributi principali:

```python
@agent.tool
async def my_tool(ctx: RunContext[MyDeps], param: str) -> str:
    ctx.deps        # l'istanza di MyDeps passata ad agent.run()
    ctx.usage       # RunUsage — token usati finora (input, output, requests)
    ctx.retry       # int — numero di tentativi correnti per questo tool
    ctx.run_id      # str — ID univoco della run corrente
    ctx.model       # Model — il modello in uso
    ctx.agent       # Agent — referenza all'agente corrente
```

Il pattern raccomandato per le dipendenze è il `@dataclass`:

```python
from dataclasses import dataclass
import httpx

@dataclass
class Deps:
    api_key: str
    http_client: httpx.AsyncClient
    db_connection: SomeDB

agent = Agent(model, deps_type=Deps)

# A runtime, inietti un'istanza:
async with httpx.AsyncClient() as client:
    deps = Deps(
        api_key=os.getenv("API_KEY"),
        http_client=client,
        db_connection=await SomeDB.connect()
    )
    result = await agent.run("fai qualcosa", deps=deps)
```

**Vantaggio per i test**: nei test puoi passare un `Deps` con client mock senza toccare il codice dell'agente.

### 3.6 Type Safety: cosa significa concretamente

"Type safety" in PydanticAI ha tre livelli:

**Livello 1 — Validazione argomenti tool a runtime:**
Quando il modello LLM chiama `transcribe_audio(audio_id="AUDIO-001")`, PydanticAI costruisce un JSON schema dalla firma della funzione e usa Pydantic per validare gli argomenti **prima** di invocare il tool. Se il modello invia `{"audio_id": 123}` invece di `{"audio_id": "AUDIO-001"}`, Pydantic lo coerce a `str` (se possibile) o genera un `ValidationError` strutturato che viene rimandato al modello come retry automatico.

```python
# Il modello ha inviato {"audio_id": 123}
# Pydantic coerce: audio_id = "123" ✓

# Il modello ha inviato {} (argomento mancante)
# Pydantic genera: RetryPromptPart con l'errore di validazione
# → il modello riceve il messaggio di errore e riprova
```

**Livello 2 — Validazione output a runtime:**
L'output dell'agente viene validato con Pydantic. Se l'`output_type` è `bool` e l'LLM restituisce `"true"` come stringa, Pydantic lo converte in `True`. Se restituisce qualcosa di incompatibile, il framework forza un retry.

**Livello 3 — Type checking statico:**
Pyright e mypy capiscono i generici di `Agent[DepsType, OutputType]`. Se scrivi:
```python
agent: Agent[None, bool] = Agent(model, output_type=bool)
result = agent.run_sync("è vero?")
result.output  # Pyright sa che questo è `bool`, non `Any`
```
Se poi scrivi `result.output + " stringa"`, Pyright ti avvisa a compile-time che `bool` non supporta la concatenazione con `str`. In LangGraph `result["messages"][-1].content` è tipato `Any` — nessun avviso.

### 3.7 Come gestisce i tool call paralleli

Quando il modello richiede più tool in una sola risposta, PydanticAI li esegue con `asyncio.create_task`:

```python
# Da _agent_graph.py (semplificato):
tasks = [
    asyncio.create_task(_call_tool(tool_manager, call), name=call.tool_name)
    for call in tool_calls  # tutti i tool richiesti dal modello
]
await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
```

La modalità di esecuzione dipende da `ParallelExecutionMode`:
- **`'parallel'`** (default): tutti i task partono insieme, si aspetta ALL_COMPLETED
- **`'parallel_ordered_events'`**: tutti i task partono insieme, gli eventi sono emessi in ordine
- **`'sequential'`**: i task vengono eseguiti uno alla volta

La modalità default è `'parallel'`. Si può sovrascrivere singolarmente per tool con `sequential=True`:

```python
@agent.tool_plain(sequential=True)  # questo tool non viene parallelizzato
async def sensitive_operation(x: str) -> str:
    ...
```

**Condizione necessaria per il parallelismo reale**: i tool devono essere `async def` con operazioni non-bloccanti. Se un tool è `def` (sincrono), PydanticAI lo wrappa in `asyncio.to_thread`, che usa un thread del pool — analogo a quello che fa LangGraph.

### 3.8 Output strutturato

A differenza di LangGraph dove l'output è sempre un messaggio testuale nell'ultima posizione di `state["messages"]`, PydanticAI può restituire **qualsiasi tipo Python** validato da Pydantic:

```python
from pydantic import BaseModel

class AudioAnalysis(BaseModel):
    audio_id: str
    language: str
    transcription: str
    speakers: list[str]

agent: Agent[None, AudioAnalysis] = Agent(model, output_type=AudioAnalysis)
result = agent.run_sync("Analizza AUDIO-001")

# result.output è un'istanza validata di AudioAnalysis
print(result.output.language)    # "it"
print(result.output.speakers)    # ["Marco R.", "Giulia T."]
```

Internamente, PydanticAI registra `AudioAnalysis` come un "output tool" (un tool speciale che il modello chiama per terminare la run). Il modello non restituisce testo libero ma deve chiamare questo tool con argomenti che matchano lo schema Pydantic di `AudioAnalysis`.

### 3.9 Memoria e multi-turn

PydanticAI non ha un sistema di memoria implicito come `MemorySaver` di LangGraph. La storia della conversazione viene gestita **esplicitamente** passandola come parametro:

```python
# Run 1
result1 = agent.run_sync("Trascrivi AUDIO-001")

# Run 2 — stessa conversazione
result2 = agent.run_sync(
    "E chi parla in quell'audio?",
    message_history=result1.new_messages()  # passa i messaggi del run precedente
)

# Run 3 — stessa conversazione
result3 = agent.run_sync(
    "Riassumi tutto",
    message_history=result2.all_messages()  # tutti i messaggi dalla storia
)
```

- `result.new_messages()`: solo i messaggi prodotti in questo run
- `result.all_messages()`: tutti i messaggi inclusi quelli passati da `message_history`

**Vantaggio**: controllo totale su cosa viene incluso nella storia. Puoi filtrare, comprimere, o trasformare i messaggi prima di passarli al run successivo.

**Svantaggio rispetto a LangGraph**: devi gestire tu la persistenza (es. salvare i messaggi su DB se vuoi conversazioni tra sessioni diverse).

---

## 4. Confronto diretto: lo stesso agente nei due framework

Questo è il codice che hai scritto per la tesi. Entrambi fanno **esattamente la stessa cosa** — ecco come si mappa ogni pezzo:

### Definizione del modello

```python
# LangGraph
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(TOOLS)
# → ChatGroq è un wrapper LangChain che adatta Groq all'interfaccia LangChain
# → .bind_tools() inietta gli schemi dei tool nel preset della chiamata API

# PydanticAI
model = GroqModel("llama-3.3-70b-versatile")
agent = Agent(model, system_prompt=SYSTEM_PROMPT)
# → GroqModel è un wrapper PydanticAI che adatta Groq all'interfaccia Model
# → i tool vengono registrati separatamente, non legati al modello
```

### Definizione di un tool

```python
# LangGraph
@tool
def transcribe_audio(audio_id: str) -> str:
    """Trascrive un file audio identificato da audio_id."""
    time.sleep(0.20)
    # ...
# → @tool di LangChain crea un oggetto StructuredTool
# → il docstring diventa la descrizione per il modello
# → DEVI aggiungere il tool a TOOLS, TOOLS_MAP, e creare un nodo dedicato

# PydanticAI
@agent.tool_plain
async def transcribe_audio(audio_id: str) -> str:
    """Trascrive un file audio identificato da audio_id."""
    await asyncio.sleep(0.20)
    # ...
# → il decorator registra direttamente il tool sull'agente
# → il docstring diventa la descrizione per il modello
# → nessun wiring manuale necessario
```

### Il loop agent

```python
# LangGraph — esplicito (scritto da te):
#
#   llm_node → should_continue → node_transcribe ─┐
#                              → node_classify   ─┤→ llm_node → ...
#                              → END
#
# ~50 righe di codice (StateGraph, add_node, add_edge, add_conditional_edges, compile)

# PydanticAI — implicito (gestito dal framework):
#
#   UserPromptNode → ModelRequestNode → CallToolsNode → ModelRequestNode → ...
#                                                     → End
#
# 0 righe di codice aggiuntive
```

### Esecuzione

```python
# LangGraph
result = graph.invoke(
    {"messages": [HumanMessage(content=task)]},
    config={"configurable": {"thread_id": "th-1"}}
)
final_answer = result["messages"][-1].content  # Any — nessuna garanzia sul tipo

# PydanticAI
result = agent.run_sync(task)
final_answer = result.output  # str — garantito da Pydantic
```

---

## 5. Differenze architetturali profonde

### 5.1 Controllo del flusso

| Aspetto | LangGraph | PydanticAI |
|---|---|---|
| Chi controlla il loop | Il programmatore (archi) | Il framework |
| Flussi non-standard | Possibile (archi custom) | Necessita `agent.iter()` |
| Visibilità del flusso | Alta (grafo ispezionabile) | Bassa (interno) |
| Errori di configurazione | Possibili (archi mancanti/sbagliati) | Impossibili (flusso fisso) |

### 5.2 Stato

| Aspetto | LangGraph | PydanticAI |
|---|---|---|
| Tipo | `TypedDict` esplicito | Implicito (lista messaggi interna) |
| Accesso | `state["messages"]` in ogni nodo | `result.all_messages()` dopo il run |
| Merge parallelo | Definito dal programmatore (`Annotated`) | Automatico |
| Persistenza | `MemorySaver` implicito per `thread_id` | Esplicita (`message_history=`) |

### 5.3 Tool

| Aspetto | LangGraph | PydanticAI |
|---|---|---|
| Registrazione | `@tool` + `TOOLS` list + nodo dedicato + arco | `@agent.tool_plain` |
| Validazione argomenti | No (stringhe grezze, parse manuale) | Sì (Pydantic, automatica) |
| Accesso a contesto | Via closure (non type-safe) | `RunContext[DepsType]` (type-safe) |
| Parallelismo | Thread pool (funziona con `time.sleep`) | asyncio tasks (richiede `async def`) |

### 5.4 Output

| Aspetto | LangGraph | PydanticAI |
|---|---|---|
| Tipo output | `Any` (`messages[-1].content`) | Qualsiasi tipo Python validato |
| Strutturazione | Manuale (parsing del testo LLM) | Automatica (`output_type=MyModel`) |
| Garanzie tipo | Nessuna | Pydantic + Pyright |

### 5.5 Complessità del codice (stesso agente)

| Metrica | LangGraph | PydanticAI |
|---|---|---|
| Righe totali (PoC) | ~175 | ~120 |
| Righe di setup agente | ~50 | ~5 |
| Concetti da conoscere | StateGraph, TypedDict, Annotated, operator.add, MemorySaver, HumanMessage, AIMessage, ToolMessage, SystemMessage, add_node, add_edge, add_conditional_edges, compile, thread_id | Agent, tool_plain/tool, RunContext, run_sync/run |

---

## 6. Rilevanza per A2A + MCP

### A2A (Agent-to-Agent)

Il pattern A2A è quando un agente delega lavoro a un altro agente come se fosse un tool.

**In PydanticAI** questo è nativamente supportato:

```python
# Agente specializzato
asr_agent = Agent(model, instructions="Sei un esperto ASR.")

# Agente orchestratore — chiama l'agente ASR come tool
@orchestrator_agent.tool
async def run_asr_pipeline(ctx: RunContext[None], audio_id: str) -> str:
    result = await asr_agent.run(
        f"Trascrivi {audio_id}",
        usage=ctx.usage  # propaga il contatore token
    )
    return result.output

# Oppure con agent.as_tool() — ancora più diretto:
orchestrator = Agent(
    model,
    tools=[asr_agent.as_tool(name="asr_service", description="Servizio ASR")]
)
```

**In LangGraph** il pattern A2A richiede di creare un nodo che invoca un secondo grafo:

```python
def asr_agent_node(state: AgentState) -> dict:
    # Invoca un secondo grafo come sub-agente
    sub_result = asr_graph.invoke({"messages": [...]})
    return {"messages": [ToolMessage(content=sub_result["messages"][-1].content, ...)]}
```

### MCP (Model Context Protocol)

MCP è uno standard per esporre tool come servizi HTTP/stdio. Un "MCP server" espone una lista di tool che qualsiasi client MCP può usare.

**In PydanticAI** l'integrazione è con un `toolset`:

```python
from pydantic_ai.mcp import MCPServerHTTP, MCPServerStdio

# Tool da un server MCP remoto
mcp_server = MCPServerHTTP(url="http://localhost:8080/mcp")

# Tool da un processo MCP locale (stdio)
mcp_process = MCPServerStdio("python", args=["my_mcp_server.py"])

agent = Agent(
    model,
    toolsets=[mcp_server, mcp_process]  # tutti i tool MCP diventano tool dell'agente
)
```

L'agente scopre automaticamente quali tool sono disponibili sul server MCP, costruisce i loro schemi, e li registra. Dal punto di vista del loop, un tool MCP funziona identicamente a un `@agent.tool_plain`.

**In LangGraph** non c'è integrazione nativa. Devi:
1. Usare il client MCP (es. `mcp` library) per scoprire i tool
2. Wrappare ogni tool MCP in un `@tool` LangChain
3. Aggiungerli al grafo come nodi

La differenza è significativa: PydanticAI può consumare un intero server MCP con 3 righe, LangGraph richiede codice di integrazione manuale per ogni tool.

### Conclusione per la tesi

Dato che il tuo obiettivo è implementare **A2A con MCP**:

- **PydanticAI** è più diretto: supporto nativo MCP come toolset, pattern A2A con `agent.as_tool()`, dependency injection per i client HTTP ai servizi remoti.
- **LangGraph** è più flessibile per flussi complessi: se il tuo sistema A2A ha routing condizionale sofisticato, stati condivisi tra agenti, o necessità di ispezionare/modificare il flusso a runtime, il grafo esplicito è più adatto.

Per un PoC di valutazione, PydanticAI riduce significativamente il boilerplate e ti permette di concentrarti sulla logica A2A+MCP piuttosto che sulla plumbing del framework.

---

*Documento generato per la tesi magistrale — Matteo Mancini — Marzo 2026*
