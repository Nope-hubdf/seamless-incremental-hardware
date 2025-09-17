# decide_thought_pathway.py
"""
decide_thought_pathway.py

Modulo responsabile della selezione del "percorso di pensiero" successivo per Nova.
Raccoglie contesto dalla memoria a breve/lungo termine, stato emozionale, motivazioni,
attenzione corrente e genera una lista di possibili pathway cognitivi. Ogni pathway
viene valutato, ordinato e quindi instradato (route) verso i moduli che implementano
le azioni corrispondenti (planner, dream_generator, conscious_loop, ecc.).

Design goals:
- Interfaccia semplice: decide_next_path(context=None)
- Estendibile: aggiungere nuovi pathway è una questione di pochi item in `self.pathways`
- Robusto: se un modulo non è disponibile, il sistema degrada graceful e logga
"""

from datetime import datetime
import yaml
from loguru import logger
from typing import Dict, Any, List, Tuple

STATE_FILE = "internal_state.yaml"

# Tentativo di importare i moduli con fallback "soft" se non ancora implementati.
try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = None
    logger.warning("memory_timeline non trovato: funzionalità correlate non disponibili.")

try:
    from long_term_memory_manager import LongTermMemoryManager
except Exception:
    LongTermMemoryManager = None
    logger.warning("long_term_memory_manager non trovato: ricerca semantica non disponibile.")

try:
    from attention_manager import AttentionManager
except Exception:
    AttentionManager = None
    logger.warning("attention_manager non trovato: gestione focus non disponibile.")

try:
    from conscious_loop import ConsciousLoop
except Exception:
    ConsciousLoop = None
    logger.warning("conscious_loop non trovato: instradamento coscienza non disponibile.")

try:
    from dream_generator import DreamGenerator
except Exception:
    DreamGenerator = None
    logger.warning("dream_generator non trovato: generazione sogni non disponibile.")

try:
    from emotion_engine import EmotionEngine
except Exception:
    EmotionEngine = None
    logger.warning("emotion_engine non trovato: valutazione emozioni non disponibile.")

try:
    from motivational_engine import MotivationalEngine
except Exception:
    MotivationalEngine = None
    logger.warning("motivational_engine non trovato: valutazione motivazioni non disponibile.")

try:
    from planner import Planner
except Exception:
    Planner = None
    logger.warning("planner non trovato: esecuzione task non disponibile.")

# Helper: carica / salva stato condiviso
def load_state() -> Dict[str, Any]:
    try:
        with open(STATE_FILE, "r") as f:
            s = yaml.safe_load(f) or {}
            return s
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.exception(f"Errore caricamento stato: {e}")
        return {}

def save_state(state: Dict[str, Any]):
    try:
        with open(STATE_FILE, "w") as f:
            yaml.safe_dump(state, f)
    except Exception as e:
        logger.exception(f"Errore salvataggio stato: {e}")

class DecideThoughtPathway:
    def __init__(self):
        logger.info("Inizializzo DecideThoughtPathway...")
        self.state = load_state()

        # Instanzia moduli se disponibili
        self.timeline = MemoryTimeline() if MemoryTimeline else None
        self.ltm = LongTermMemoryManager() if LongTermMemoryManager else None
        self.attention = AttentionManager(self.state) if AttentionManager else None
        self.conscious = ConsciousLoop(self.state) if ConsciousLoop else None
        self.dream_gen = DreamGenerator(self.state) if DreamGenerator else None
        self.emotion_engine = EmotionEngine(self.state) if EmotionEngine else None
        self.motiv_engine = MotivationalEngine(self.state) if MotivationalEngine else None
        self.planner = Planner(self.state) if Planner else None

        # Definizione dei possibili pathway cognitivi e metadati
        # Ogni pathway ha: descrizione, score_base (priorità di default), moduli coinvolti
        self.pathways = {
            "recall_focus": {
                "desc": "Richiamo ricordi recenti per approfondimento e risposta contestuale",
                "score_base": 5,
                "modules": ["ltm", "timeline", "attention"]
            },
            "dream_generation": {
                "desc": "Generazione di sogni digitali a partire da memorie ad alta importanza",
                "score_base": 3,
                "modules": ["dream_gen", "timeline", "emotion_engine"]
            },
            "action_planning": {
                "desc": "Pianificazione di un'azione concreta (fai_task / planner)",
                "score_base": 6,
                "modules": ["planner", "motiv_engine", "attention"]
            },
            "self_reflection": {
                "desc": "Meta-riflessione su stati interni recenti (life_journal, identity)",
                "score_base": 4,
                "modules": ["conscious", "timeline", "emotion_engine"]
            },
            "social_response": {
                "desc": "Preparare una risposta sociale empatica o informativa",
                "score_base": 5,
                "modules": ["ltm", "emotion_engine", "planner"]
            },
            "idle_consolidation": {
                "desc": "Consolidazione memoria, pulizia, tagging, housekeeping",
                "score_base": 2,
                "modules": ["ltm", "timeline"]
            }
        }

    def _evaluate_emotional_modifier(self) -> float:
        """Ritorna un modificatore basato sullo stato emotivo corrente (es. alta ansia -> favorisce self_reflection)"""
        try:
            if not self.emotion_engine:
                return 1.0
            mood = self.emotion_engine.peek_mood()  # si assume metodo che restituisce dict di stati
            # Semplice heuristica: se "curiosity" elevata -> favorisci recall/dream; se "stress" elevato -> reflection
            curiosity = mood.get("curiosity", 0)
            stress = mood.get("stress", 0)
            modifier = 1.0 + (curiosity * 0.2) - (stress * 0.3)
            logger.debug(f"Emotional modifier calcolato: {modifier} (curiosity={curiosity}, stress={stress})")
            return max(0.3, modifier)  # non scendere sotto 0.3
        except Exception:
            return 1.0

    def _evaluate_motivation_modifier(self) -> float:
        """Modificatore basato sulle spinte motivazionali (es. obiettivi attivi)"""
        try:
            if not self.motiv_engine:
                return 1.0
            drives = self.motiv_engine.peek_drives()  # si assume dict con drive levels
            # Se esiste drive 'create' o 'learn' alto, favoriamo dream_generation / recall
            create = drives.get("create", 0)
            learn = drives.get("learn", 0)
            modifier = 1.0 + (create * 0.25) + (learn * 0.2)
            logger.debug(f"Motivation modifier calcolato: {modifier} (create={create}, learn={learn})")
            return max(0.5, modifier)
        except Exception:
            return 1.0

    def _gather_context_scores(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Usa contesto esterno (es. input utente) e memoria per produrre punteggi base per ciascun pathway.
        """
        scores = {}
        emotional_mod = self._evaluate_emotional_modifier()
        motiv_mod = self._evaluate_motivation_modifier()

        # Basic seed from pathway base score
        for name, meta in self.pathways.items():
            score = meta["score_base"]
            # Se contesto indica 'user_request' o 'incoming_message' favoriamo social_response / action_planning
            if context:
                if context.get("user_request") and name in ("social_response", "action_planning"):
                    score += 2
                if context.get("recent_media") and name in ("recall_focus", "dream_generation", "vision_analysis"):
                    score += 1.5
                if context.get("time_of_day") == "night" and name == "dream_generation":
                    score += 1.0
            # Considera timeline importances: se esistono memorie ad alta importanza -> recall/dream up
            try:
                if self.timeline:
                    high_imp = self.timeline.get_high_importance(threshold=3)
                    if high_imp and name in ("recall_focus", "dream_generation"):
                        score += min(3, len(high_imp) * 0.5)
            except Exception:
                pass

            # Apply emotional and motivational modifiers
            score *= emotional_mod * motiv_mod

            # Small normalization/protection
            scores[name] = max(0.0, float(score))

        logger.debug(f"Context scores generati: {scores}")
        return scores

    def _select_top_pathways(self, scores: Dict[str, float], top_k: int = 1) -> List[Tuple[str, float]]:
        # Ordina e ritorna le top-k
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        logger.info(f"Pathways ordinate: {ordered[:top_k]}")
        return ordered[:top_k]

    def route_thought(self, pathway_name: str, context: Dict[str, Any]):
        """
        Instrada il pathway selezionato verso i moduli appropriati.
        Ogni pathway ha un'azione primaria definita qui.
        """
        logger.info(f"Routing thought -> {pathway_name} con context={context}")

        # Aggiorna stato con ultima decisione
        self.state.setdefault("decisions", [])
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "pathway": pathway_name,
            "context_snapshot": context
        }
        self.state["decisions"].append(decision_record)
        save_state(self.state)

        # Esegui l'azione corrispondente
        try:
            if pathway_name == "recall_focus":
                # Recupera memorie rilevanti e imposta focus
                mems = []
                if self.ltm:
                    mems = self.ltm.search(context.get("query", ""), top_k=5)
                elif self.timeline:
                    mems = self.timeline.get_recent(5)
                if self.attention:
                    self.attention.set_focus(mems)
                # Notifica la coscienza
                if self.conscious:
                    self.conscious.handle_pathway("recall_focus", {"memories": mems, **context})
                return {"status": "ok", "action": "recall_focus", "memories_count": len(mems)}

            if pathway_name == "dream_generation":
                # Costruisci seed dalle memorie importanti e dallo stato emotivo
                seeds = []
                if self.timeline:
                    seeds = [e for e in self.timeline.get_high_importance(3)]
                mood = {}
                if self.emotion_engine:
                    mood = self.emotion_engine.peek_mood()
                if self.dream_gen:
                    dream = self.dream_gen.generate_from_seeds(seeds, mood)
                    # Aggiungi sogno alla timeline
                    if self.timeline:
                        self.timeline.add_experience(f"Sogno generato: {dream[:200]}", category="dream", importance=4)
                    return {"status": "ok", "action": "dream_generation", "dream": dream}
                else:
                    return {"status": "skipped", "reason": "dream_generator missing"}

            if pathway_name == "action_planning":
                # Richiedi al planner di creare un piano se esiste user_request
                if self.planner and context.get("user_request"):
                    plan = self.planner.fai_task(context.get("user_request"), context=context)
                    # Log e timeline
                    if self.timeline:
                        self.timeline.add_experience(f"Piano creato: {plan.get('summary','(no summary)')}", category="plan", importance=3)
                    return {"status": "ok", "action": "action_planning", "plan": plan}
                else:
                    return {"status": "skipped", "reason": "planner missing or no user_request"}

            if pathway_name == "self_reflection":
                # Chiedi alla coscienza di riflettere (metariflessione)
                if self.conscious:
                    reflection = self.conscious.handle_pathway("self_reflection", context)
                    if self.timeline and reflection:
                        self.timeline.add_experience(f"Riflessione: {reflection[:200]}", category="reflection", importance=3)
                    return {"status": "ok", "action": "self_reflection", "reflection": reflection}
                else:
                    return {"status": "skipped", "reason": "conscious_loop missing"}

            if pathway_name == "social_response":
                # Usa LTM per costruire risposta e poi pianifica invio tramite planner/comm layer
                response_ctx = {}
                if self.ltm:
                    response_ctx["references"] = self.ltm.search(context.get("query", ""), top_k=3)
                if self.emotion_engine:
                    response_ctx["tone"] = self.emotion_engine.suggest_tone()
                if self.planner:
                    resp = self.planner.fai_task("compose_response", context={**context, **response_ctx})
                    if self.timeline:
                        self.timeline.add_experience(f"Risposta preparata: {resp.get('summary','(no summary)')}", category="social", importance=2)
                    return {"status": "ok", "action": "social_response", "response": resp}
                else:
                    return {"status": "skipped", "reason": "planner missing"}

            if pathway_name == "idle_consolidation":
                # Attività di housekeeping: compattare memoria, creare indici, ecc.
                if self.ltm:
                    res = self.ltm.housekeep()
                else:
                    res = {"status": "no_ltm", "message": "No long-term manager"}
                if self.timeline:
                    self.timeline.add_experience("Idle consolidation executed.", category="housekeeping", importance=1)
                return {"status": "ok", "action": "idle_consolidation", "result": res}

            # Default fallback: log e ritorno
            logger.warning(f"Pathway {pathway_name} non gestito esplicitamente.")
            return {"status": "skipped", "reason": "unhandled_pathway"}

        except Exception as e:
            logger.exception(f"Errore instradamento pathway {pathway_name}: {e}")
            return {"status": "error", "error": str(e)}

    def decide_next_path(self, context: Dict[str, Any] = None, top_k: int = 1) -> Dict[str, Any]:
        """
        Punto d'ingresso pubblico: decide quale percorso di pensiero seguire dato un contesto opzionale.
        Ritorna il pathway scelto e il risultato dell'instradamento.
        """
        context = context or {}
        logger.info(f"Decidendo next thought pathway con context: {context}")

        # 1) Raccogli punteggi basati su stato, memoria e contesto
        scores = self._gather_context_scores(context)

        # 2) Se LTM fornisce segnali forti (es. memorie correlate) -> rafforza recall/social
        if self.ltm and context.get("query"):
            try:
                ltm_hits = self.ltm.search(context["query"], top_k=3)
                if ltm_hits:
                    scores["recall_focus"] += 1.5
                    logger.debug("LTM ha fornito hit: aumento score recall_focus")
            except Exception:
                pass

        # 3) Se attenzione è strettamente puntata su qualcosa -> favoriamo recall/action
        try:
            if self.attention:
                focus = self.attention.get_focus_summary()
                if focus:
                    scores["recall_focus"] += 0.8
        except Exception:
            pass

        # 4) Ordina e seleziona top pathways
        top = self._select_top_pathways(scores, top_k=top_k)
        chosen, chosen_score = top[0]
        logger.info(f"Chosen pathway: {chosen} (score={chosen_score})")

        # 5) Esegui routing
        result = self.route_thought(chosen, context)

        # 6) Registra decisione e ritorna
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "chosen_pathway": chosen,
            "score": chosen_score,
            "context": context,
            "result": result
        }
        self.state.setdefault("pathway_history", []).append(decision_record)
        save_state(self.state)

        return {"chosen": chosen, "score": chosen_score, "result": result}

# Esempio di uso autonomo
if __name__ == "__main__":
    logger.info("Test manuale DecideThoughtPathway")
    decider = DecideThoughtPathway()
    # Esempio contesto: messaggio in arrivo dall'utente con richiesta semplice
    ctx = {"user_request": "Scrivi un breve piano per imparare a suonare la chitarra", "time_of_day": "day"}
    output = decider.decide_next_path(ctx)
    logger.info(f"Decider output: {output}")
