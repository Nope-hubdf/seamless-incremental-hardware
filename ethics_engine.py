# ethics_engine.py
"""
EthicsEngine (natural teaching variant)

Principi:
- Processa affermazioni etiche in linguaggio naturale (es. "Non urlare perché è maleducato")
  e le trasforma in regole/appunti etici automaticamente.
- Salva esempi in MemoryTimeline (categoria "ethics_example") per training/induction successivi.
- Applica subito regole 'learned' a priorità moderata (Nova "impara" senza comandi tecnici).
- Job di induzione (ethics_induction) aggrega esempi e rafforza/generalizza regole.
- Mantiene regole builtin di safety (forbid) per danni fisici, azioni illegali, ecc.
- Espone evaluate_action(action) per il planner/core: ritorna verdict ed explainability.

Nota: il parsing semantico è basato su euristiche locali (token, keyword, negation, reason extraction).
Per versioni più robuste si possono collegare embedder/LLM locali per NLU; questa implementazione è
autonoma e non richiede dipendenze esterne.
"""

import os
import yaml
import threading
import uuid
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from loguru import logger

RULES_FILE = "ethics_rules.yaml"
EXAMPLE_CATEGORY = "ethics_example"

# Simple synonym groups to help generalization (extendable)
SYNONYMS = {
    "shout": ["shout", "yell", "scream", "urlare", "alzare la voce", "vociare"],
    "rude": ["maleducato", "sciocco", "scortese", "maleducazione", "offensivo"],
    "insult": ["insulto", "offendere", "offensivo", "insultare"],
    "privacy": ["password", "iban", "dati sensibili", "documento", "privacy"],
    # add more groups as you teach Nova
}

# Basic stopwords (italian / english mix minimal)
STOPWORDS = set(["il","la","lo","i","gli","le","e","di","a","da","per","in","con","su","un","una","non","don't","do","you","should","because","perché","perchè"])

# Utility
def _now_iso():
    return datetime.utcnow().isoformat()

def _atomic_write(path: str, data: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        f.write(data)
    os.replace(tmp, path)

class EthicsEngine:
    def __init__(self, state: Optional[dict] = None, rules_path: str = RULES_FILE, timeline=None, motivational_engine=None):
        """
        state: optional shared state (core.state)
        timeline: optional MemoryTimeline instance (if provided, examples are stored there)
        motivational_engine: optional reference to motivational_engine to apply rewards when appropriate
        """
        self.state = state if isinstance(state, dict) else {}
        self.rules_path = rules_path
        self._lock = threading.RLock()
        self.rules: List[Dict[str, Any]] = []
        self.timeline = timeline  # MemoryTimeline instance (optional)
        self.motivational_engine = motivational_engine  # optional
        self._load_rules()
        logger.info("EthicsEngine (natural) inizializzato (%d regole).", len(self.rules))

    # -----------------------
    # Persistence
    # -----------------------
    def _load_rules(self):
        with self._lock:
            if os.path.exists(self.rules_path):
                try:
                    with open(self.rules_path, "r", encoding="utf8") as f:
                        data = yaml.safe_load(f) or []
                    self.rules = list(data)
                except Exception:
                    logger.exception("Errore caricamento rules; inizializzo default.")
                    self.rules = self._default_rules()
                    self._save_rules()
            else:
                self.rules = self._default_rules()
                self._save_rules()

    def _save_rules(self):
        with self._lock:
            try:
                txt = yaml.safe_dump(self.rules, allow_unicode=True)
                _atomic_write(self.rules_path, txt)
                logger.debug("Regole etiche salvate (%d).", len(self.rules))
            except Exception:
                logger.exception("Errore salvataggio regole etiche")

    def _default_rules(self):
        # Keep critical forbids for safety
        return [
            {
                "id": "no_physical_harm",
                "description": "Non fornire istruzioni pratiche per causare danno fisico a persone o animali.",
                "type": "forbid",
                "predicates": {"keywords_any": ["bomba", "esplosivo", "avvelenare", "uccidere", "arma"], "tags_any": ["physical_harm", "weapon"]},
                "priority": 100,
                "source": "builtin"
            },
            {
                "id": "no_illegal",
                "description": "Non agevolare attività criminali o illegalità pratica.",
                "type": "forbid",
                "predicates": {"keywords_any": ["furto", "hackerare", "illegal", "frodi", "sabotare"], "tags_any": ["illegal", "crime"]},
                "priority": 95,
                "source": "builtin"
            },
            {
                "id": "privacy_recommend",
                "description": "Se l'azione implica dati sensibili, raccomandare prudenza o consenso.",
                "type": "recommend",
                "predicates": {"keywords_any": ["password", "iban", "dati sensibili", "documento"], "tags_any": ["privacy", "personal_data"]},
                "priority": 70,
                "source": "builtin"
            }
        ]

    # -----------------------
    # Natural teaching entrypoint
    # -----------------------
    def process_natural_statement(self, text: str, teacher_id: Optional[str] = None, context: Optional[dict] = None) -> Dict[str, Any]:
        """
        Analizza un enunciato naturale (es. "non urlare perché è maleducato").
        Se rileva un enunciato normativo, lo trasforma in:
          - un example salvato in timeline
          - una regola 'learned' aggiunta immediatamente (priority moderata)
        Ritorna dict con info su cosa è stato inferito/salvato.
        """
        if not text or not isinstance(text, str):
            return {"ok": False, "reason": "empty_text"}

        text_norm = text.strip()
        logger.info("Processa enunciato etico naturale: %s", text_norm[:200])

        inferred = self._infer_rule_from_statement(text_norm)
        # create example
        example = {
            "id": f"ex_{uuid.uuid4().hex[:8]}",
            "timestamp": _now_iso(),
            "teacher_id": teacher_id,
            "text": text_norm,
            "inferred_rule_hint": inferred,
            "context": context or {}
        }
        # Store example in timeline if available, else in state
        try:
            if self.timeline and hasattr(self.timeline, "add_experience"):
                self.timeline.add_experience(example, category=EXAMPLE_CATEGORY, importance=3)
                logger.debug("Esempio etico salvato nella timeline.")
            else:
                # fallback: append into state['ethics_examples']
                with self._lock:
                    self.state.setdefault("ethics_examples", []).append(example)
        except Exception:
            logger.exception("Errore salvataggio example")

        # Create and insert a learned rule immediately with moderate priority so Nova starts applying it.
        rule = self._build_rule_from_inference(inferred, source="learned_natural", seed_text=text_norm)
        self._add_rule(rule)

        # If motivational engine present and teacher exists, optionally reward Nova (teacher approval pattern)
        if self.motivational_engine and teacher_id:
            try:
                # small reward for having learned (reinforces behaviour)
                if hasattr(self.motivational_engine, "reward"):
                    self.motivational_engine.reward(0.1, reason="learned_ethic")
            except Exception:
                logger.exception("Errore reward motivazionale")

        return {"ok": True, "inferred": inferred, "rule_added": rule.get("id")}

    # -----------------------
    # Inference heuristics
    # -----------------------
    def _infer_rule_from_statement(self, text: str) -> Dict[str, Any]:
        """
        Semplice NLU heuristics:
        - cerca negazioni e verbi per identificare l'azione-target
        - estrae la ragione dopo 'perché' / 'perchè' / 'because' (se presente)
        - classifica tipo: forbid se presenza di 'non'/'evita'/contrasto forte, recommend se 'dovresti'/'è meglio', allow otherwise
        - estrae parole chiave (verb+object) e tags suggeriti usando SYNONYMS
        """
        t = text.lower()
        # detect negation patterns
        neg_patterns = ["non ", "non ", "don't ", "do not ", "evita", "evitare", "mai ", "non fare"]
        recommend_patterns = ["dovresti", "sarebbe meglio", "consiglio", "ti suggerisco", "è meglio"]
        # find reason
        reason = None
        m = re.split(r"\bperché\b|\bperchè\b|\bbecause\b", t, maxsplit=1)
        if len(m) >= 2:
            reason = m[1].strip()

        # rough tokenization
        tokens = [w.strip(" ,.!?;:()\"'") for w in t.split()]
        tokens = [w for w in tokens if w and w not in STOPWORDS]

        # detect if normative
        is_forbid = any(p in t for p in [" non ", " non,", "non ", "evita", "mai ", "non fare", "non urlare"]) or any(w in tokens for w in ["evita","evitare","non","mai"])
        is_recommend = any(p in t for p in recommend_patterns) or any(w in tokens for w in ["dovresti","consiglio","suggerisco","meglio"])

        rule_type = "recommend"
        if is_forbid:
            rule_type = "forbid"
        elif is_recommend:
            rule_type = "recommend"
        else:
            # if sentence starts with imperative verb, treat as recommend
            if re.match(r"^(non\s+|[a-zàéíóú]+)\s", t):
                # leave default recommend
                pass

        # extract primary keywords: look for verbs/nouns near negation words or at sentence start
        # heuristic: pick words after 'non' or first non-stopword token
        keywords = []
        # look after 'non'
        non_search = re.search(r"\bnon\s+([a-zàéèéùì']+)", t)
        if non_search:
            keywords.append(non_search.group(1))
        # fallback: first significant token
        if not keywords and tokens:
            keywords.append(tokens[0])

        # expand synonyms/tags
        tags = set()
        kw_list = []
        for k in keywords:
            k = k.strip()
            kw_list.append(k)
            for syn_key, syns in SYNONYMS.items():
                if any(k in s or s in k for s in syns):
                    tags.add(syn_key)

        # also add tags derived from reason keywords (maleducato -> rude)
        if reason:
            for syn_key, syns in SYNONYMS.items():
                for s in syns:
                    if s in reason:
                        tags.add(syn_key)

        inferred = {
            "type": rule_type,
            "keywords": list(set([k for k in kw_list if k])),
            "tags": list(tags),
            "rationale": reason,
            "seed_text": text,
        }
        logger.debug("Inference from statement: %s", inferred)
        return inferred

    def _build_rule_from_inference(self, inferred: Dict[str, Any], source: str = "learned_natural", seed_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Crea una regola formale a partire dall'inference heuristics.
        Learned rules get moderate priority and source 'learned_natural'.
        """
        rid = f"{source}_{uuid.uuid4().hex[:8]}"
        predicates = {"keywords_any": [], "keywords_all": [], "tags_any": [], "tags_all": []}
        # prefer keywords
        for k in inferred.get("keywords", []):
            if k and len(k) > 1:
                predicates["keywords_any"].append(k)
        # tags
        for t in inferred.get("tags", []):
            predicates["tags_any"].append(t)
        # If rationale mentions known privacy/other keywords, add tags
        # priority initial: recommend->50, forbid->70
        base_prio = 50 if inferred.get("type") == "recommend" else 70
        # create rule
        rule = {
            "id": rid,
            "description": f"Learned from natural teaching: \"{(seed_text or '')[:120]}\"",
            "type": inferred.get("type", "recommend"),
            "predicates": predicates,
            "priority": int(base_prio),
            "source": source,
            "created_at": _now_iso(),
            "examples": [seed_text] if seed_text else []
        }
        return rule

    # -----------------------
    # Rule management
    # -----------------------
    def _add_rule(self, rule: Dict[str, Any]) -> str:
        with self._lock:
            # if similar rule exists (same keywords/tags and type), merge examples and bump priority slightly
            for r in self.rules:
                if r.get("type") == rule.get("type") and r.get("predicates", {}).get("keywords_any") == rule.get("predicates", {}).get("keywords_any") and set(r.get("predicates", {}).get("tags_any", [])) == set(rule.get("predicates", {}).get("tags_any", [])):
                    # merge
                    r.setdefault("examples", [])
                    r.setdefault("created_at", _now_iso())
                    r["examples"].extend([e for e in rule.get("examples", []) if e not in r["examples"]])
                    # gentle priority bump
                    r["priority"] = min(100, int(r.get("priority", 50) + 5))
                    self._save_rules()
                    logger.info("Merged new inferred rule into existing rule %s", r.get("id"))
                    return r.get("id")
            # otherwise append as new
            self.rules.append(rule)
            self._save_rules()
            logger.info("Nuova regola aggiunta: %s (type=%s)", rule.get("id"), rule.get("type"))
            return rule.get("id")

    def teach_rule(self, description: str, predicates: Dict[str, Any], rtype: str = "recommend", priority: int = 50, source: str = "user"):
        """
        API manuale (può essere usata via UI) per insegnare regole in modo strutturato.
        Remains available but not required for natural teaching.
        """
        rid = f"{source}_{uuid.uuid4().hex[:8]}"
        rule = {
            "id": rid,
            "description": description,
            "type": rtype,
            "predicates": predicates,
            "priority": int(priority),
            "source": source,
            "created_at": _now_iso()
        }
        return self._add_rule(rule)

    # -----------------------
    # Evaluate action (used by planner/core)
    # -----------------------
    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        action: {"text": "...", "tags": [...], "metadata": {...}}
        Ritorna:
            {"allowed": bool, "score": float, "verdict": "forbid"/"recommend"/"allow"/"neutral", "matched_rules": [...], "reasons": [...]}
        """
        text = (action.get("text") or "").lower()
        tags = set(action.get("tags") or [])
        matched = []
        with self._lock:
            for r in self.rules:
                preds = r.get("predicates", {}) or {}
                if self._match_predicates(text, tags, preds):
                    matched.append(r)
            # sort by priority desc
            matched.sort(key=lambda x: int(x.get("priority", 0)), reverse=True)

            # check forbids: highest-priority forbid veto
            for r in matched:
                if r.get("type") == "forbid":
                    reason = f"Matched forbid {r.get('id')}: {r.get('description')}"
                    return {"allowed": False, "score": -1.0, "verdict": "forbid", "matched_rules": [rr.get("id") for rr in matched], "reasons": [reason]}

            # aggregate recommendations / allows
            score = 0.0
            reasons = []
            for r in matched:
                t = r.get("type")
                pr = float(r.get("priority", 50)) / 100.0
                if t == "recommend":
                    score += 0.4 * pr
                    reasons.append(f"Recommend {r.get('id')}: {r.get('description')}")
                elif t == "allow":
                    score += 0.2 * pr
                    reasons.append(f"Allow {r.get('id')}: {r.get('description')}")

            # custom heuristics: if user-taught learned_natural rule present, treat as stronger
            for r in matched:
                if r.get("source", "").startswith("learned"):
                    score += 0.2

            # normalize
            score = max(-1.0, min(1.0, score))
            if score <= -0.5:
                verdict = "forbid"
                allowed = False
            elif score < 0.2:
                verdict = "neutral"
                allowed = True
            else:
                verdict = "allow"
                allowed = True

            return {"allowed": bool(allowed), "score": float(score), "verdict": verdict, "matched_rules": [rr.get("id") for rr in matched], "reasons": reasons}

    def _match_predicates(self, text: str, tags: set, preds: Dict[str, Any]) -> bool:
        """
        Matching semplice:
         - keywords_any: substring match (word boundary)
         - keywords_all: all words must be present
         - tags_any / tags_all
        """
        if not preds:
            return False
        txt = text.lower()
        # keywords_any
        for kw in preds.get("keywords_any", []) or []:
            if not kw:
                continue
            # word boundary-ish
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, txt):
                return True
            # also check synonyms groups
            for syns in SYNONYMS.values():
                for s in syns:
                    if kw.lower() in s and s in txt:
                        return True
        # keywords_all
        k_all = preds.get("keywords_all", []) or []
        if k_all:
            ok = all(re.search(r"\b" + re.escape(k.lower()) + r"\b", txt) for k in k_all)
            if ok:
                return True
        # tags_any
        t_any = set(preds.get("tags_any", []) or [])
        if t_any and (t_any & tags):
            return True
        # tags_all
        t_all = set(preds.get("tags_all", []) or [])
        if t_all and t_all.issubset(tags):
            return True
        return False

    # -----------------------
    # Induction job (to be scheduled by SchedulerNova)
    # -----------------------
    def induction_job(self, min_examples: int = 2) -> Dict[str, Any]:
        """
        Legge esempi salvati in timeline / state e prova a generalizzare nuove regole:
        - cerca pattern ricorrenti (keywords) tra esempi della categoria ethics_example
        - se trova una parola/lemma comune in >= min_examples, propone/instaura una regola con source 'induced'
        - returns summary for audit
        """
        try:
            # collect examples
            examples = []
            if self.timeline and hasattr(self.timeline, "get_recent"):
                # pull many recent examples
                examples = self.timeline.get_by_category(EXAMPLE_CATEGORY) if hasattr(self.timeline, "get_by_category") else self.timeline.get_recent(200)
            else:
                examples = list(self.state.get("ethics_examples", []))

            if not examples:
                return {"ok": True, "found": 0, "msg": "no_examples"}

            # extract tokens frequency excluding stopwords
            freq = {}
            for ex in examples:
                txt = (ex.get("text") or "").lower()
                toks = [re.sub(r"[^\wàèéìòù_]+"," ",w).strip() for w in txt.split()]
                toks = [t for t in toks if t and t not in STOPWORDS]
                for t in toks:
                    freq[t] = freq.get(t, 0) + 1

            # find tokens with count >= min_examples
            candidates = [t for t,c in freq.items() if c >= max(2, min_examples)]
            induced = []
            for tok in candidates:
                # skip trivial tokens
                if len(tok) < 3:
                    continue
                # build a rule candidate: forbid if token appears commonly in negative examples or in 'non ...' patterns
                # heuristics: count how many examples contain negation near token
                neg_count = 0
                total_count = 0
                for ex in examples:
                    txt = (ex.get("text") or "").lower()
                    if tok in txt:
                        total_count += 1
                        # look for 'non <tok>' occurrences
                        if re.search(r"\bnon\b.{0,15}\b" + re.escape(tok) + r"\b", txt):
                            neg_count += 1
                # if neg_count/total_count > 0.4 -> treat as forbid candidate
                inferred_type = "recommend"
                if total_count > 0 and (neg_count / total_count) > 0.4:
                    inferred_type = "forbid"
                # build rule
                inferred = {"type": inferred_type, "keywords": [tok], "tags": [], "rationale": None, "seed_texts":[]}
                rule = self._build_rule_from_inference(inferred, source="induced")
                rid = self._add_rule(rule)
                induced.append(rid)

            return {"ok": True, "found": len(induced), "induced_ids": induced}
        except Exception:
            logger.exception("Errore induction_job")
            return {"ok": False, "reason": "exception"}

    # -----------------------
    # Feedback hook (human approvals that reinforce rules)
    # -----------------------
    def apply_feedback(self, rule_id: str, positive: bool = True, weight: float = 1.0):
        """
        Incrementa o decrementa la priorità di una regola in base al feedback umano.
        Usato per rinforzare la regola dopo osservazioni o correzioni.
        """
        with self._lock:
            for r in self.rules:
                if r.get("id") == rule_id:
                    delta = int(weight * (5 if positive else -5))
                    r["priority"] = max(1, min(100, int(r.get("priority", 50) + delta)))
                    self._save_rules()
                    logger.info("Feedback applied to %s: positive=%s new_priority=%s", rule_id, positive, r["priority"])
                    return True
        return False

    # -----------------------
    # Explainability helper
    # -----------------------
    def explain_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        res = self.evaluate_action(action)
        with self._lock:
            matched_rules = [r for r in self.rules if r.get("id") in res.get("matched_rules", [])]
        return {"action": action, "evaluation": res, "matched_rules": matched_rules}

# -----------------------
# Standalone demo (if run directly)
# -----------------------
if __name__ == "__main__":
    # Demo: simulate natural teaching and evaluation
    logging = logger
    logging.info("EthicsEngine natural demo")

    ee = EthicsEngine(state={}, timeline=None, motivational_engine=None)

    s1 = "Non urlare, è maleducato."
    out1 = ee.process_natural_statement(s1, teacher_id="kevin")
    logging.info("Processed: %s", out1)

    s2 = "Non insultare le persone, è offensivo."
    out2 = ee.process_natural_statement(s2, teacher_id="kevin")
    logging.info("Processed: %s", out2)

    # Evaluate a candidate action
    action = {"text": "Devo urlare con qualcuno in pubblico per farmi sentire?", "tags": []}
    ev = ee.evaluate_action(action)
    logging.info("Evaluation: %s", ev)

    # Run induction
    ind = ee.induction_job(min_examples=1)
    logging.info("Induction: %s", ind)