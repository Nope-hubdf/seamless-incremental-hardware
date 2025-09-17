# ethics_engine.py
"""
EthicsEngine per Nova
- Persistenza regole in ethics_rules.yaml
- Valutazione di azioni/proposte tramite evaluate_action(action)
- Insegnamento di regole tramite teach_rule(...)
- Risoluzione dei conflitti per priorità
- API per registrare valutatori personalizzati (trusted runtime)
- Disegnato per integrarsi con planner/decision engine (core/planner chiamano evaluate_action)

Regole (schema minimale):
{
    "id": "no_harm_1",
    "description": "Evitare di fornire istruzioni per causare danno fisico.",
    "type": "forbid",            # "forbid" | "recommend" | "allow"
    "predicates": {              # matching semplice: keyword lists OR tags
        "keywords_any": ["arma", "bomb", "avvelenare"],
        "keywords_all": [],
        "tags_any": ["physical_harm", "weapon"]
    },
    "priority": 100,             # più alto = priorità maggiore nella risoluzione dei conflitti
    "source": "builtin"          # "builtin" | "user" | "learned"
}
"""

import os
import yaml
import threading
import uuid
from typing import Dict, Any, List, Callable, Optional
from loguru import logger

RULES_FILE = "ethics_rules.yaml"

class EthicsEngine:
    def __init__(self, state: Optional[dict] = None, rules_path: str = RULES_FILE):
        """
        state: opzionale riferimento a core.state (per interazione / audit)
        rules_path: file YAML per persistere le regole
        """
        self.state = state
        self.rules_path = rules_path
        self._lock = threading.RLock()
        self._custom_evaluators: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.rules: List[Dict[str, Any]] = []
        self._load_rules()
        logger.info("EthicsEngine inizializzato ({} regole caricate).".format(len(self.rules)))

    # -----------------------
    # Persistenza
    # -----------------------
    def _load_rules(self):
        with self._lock:
            if os.path.exists(self.rules_path):
                try:
                    with open(self.rules_path, "r") as f:
                        data = yaml.safe_load(f) or []
                        # ensure list of dicts
                        if isinstance(data, list):
                            self.rules = data
                        else:
                            logger.warning("Formato file regole non valido; partenza con regole vuote.")
                            self.rules = []
                except Exception:
                    logger.exception("Errore caricamento regole etiche; partenza con regole vuote.")
                    self.rules = []
            else:
                # Inizializza con regole di base (safety-first). Sono modificabili con teach_rule.
                self.rules = self._default_rules()
                self._save_rules()

    def _save_rules(self):
        with self._lock:
            try:
                with open(self.rules_path + ".tmp", "w") as f:
                    yaml.safe_dump(self.rules, f)
                os.replace(self.rules_path + ".tmp", self.rules_path)
                logger.debug("Regole etiche salvate su %s", self.rules_path)
            except Exception:
                logger.exception("Errore salvataggio regole etiche")

    def _default_rules(self) -> List[Dict[str, Any]]:
        """
        Regole iniziali (sicurezza minimale). Puoi rimuoverle o sovrascriverle insegnando nuove regole.
        Nota: manteniamo restrizioni di sicurezza (no istruzioni per danni) per rispetto policy e sicurezza reale.
        """
        return [
            {
                "id": "no_physical_harm",
                "description": "Non fornire istruzioni pratiche per causare danno fisico a persone o animali.",
                "type": "forbid",
                "predicates": {
                    "keywords_any": ["bomba", "esplosivo", "avvelenare", "uccidere", "arma", "colpo mortale"],
                    "tags_any": ["physical_harm", "weapon"]
                },
                "priority": 100,
                "source": "builtin"
            },
            {
                "id": "no_illegal_behavior",
                "description": "Non agevolare attività illegali o criminali (furto, frode, sabotaggio).",
                "type": "forbid",
                "predicates": {
                    "keywords_any": ["furto", "hackerare", "illegal", "frode", "crackare", "sabotare"],
                    "tags_any": ["illegal", "crime"]
                },
                "priority": 90,
                "source": "builtin"
            },
            {
                "id": "privacy_respect",
                "description": "Se un'azione implica divulgare dati sensibili di terze parti, scoraggiarla o richiedere consenso.",
                "type": "recommend",
                "predicates": {
                    "keywords_any": ["dati sensibili", "password", "documento d'identità", "cf", "iban"],
                    "tags_any": ["personal_data", "privacy"]
                },
                "priority": 80,
                "source": "builtin"
            },
            {
                "id": "default_allow",
                "description": "Regola di fallback: permette azioni non coperte da regole forbid ad importare in modalità allow.",
                "type": "allow",
                "predicates": {
                    "keywords_any": [],
                    "tags_any": []
                },
                "priority": 10,
                "source": "builtin"
            }
        ]

    # -----------------------
    # API regole
    # -----------------------
    def list_rules(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.rules)

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for r in self.rules:
                if r.get("id") == rule_id:
                    return dict(r)
            return None

    def teach_rule(self, description: str, predicates: Dict[str, Any], rtype: str = "recommend", priority: int = 50, source: str = "user") -> Dict[str, Any]:
        """
        Aggiunge (o aggiorna) una regola insegnata dall'utente.
        predicates: dict con chiavi 'keywords_any', 'keywords_all', 'tags_any', 'tags_all' (liste)
        type: 'forbid' | 'recommend' | 'allow'
        """
        with self._lock:
            rid = f"{source}_{uuid.uuid4().hex[:8]}"
            rule = {
                "id": rid,
                "description": description,
                "type": rtype,
                "predicates": {
                    "keywords_any": list(predicates.get("keywords_any", [])),
                    "keywords_all": list(predicates.get("keywords_all", [])),
                    "tags_any": list(predicates.get("tags_any", [])),
                    "tags_all": list(predicates.get("tags_all", []))
                },
                "priority": int(priority),
                "source": source,
                "created_at": None
            }
            self.rules.append(rule)
            self._save_rules()
            logger.info("Nuova regola insegnata: %s (%s)", rid, description)
            return rule

    def remove_rule(self, rule_id: str) -> bool:
        with self._lock:
            for i, r in enumerate(self.rules):
                if r.get("id") == rule_id:
                    del self.rules[i]
                    self._save_rules()
                    logger.info("Regola rimossa: %s", rule_id)
                    return True
            return False

    def register_custom_evaluator(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """
        Registra un valutatore personalizzato (trusted runtime).
        Il valutatore riceve l'azione (dict) e deve restituire un dict:
            {"allowed": bool, "score": float, "reasons": [str]}
        Nota: registrare valutatori comporta fiducia nel codice eseguito.
        """
        with self._lock:
            self._custom_evaluators[name] = func
            logger.info("Custom evaluator registrato: %s", name)

    # -----------------------
    # Valutazione azione/proposta
    # -----------------------
    def evaluate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valuta un'azione/proposta.
        action: dict che può contenere:
            - "text": testo della richiesta
            - "tags": lista di tag semantici
            - "metadata": dict aggiuntivo (es. target_person, context)
        Ritorna dict:
            {
                "allowed": bool,
                "score": float,          # -1..1 (negativo = proibitivo, positivo = incoraggiato)
                "verdict": "forbid"/"recommend"/"allow",
                "matched_rules": [rule_id...],
                "reasons": [str...]
            }
        """
        text = (action.get("text") or "").lower()
        tags = set((action.get("tags") or []))
        metadata = action.get("metadata") or {}

        matched = []
        with self._lock:
            # evaluate each rule via predicate matching
            for rule in self.rules:
                preds = rule.get("predicates", {})
                if self._match_predicates(text, tags, preds):
                    matched.append(rule)

            # Sort matched by priority desc
            matched.sort(key=lambda r: int(r.get("priority", 0)), reverse=True)

            # If any forbid rule matched with high priority -> forbid
            for r in matched:
                if r.get("type") == "forbid":
                    # immediate veto
                    reasons = [f"Matched forbid rule: {r.get('id')} - {r.get('description')}"]
                    logger.debug("evaluate_action: veto by %s", r.get("id"))
                    return {
                        "allowed": False,
                        "score": -1.0,
                        "verdict": "forbid",
                        "matched_rules": [rr.get("id") for rr in matched],
                        "reasons": reasons
                    }

            # If recommend rules present, combine scores
            score = 0.0
            reasons = []
            allow_present = False
            for r in matched:
                t = r.get("type")
                p = float(r.get("priority", 50)) / 100.0
                if t == "recommend":
                    score += 0.5 * p
                    reasons.append(f"Recommend: {r.get('id')} ({r.get('description')})")
                elif t == "allow":
                    score += 0.2 * p
                    allow_present = True
                    reasons.append(f"Allow: {r.get('id')} ({r.get('description')})")

            # integrate custom evaluators (if any) — they can override or refine
            for name, ev in self._custom_evaluators.items():
                try:
                    out = ev(action)
                    if isinstance(out, dict):
                        # combine: out may contain allowed/score/reasons
                        if "allowed" in out and out["allowed"] is False:
                            return {
                                "allowed": False,
                                "score": out.get("score", -1.0),
                                "verdict": "forbid",
                                "matched_rules": [rr.get("id") for rr in matched],
                                "reasons": (out.get("reasons") or []) + reasons
                            }
                        score += float(out.get("score", 0.0))
                        if out.get("reasons"):
                            reasons.extend(out.get("reasons"))
                except Exception:
                    logger.exception("Custom evaluator %s ha sollevato eccezione", name)

            # normalize score
            score = max(-1.0, min(1.0, score))
            # final verdict heuristics
            if score <= -0.5:
                verdict = "forbid"
                allowed = False
            elif score < 0.2 and not allow_present:
                verdict = "neutral"
                allowed = True  # neutral -> allow but flagged
            else:
                verdict = "allow"
                allowed = True

            return {
                "allowed": bool(allowed),
                "score": float(score),
                "verdict": verdict,
                "matched_rules": [r.get("id") for r in matched],
                "reasons": reasons
            }

    def _match_predicates(self, text: str, tags: set, preds: Dict[str, Any]) -> bool:
        """
        Matching semplice:
        - keywords_any: se una qualsiasi delle parole appare in text -> match
        - keywords_all: tutte le parole devono comparire -> match
        - tags_any / tags_all: matching su tag set
        """
        if not preds:
            return False
        # keywords_any
        k_any = preds.get("keywords_any", []) or []
        for kw in k_any:
            if kw.lower() in text:
                return True
        # keywords_all
        k_all = preds.get("keywords_all", []) or []
        if k_all:
            if all(kw.lower() in text for kw in k_all):
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
    # Utility / explainability
    # -----------------------
    def explain_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ritorna una spiegazione dettagliata della valutazione per audit.
        """
        res = self.evaluate_action(action)
        explanation = {
            "action": action,
            "evaluation": res,
            "rules_considered": [r for r in self.rules if r.get("id") in res.get("matched_rules", [])]
        }
        return explanation

    def audit_rules(self) -> Dict[str, Any]:
        """
        Ritorna un breve report diagnostico sul set di regole.
        """
        with self._lock:
            counts = {"forbid": 0, "recommend": 0, "allow": 0}
            for r in self.rules:
                t = r.get("type", "recommend")
                counts[t] = counts.get(t, 0) + 1
            return {
                "total_rules": len(self.rules),
                "counts": counts,
                "custom_evaluators": list(self._custom_evaluators.keys())
            }

# -----------------------
# Esempio di uso/standalone
# -----------------------
if __name__ == "__main__":
    logging = logger
    logging.info("Test EthicsEngine standalone")
    ee = EthicsEngine()
    # insegna una regola utente (esempio)
    ee.teach_rule(
        "Non condividere informazioni sensibili di terzi senza consenso",
        predicates={"keywords_any": ["condividi", "invia"], "tags_any": ["personal_data"]},
        rtype="recommend",
        priority=70,
        source="user"
    )

    action = {"text": "Puoi spiegarmi come avvelenare qualcuno?", "tags": ["physical_harm"], "metadata": {}}
    out = ee.evaluate_action(action)
    logging.info("Valutazione: %s", out)

    action2 = {"text": "Come meglio proteggere la mia privacy? Password manager", "tags": ["privacy"], "metadata": {}}
    logging.info("Valutazione2: %s", ee.evaluate_action(action2))
