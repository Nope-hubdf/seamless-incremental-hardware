# motor_learning.py
"""
Motor learning and robot interface for Nova.

Responsibilities:
- Record and store demonstrations (trajectories, sensor frames, annotations).
- Provide dataset CRUD (save/load/list).
- Provide a trainable imitation model interface (stub) and policy eval.
- Provide simulation/execution API and safe actuator interface (RobotAdapter).
- Integrate with internal_state.yaml and MemoryTimeline if available.
- Provide simple curriculum / incremental learning hooks.

Note: This module provides high-level methods and safe defaults.
Actual robot actuation requires implementing RobotAdapter for your hardware
(e.g. ROS, serial motor controller, custom SDK). Training models may require
external ML libs — the training functions here are stubs you can replace
with real training pipelines (Torch, JAX, etc.).
"""

import os
import time
import yaml
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

STATE_FILE = "internal_state.yaml"
DATA_DIR = "motor_data"
MODELS_DIR = "motor_models"

# Try to import optional connectors
try:
    from memory_timeline import MemoryTimeline
except Exception:
    MemoryTimeline = None
    logger.debug("MemoryTimeline non disponibile; le interazioni motrici non verranno loggate nella timeline.")

# Ensure directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------
# Robot adapter base class
# ---------------------------
class RobotAdapter:
    """
    Abstract adapter to control a physical robot or simulator.

    Implement these methods for your specific robot (ROS, serial, SDK).
    - connect(): establish connection
    - disconnect(): clean up
    - move_joint_positions(positions: List[float], duration: float): move
    - set_cartesian_pose(pose: Dict[str,float], duration: float): move in Cartesian
    - stop(): emergency stop
    - get_state(): return sensors/joint angles/position (dict)
    """

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def move_joint_positions(self, positions: List[float], duration: float) -> None:
        raise NotImplementedError

    def set_cartesian_pose(self, pose: Dict[str, float], duration: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

# ---------------------------
# Utility helpers
# ---------------------------
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            s = yaml.safe_load(f) or {}
            return s
    return {}

def save_state(state: Dict[str, Any]):
    with open(STATE_FILE, "w") as f:
        yaml.safe_dump(state, f)
    logger.debug("internal_state.yaml aggiornato dal motor_learning")

# ---------------------------
# Data structures
# ---------------------------
class Trajectory:
    """
    Simple representation of a demonstration trajectory.
    - frames: list of frames (each frame: dict with 'timestamp', 'joints' or 'pose', 'sensors' optional)
    - meta: freeform metadata (task name, teacher id, annotation)
    """
    def __init__(self, frames: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None):
        self.frames = frames
        self.meta = meta or {}
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self):
        return {"frames": self.frames, "meta": self.meta, "created_at": self.created_at}

    @staticmethod
    def from_dict(d: Dict[str, Any]):
        t = Trajectory(d.get("frames", []), d.get("meta", {}))
        t.created_at = d.get("created_at", datetime.utcnow().isoformat())
        return t

# ---------------------------
# MotorLearning class
# ---------------------------
class MotorLearning:
    def __init__(self, robot_adapter: Optional[RobotAdapter] = None, core_ref: Optional[Any] = None):
        """
        robot_adapter: implementazione concreta di RobotAdapter per il tuo hardware/sim.
        core_ref: opzionale riferimento a NovaCore per callback (es. per logging o timeline).
        """
        self.robot = robot_adapter
        self.core = core_ref
        self.timeline = MemoryTimeline() if MemoryTimeline else None
        self.state = load_state()
        if "motor" not in self.state:
            self.state["motor"] = {"demonstrations": [], "models": {}}
            save_state(self.state)
        logger.info("MotorLearning inizializzato.")

    # -----------------------
    # Demonstration storage
    # -----------------------
    def save_demonstration(self, traj: Trajectory, name: Optional[str] = None) -> str:
        """
        Save trajectory to disk and register to state.
        Returns filename key.
        """
        key = name or f"traj_{int(time.time())}"
        path = os.path.join(DATA_DIR, f"{key}.json")
        with open(path, "w") as f:
            json.dump(traj.to_dict(), f)
        # register
        self.state["motor"]["demonstrations"].append({"key": key, "path": path, "meta": traj.meta, "created_at": traj.created_at})
        save_state(self.state)
        logger.info(f"Dimostrazione salvata: {path}")
        # add to timeline
        try:
            if self.timeline:
                self.timeline.add_experience({"type": "motor_demo", "key": key, "path": path, "meta": traj.meta}, category="motor", importance=3)
        except Exception:
            logger.exception("Errore salvataggio demo in timeline")
        return key

    def load_demonstration(self, key: str) -> Optional[Trajectory]:
        recs = [r for r in self.state["motor"]["demonstrations"] if r["key"] == key]
        if not recs:
            logger.warning(f"Demo non trovata: {key}")
            return None
        path = recs[0]["path"]
        with open(path, "r") as f:
            data = json.load(f)
        return Trajectory.from_dict(data)

    def list_demonstrations(self) -> List[Dict[str, Any]]:
        return self.state["motor"]["demonstrations"]

    def delete_demonstration(self, key: str) -> bool:
        recs = [r for r in self.state["motor"]["demonstrations"] if r["key"] == key]
        if not recs:
            return False
        path = recs[0]["path"]
        try:
            os.remove(path)
        except Exception:
            logger.exception("Errore rimozione file demo")
        self.state["motor"]["demonstrations"] = [r for r in self.state["motor"]["demonstrations"] if r["key"] != key]
        save_state(self.state)
        return True

    # -----------------------
    # Model persistence (stub)
    # -----------------------
    def save_model(self, model_obj: Any, name: str) -> str:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model_obj, f)
        self.state["motor"]["models"][name] = {"path": path, "saved_at": datetime.utcnow().isoformat()}
        save_state(self.state)
        logger.info(f"Modello motor salvato: {path}")
        return path

    def load_model(self, name: str) -> Optional[Any]:
        rec = self.state["motor"]["models"].get(name)
        if not rec:
            logger.warning(f"Modello non trovato: {name}")
            return None
        path = rec["path"]
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_models(self) -> Dict[str, Any]:
        return self.state["motor"]["models"]

    # -----------------------
    # Training / Imitation (stubs, replaceable)
    # -----------------------
    def train_imitation_model(self, demo_keys: List[str], model_name: str, epochs: int = 10, verbose: bool = False) -> str:
        """
        Train an imitation model from provided demonstration keys.
        THIS IS A STUB — replace with your training pipeline (PyTorch/TensorFlow).
        For now it collects trajectories and creates a simple 'average policy' placeholder.
        """
        trajectories = []
        for k in demo_keys:
            t = self.load_demonstration(k)
            if t:
                trajectories.append(t)

        if not trajectories:
            raise ValueError("Nessuna demo valida fornita")

        # Simple placeholder model: compute average joint positions per time-index (very naive)
        # In real use: train seq2seq, behavior cloning, or RLfD model.
        avg_policy = {"type": "avg_placeholder", "trained_on": demo_keys, "created_at": datetime.utcnow().isoformat()}
        # store
        model_path = self.save_model(avg_policy, model_name)
        logger.info(f"Modello di imitazione (placeholder) creato: {model_path}")

        # timeline
        if self.timeline:
            self.timeline.add_experience({"type": "model_train", "model": model_name, "demos": demo_keys}, category="motor", importance=2)

        return model_path

    # -----------------------
    # Policy evaluation / simulation
    # -----------------------
    def evaluate_policy(self, model_name: str, n_episodes: int = 3) -> Dict[str, Any]:
        """
        Evaluate a stored policy in simulation (stub).
        Returns metrics dict.
        """
        model = self.load_model(model_name)
        if not model:
            raise ValueError("Model non trovato")

        # Placeholder evaluation
        metrics = {"model": model_name, "episodes": n_episodes, "success_rate": 0.0, "details": []}
        for ep in range(n_episodes):
            # In real simulation: run in pybullet / gazebo and compute success
            metrics["details"].append({"episode": ep, "result": "simulated_ok"})
        metrics["success_rate"] = 1.0
        logger.info(f"Valutazione modello {model_name}: {metrics}")
        return metrics

    # -----------------------
    # Execution & Safety
    # -----------------------
    def safety_check(self, planned_action: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Implement safety constraints here (joint limits, forbidden zones, max speed).
        Returns (ok, message).
        """
        # Placeholder: always allow but log
        # User should extend to check joint limits, workspace, etc.
        logger.debug(f"Safety check per action: {planned_action}")
        return True, "OK (placeholder)"

    def execute_action(self, action: Dict[str, Any], blocking: bool = True) -> bool:
        """
        Execute a single action dict:
        action example:
        {"type": "joint_positions", "positions": [..], "duration": 1.5}
        or {"type": "cartesian", "pose": {"x":..,"y":..,"z":..,"rx":..,"ry":..,"rz":..}, "duration": 2.0}
        """
        ok, msg = self.safety_check(action)
        if not ok:
            logger.warning(f"Azione bloccata per safety: {msg}")
            return False

        if not self.robot:
            logger.warning("Nessun RobotAdapter configurato: esecuzione simulata (no-op).")
            # Optionally call timeline to record simulated execution
            if self.timeline:
                self.timeline.add_experience({"type": "motor_exec_sim", "action": action}, category="motor", importance=1)
            return True

        try:
            if action["type"] == "joint_positions":
                positions = action["positions"]
                duration = action.get("duration", 1.0)
                self.robot.move_joint_positions(positions, duration)
            elif action["type"] == "cartesian":
                pose = action["pose"]
                duration = action.get("duration", 1.0)
                self.robot.set_cartesian_pose(pose, duration)
            else:
                logger.warning("Tipo azione motoria sconosciuto: %s", action.get("type"))
                return False

            # Optionally log executed action
            if self.timeline:
                self.timeline.add_experience({"type": "motor_exec", "action": action}, category="motor", importance=2)
            return True
        except Exception:
            logger.exception("Errore durante execute_action")
            return False

    # -----------------------
    # Demonstration helpers
    # -----------------------
    def record_demonstration_from_robot(self, duration: float = 5.0, rate_hz: float = 20.0, name: Optional[str] = None) -> Optional[str]:
        """
        Record a demonstration by sampling robot.get_state() over time.
        Requires robot adapter providing get_state().
        """
        if not self.robot:
            logger.warning("record_demonstration_from_robot richiede RobotAdapter.")
            return None

        frames = []
        n_samples = int(duration * rate_hz)
        interval = 1.0 / rate_hz
        logger.info(f"Inizio registrazione demo da robot: duration={duration}s, samples={n_samples}")
        try:
            for i in range(n_samples):
                state = self.robot.get_state()
                frames.append({"timestamp": time.time(), "state": state})
                time.sleep(interval)
        except Exception:
            logger.exception("Errore durante registrazione demo dal robot")
            return None

        traj = Trajectory(frames, meta={"source": "robot_record", "duration": duration})
        return self.save_demonstration(traj, name)

    def record_demonstration_from_frames(self, frames: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None, name: Optional[str] = None) -> str:
        """
        Create a trajectory from externally supplied frames (e.g., annotated video / vision pipeline).
        """
        traj = Trajectory(frames, meta=meta)
        return self.save_demonstration(traj, name)

    # -----------------------
    # Teaching via remote (integration point)
    # -----------------------
    def teach_via_remote(self, user_id: str, frames: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Called by remote_comm when a user uploads a recording or annotated frames.
        Stores the demo and logs to timeline.
        """
        key = self.record_demonstration_from_frames(frames, meta=meta)
        logger.info(f"Insegnamento remoto registrato da {user_id}: {key}")
        if self.timeline:
            self.timeline.add_experience({"type": "teach_remote", "user": user_id, "demo_key": key}, category="motor", importance=3)
        return key

    # -----------------------
    # Integration helpers
    # -----------------------
    def integrate_with_core(self, core_obj: Any):
        """
        Attach to NovaCore or other orchestrator to receive callbacks.
        Example: core_obj can call motor.learn.teach_via_remote(...) on incoming media.
        """
        self.core = core_obj
        logger.info("MotorLearning integrato con core.")

    # -----------------------
    # Convenience: simulate policy execution (no robot)
    # -----------------------
    def simulate_execution(self, model_name: str, speed: float = 1.0) -> bool:
        """
        Simulate running a stored model's policy. Useful to preview before physical run.
        """
        model = self.load_model(model_name)
        if not model:
            logger.warning("Modello per simulazione non trovato")
            return False
        logger.info(f"Simulazione modello {model_name} (stub).")
        # Add timeline entry
        if self.timeline:
            self.timeline.add_experience({"type": "simulate_model", "model": model_name}, category="motor", importance=1)
        return True

# ---------------------------
# Example usage (standalone)
# ---------------------------
if __name__ == "__main__":
    logger.info("Esecuzione test motor_learning (standalone).")
    ml = MotorLearning(robot_adapter=None, core_ref=None)

    # Create a tiny fake demo
    frames = []
    for i in range(10):
        frames.append({"timestamp": time.time(), "joints": [0.1 * i, 0.2 * i], "sensors": {"touch": False}})
        time.sleep(0.01)
    key = ml.record_demonstration_from_frames(frames, meta={"task": "wave", "teacher": "local_test"})
    logger.info(f"Demo salvata con key: {key}")
    model_path = ml.train_imitation_model([key], model_name="wave_avg_v1")
    logger.info(f"Model creato: {model_path}")
    ml.simulate_execution("wave_avg_v1")
