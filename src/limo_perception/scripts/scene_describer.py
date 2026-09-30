#!/usr/bin/env python3
"""
scene_describer_v2.py -- Description semantique enrichie avec etats.

Difference V1 : exploite /track_states pour annoncer l'etat de chaque
personne. Exemple :
    "Je vois deux personnes statiques, une personne en mouvement a 0.6
     metres par seconde, et un enfant en mouvement."

Subscribe :
    /detections      (vision_msgs/Detection2DArray) -- objets non-humains
    /track_states    (std_msgs/String JSON)          -- etats des humains

Publish :
    /scene_description       (std_msgs/String)
    /scene_description_short (std_msgs/String)

TTS :
    espeak-ng en sous-processus detache, fallback aplay ALSA si
    PulseAudio absent.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String


FR_TRANSLATIONS = {
    "person": "personne", "chair": "chaise", "couch": "canape",
    "potted plant": "plante en pot", "bed": "lit", "dining table": "table",
    "tv": "television", "laptop": "ordinateur portable",
    "cell phone": "telephone", "book": "livre", "clock": "horloge",
    "vase": "vase", "bottle": "bouteille", "cup": "tasse",
    "refrigerator": "refrigerateur", "microwave": "micro-ondes",
    "oven": "four", "sink": "evier", "scissors": "ciseaux",
    "backpack": "sac a dos", "handbag": "sac a main", "umbrella": "parapluie",
    "suitcase": "valise", "bicycle": "velo", "motorcycle": "moto",
    "car": "voiture", "bus": "bus", "truck": "camion", "cat": "chat",
    "dog": "chien", "bird": "oiseau",
}

PLURAL_OVERRIDES = {"personne": "personnes", "souris": "souris"}

# Etats traduits
STATE_FR = {
    "static": "statique",
    "probably_static": "probablement statique",
    "starts_moving": "qui commence a bouger",
    "dynamic": "en mouvement",
}


def to_fr(class_name: str) -> str:
    return FR_TRANSLATIONS.get(class_name, class_name)


def pluralize(noun: str, n: int) -> str:
    if n <= 1:
        return noun
    if noun in PLURAL_OVERRIDES:
        return PLURAL_OVERRIDES[noun]
    if noun.endswith("s") or noun.endswith("x"):
        return noun
    return noun + "s"


def number_word(n: int) -> str:
    return {1: "une", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq",
            6: "six", 7: "sept", 8: "huit", 9: "neuf", 10: "dix"}.get(
        n, str(n))


class SceneDescriberV2(Node):
    def __init__(self):
        super().__init__("scene_describer")

        self.declare_parameter("min_score", 0.55)
        self.declare_parameter("min_period_s", 4.0)
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("alsa_device", "plughw:CARD=PCH,DEV=0")

        self.min_score = self.get_parameter("min_score").value
        self.min_period = float(self.get_parameter("min_period_s").value)
        self.tts_enabled = bool(self.get_parameter("tts_enabled").value)
        self.alsa_device = self.get_parameter("alsa_device").value

        # Detection si pulseaudio actif, sinon ALSA
        self.use_alsa = self._detect_alsa_only()
        if self.tts_enabled:
            mode = "ALSA direct" if self.use_alsa else "PulseAudio"
            self.get_logger().info(f"TTS active : {mode}")

        # Etat
        self.last_speech_t = 0.0
        self.last_signature = ""
        self.tts_proc = None
        self.last_track_states = []   # depuis /track_states
        self.last_objects = Counter()  # classes non-humaines

        self.create_subscription(
            Detection2DArray, "/detections", self._cb_detections, 10)
        self.create_subscription(
            String, "/track_states", self._cb_track_states, 10)

        self.pub_full = self.create_publisher(String, "/scene_description", 10)
        self.pub_short = self.create_publisher(
            String, "/scene_description_short", 10)

        self.create_timer(1.0, self._publish)

    def _detect_alsa_only(self) -> bool:
        try:
            r = subprocess.run(
                ["pgrep", "-af", "pulseaudio|pipewire"],
                capture_output=True, text=True, timeout=2)
            return r.returncode != 0 or not r.stdout.strip()
        except Exception:
            return True

    def _cb_track_states(self, msg: String):
        try:
            self.last_track_states = json.loads(msg.data)
        except Exception:
            self.last_track_states = []

    def _cb_detections(self, msg: Detection2DArray):
        counts = Counter()
        for det in msg.detections:
            if not det.results:
                continue
            best = max(det.results, key=lambda h: h.hypothesis.score)
            if best.hypothesis.score < self.min_score:
                continue
            cls = best.hypothesis.class_id
            # On exclut les humains -- ils sont decrits via /track_states
            if cls in ("person", "cat", "dog"):
                continue
            counts[cls] += 1
        self.last_objects = counts

    def _build_sentence(self) -> str:
        parts = []
        # 1. Decrit les humains par etat
        if self.last_track_states:
            by_state = {}
            for t in self.last_track_states:
                state = t.get("state", "static")
                cls = t.get("class", "person")
                by_state.setdefault((cls, state), []).append(t)
            for (cls, state), lst in by_state.items():
                n = len(lst)
                noun_fr = pluralize(to_fr(cls), n)
                state_fr = STATE_FR.get(state, state)
                # Pour les dynamiques on ajoute la vitesse moyenne
                if state == "dynamic" and lst:
                    speeds = [x.get("speed", 0.0) for x in lst]
                    v_mean = sum(speeds) / len(speeds)
                    parts.append(
                        f"{number_word(n)} {noun_fr} {state_fr} "
                        f"a {v_mean:.1f} metres par seconde"
                    )
                else:
                    parts.append(f"{number_word(n)} {noun_fr} {state_fr}")

        # 2. Decrit les autres objets (mobilier, plantes, etc.)
        for cls, n in self.last_objects.most_common():
            noun_fr = pluralize(to_fr(cls), n)
            parts.append(f"{number_word(n)} {noun_fr}")

        if not parts:
            return ""

        if len(parts) == 1:
            return f"Je vois {parts[0]}."
        return f"Je vois {', '.join(parts[:-1])} et {parts[-1]}."

    def _build_short(self) -> str:
        parts = ["[scene]"]
        # Humains par etat
        for t in self.last_track_states:
            state = t.get("state", "?")
            cls = t.get("class", "?")
            tid = t.get("id", "?")
            spd = t.get("speed", 0.0)
            parts.append(f"{cls}#{tid}:{state}({spd:.2f})")
        # Autres
        for cls, n in self.last_objects.most_common():
            parts.append(f"{cls}:{n}")
        return " | ".join(parts)

    def _publish(self):
        sentence = self._build_sentence()
        short = self._build_short()

        self.pub_short.publish(String(data=short))
        if not sentence:
            return
        self.pub_full.publish(String(data=sentence))

        # TTS avec rate-limit + signature
        signature = sentence  # phrase entiere
        now = time.time()
        if (signature != self.last_signature
                and now - self.last_speech_t >= self.min_period
                and self.tts_enabled):
            self.last_signature = signature
            self.last_speech_t = now
            self._speak(sentence)

    def _speak(self, text: str):
        # Si tts_proc encore actif, on skip
        if self.tts_proc is not None and self.tts_proc.poll() is None:
            return
        try:
            if self.use_alsa:
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False).name
                # 1. Genere wav
                subprocess.run(
                    ["espeak-ng", "-v", "fr", "-w", tmp, text],
                    check=False, timeout=5,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # 2. aplay non bloquant
                self.tts_proc = subprocess.Popen(
                    ["aplay", "-q", "-D", self.alsa_device, tmp],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                self.tts_proc = subprocess.Popen(
                    ["espeak-ng", "-v", "fr", text],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.get_logger().warn(f"TTS echec : {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SceneDescriberV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
