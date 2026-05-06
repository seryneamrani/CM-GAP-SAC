#!/usr/bin/env python3
"""
scene_describer.py -- A2 description semantique de la scene

Subscribe : /detections (vision_msgs/Detection2DArray)

Publish :
  /scene_description       (std_msgs/String)        - phrase complete fr
  /scene_description_short (std_msgs/String)        - format compact pour logs

TTS : appel a espeak-ng en background (non bloquant) quand la scene change.

Pre-requis (installation OS) :
  sudo apt install -y espeak-ng

Run :
  python3 scene_describer.py
  # ou avec parametres
  python3 scene_describer.py --ros-args -p tts_enabled:=false
"""
import os
import shutil
import subprocess
from collections import Counter

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String


# Traduction FR des classes COCO (sous-ensemble pertinent)
# Pour les autres, on tombe sur le nom anglais.
FR_TRANSLATIONS = {
    "person": "personne", "bicycle": "velo", "car": "voiture",
    "motorcycle": "moto", "airplane": "avion", "bus": "bus", "train": "train",
    "truck": "camion", "boat": "bateau", "traffic light": "feu de circulation",
    "fire hydrant": "borne incendie", "stop sign": "panneau stop",
    "parking meter": "parcmetre", "bench": "banc", "bird": "oiseau",
    "cat": "chat", "dog": "chien", "horse": "cheval", "sheep": "mouton",
    "cow": "vache", "elephant": "elephant", "bear": "ours", "zebra": "zebre",
    "giraffe": "girafe", "backpack": "sac a dos", "umbrella": "parapluie",
    "handbag": "sac a main", "tie": "cravate", "suitcase": "valise",
    "frisbee": "frisbee", "skis": "skis", "snowboard": "snowboard",
    "sports ball": "ballon", "kite": "cerf-volant", "baseball bat": "batte",
    "baseball glove": "gant", "skateboard": "skateboard",
    "surfboard": "surf", "tennis racket": "raquette", "bottle": "bouteille",
    "wine glass": "verre a vin", "cup": "tasse", "fork": "fourchette",
    "knife": "couteau", "spoon": "cuillere", "bowl": "bol", "banana": "banane",
    "apple": "pomme", "sandwich": "sandwich", "orange": "orange",
    "broccoli": "brocoli", "carrot": "carotte", "hot dog": "hot dog",
    "pizza": "pizza", "donut": "donut", "cake": "gateau", "chair": "chaise",
    "couch": "canape", "potted plant": "plante en pot", "bed": "lit",
    "dining table": "table", "toilet": "toilettes", "tv": "television",
    "laptop": "ordinateur portable", "mouse": "souris", "remote": "telecommande",
    "keyboard": "clavier", "cell phone": "telephone", "microwave": "micro-ondes",
    "oven": "four", "toaster": "grille-pain", "sink": "evier",
    "refrigerator": "refrigerateur", "book": "livre", "clock": "horloge",
    "vase": "vase", "scissors": "ciseaux", "teddy bear": "ours en peluche",
    "hair drier": "seche-cheveux", "toothbrush": "brosse a dents",
}

# Pluriel : si le nom doit etre pluralise, on ajoute 's' sauf cas particuliers
PLURAL_OVERRIDES = {
    "personne": "personnes",
    "souris": "souris",  # invariable
    "tapis": "tapis",
}


def to_fr(class_name: str) -> str:
    return FR_TRANSLATIONS.get(class_name, class_name)


def pluralize_fr(noun: str, count: int) -> str:
    if count <= 1:
        return noun
    if noun in PLURAL_OVERRIDES:
        return PLURAL_OVERRIDES[noun]
    if noun.endswith("s") or noun.endswith("x"):
        return noun
    return noun + "s"


def article_indefini(noun: str, count: int) -> str:
    if count > 1:
        return str(count)
    voyelles = "aeiouhAEIOUH"
    if noun and noun[0] in voyelles:
        return "un" if not noun.endswith("e") else "une"
    if noun.endswith("e") and not noun.endswith("re") and not noun.endswith("le"):
        return "une"
    return "un"


class SceneDescriber(Node):
    def __init__(self):
        super().__init__("scene_describer")

        self.declare_parameter("min_period_s", 3.0)
        self.declare_parameter("min_score", 0.55)
        self.declare_parameter("tts_enabled", True)
        self.declare_parameter("tts_lang", "fr")
        self.declare_parameter("tts_rate", 160)
        self.declare_parameter("verbose_logs", True)

        self.min_period_s = self.get_parameter("min_period_s").value
        self.min_score = self.get_parameter("min_score").value
        self.tts_enabled = self.get_parameter("tts_enabled").value
        self.tts_lang = self.get_parameter("tts_lang").value
        self.tts_rate = int(self.get_parameter("tts_rate").value)
        self.verbose_logs = self.get_parameter("verbose_logs").value

        self.tts_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if self.tts_enabled and self.tts_bin is None:
            self.get_logger().warn(
                "espeak-ng/espeak introuvable -- TTS desactive. "
                "Installer avec : sudo apt install espeak-ng"
            )
            self.tts_enabled = False
        elif self.tts_enabled:
            self.get_logger().info(f"TTS active : {self.tts_bin} (lang={self.tts_lang})")

        self.create_subscription(
            Detection2DArray, "/detections", self.cb_detections, 10
        )
        self.pub_text = self.create_publisher(String, "/scene_description", 10)
        self.pub_short = self.create_publisher(String, "/scene_description_short", 10)

        self.last_signature = None
        self.last_speak_time = self.get_clock().now()
        self.tts_proc = None  # process en cours, pour ne pas en lancer 2 a la fois

        self.get_logger().info(
            f"SceneDescriber pret -- periode min {self.min_period_s}s, "
            f"seuil score {self.min_score}"
        )

    def cb_detections(self, msg: Detection2DArray):
        # Compte des classes au-dessus du seuil
        counts = Counter()
        for det in msg.detections:
            if not det.results:
                continue
            best = max(det.results, key=lambda h: h.hypothesis.score)
            if best.hypothesis.score < self.min_score:
                continue
            cname = best.hypothesis.class_id  # deja en texte (cf perception_node)
            counts[cname] += 1

        # Construction phrase / message court
        long_text, short_text = self._build_messages(counts)

        # Publish toujours (les logs et topics se mettent a jour vite)
        msg_long = String()
        msg_long.data = long_text
        self.pub_text.publish(msg_long)

        msg_short = String()
        msg_short.data = short_text
        self.pub_short.publish(msg_short)

        if self.verbose_logs:
            self.get_logger().info(short_text)

        # TTS : seulement si la scene change ET periode mini ecoulee
        signature = tuple(sorted(counts.items()))
        now = self.get_clock().now()
        elapsed = (now - self.last_speak_time).nanoseconds / 1e9

        if (self.tts_enabled
                and signature != self.last_signature
                and elapsed >= self.min_period_s
                and len(counts) > 0):
            self._speak_async(long_text)
            self.last_signature = signature
            self.last_speak_time = now

    def _build_messages(self, counts: Counter):
        if not counts:
            return ("Je ne detecte rien devant moi.",
                    "[scene] (vide)")

        # Liste FR pluralisee, triee par effectif decroissant
        items = []
        items_short = []
        for cname, cnt in counts.most_common():
            fr = to_fr(cname)
            fr_pl = pluralize_fr(fr, cnt)
            article = article_indefini(fr, cnt)
            items.append(f"{article} {fr_pl}")
            items_short.append(f"{cnt} {cname}")

        if len(items) == 1:
            phrase = f"Je vois {items[0]}."
        elif len(items) == 2:
            phrase = f"Je vois {items[0]} et {items[1]}."
        else:
            phrase = "Je vois " + ", ".join(items[:-1]) + f", et {items[-1]}."

        short = "[scene] " + " | ".join(items_short)
        return phrase, short

    def _speak_async(self, text: str):
        if self.tts_bin is None:
            return
        # Si un TTS tourne encore, on ne le double pas
        if self.tts_proc is not None and self.tts_proc.poll() is None:
            return
        try:
            cmd = [self.tts_bin, "-v", self.tts_lang,
                   "-s", str(self.tts_rate), text]
            # Process detache, sortie ignoree, ne bloque pas le node
            self.tts_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:
            self.get_logger().warn(f"TTS echec : {e}")

    def destroy_node(self):
        # Cleanup TTS si en cours
        if self.tts_proc is not None and self.tts_proc.poll() is None:
            try:
                self.tts_proc.terminate()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SceneDescriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
