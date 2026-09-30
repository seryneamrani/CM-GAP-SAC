#!/usr/bin/env python3
"""
Test YOLOv8-nano standalone — validation GPU + perf avant intégration ROS.
Usage : python3 test_yolo.py
"""
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

print("=" * 50)
print("  TEST YOLOv8-nano")
print("=" * 50)

# 1. Vérif GPU
print(f"\n[GPU] CUDA disponible : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device         : {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("[GPU] FALLBACK CPU — perfs dégradées attendues")
    device = "cpu"

# 2. Chargement modèle
print("\n[MODEL] Chargement yolov8n.pt...")
model = YOLO("yolov8n.pt")
model.to(device)
print(f"[MODEL] Chargé sur {device}")

# 3. Image test : on génère une image bruitée 640x480
#    (YOLO va probablement ne rien détecter, mais on mesure la perf)
print("\n[BENCH] Inference sur 30 images aléatoires...")
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Warmup
for _ in range(3):
    _ = model(img, verbose=False)

# Bench
times = []
for i in range(30):
    t0 = time.time()
    results = model(img, verbose=False)
    t1 = time.time()
    times.append((t1 - t0) * 1000)

avg = sum(times) / len(times)
fps = 1000.0 / avg

print(f"[BENCH] Latence moyenne : {avg:.1f} ms")
print(f"[BENCH] FPS estimés     : {fps:.1f}")

# 4. Test sur image réelle si une webcam est dispo (optionnel)
print("\n[REAL] Test détection sur image COCO de référence...")
# On utilise une URL d'image COCO connue (chargée via cv2 si possible)
# Sinon on skippe
try:
    # Image bus.jpg fournie avec ultralytics
    from ultralytics.utils import ASSETS
    bus_img = cv2.imread(str(ASSETS / "bus.jpg"))
    if bus_img is not None:
        results = model(bus_img, verbose=False)
        n_det = len(results[0].boxes)
        classes = results[0].boxes.cls.cpu().numpy().astype(int) if n_det > 0 else []
        names = [model.names[c] for c in classes]
        print(f"[REAL] Detections : {n_det}")
        print(f"[REAL] Classes    : {names}")
    else:
        print("[REAL] bus.jpg non trouvée, skip")
except Exception as e:
    print(f"[REAL] Skip ({e})")

# 5. Verdict
print("\n" + "=" * 50)
if device == "cuda" and fps >= 30:
    print("  ✅ PRÊT POUR A2 — GPU + FPS suffisants")
elif device == "cuda":
    print("  ⚠️  GPU OK mais FPS bas — vérifier la charge GPU")
else:
    print("  ❌ CPU only — A2 sera lent (acceptable mais sous-optimal)")
print("=" * 50)
