#!/usr/bin/env python3
"""
Nettoie un fichier de résultats JSONL :
  - retire le champ `episode_seed` de chaque ligne
  - extrait toutes les lignes où outcome.collision_pedestrian == True

Usage:
    python clean_results.py results_backup_20260727_1157.jsonl
    python clean_results.py results_backup_20260727_1157.jsonl -o out_dir/
"""
import argparse
import json
import sys
from pathlib import Path


def process(input_path: Path, output_dir: Path | None = None) -> None:
    if not input_path.is_file():
        sys.exit(f"[erreur] fichier introuvable : {input_path}")

    out_dir = output_dir if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    cleaned_path = out_dir / f"{stem}__no_seeds.jsonl"
    collisions_path = out_dir / f"{stem}__collisions_pedestrian.jsonl"

    n_total = 0
    n_bad = 0
    n_seeds_removed = 0
    n_collisions = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         cleaned_path.open("w", encoding="utf-8") as fout_clean, \
         collisions_path.open("w", encoding="utf-8") as fout_coll:

        for line_num, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            n_total += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                n_bad += 1
                print(f"[warn] ligne {line_num} invalide ({e}) — ignorée",
                      file=sys.stderr)
                continue

            # 1) collision pedestrian → on écrit AVANT de retirer quoi que ce soit
            #    (on garde le seed dans le fichier collisions pour pouvoir rejouer l'épisode)
            outcome = rec.get("outcome") or {}
            if outcome.get("collision_pedestrian") is True:
                fout_coll.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_collisions += 1

            # 2) retirer le seed pour le fichier nettoyé
            if "episode_seed" in rec:
                del rec["episode_seed"]
                n_seeds_removed += 1

            fout_clean.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ok] lignes lues        : {n_total}")
    print(f"[ok] lignes invalides   : {n_bad}")
    print(f"[ok] seeds retirés      : {n_seeds_removed}")
    print(f"[ok] collisions piétons : {n_collisions}")
    print(f"[ok] → {cleaned_path}")
    print(f"[ok] → {collisions_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, nargs="?",
                   default=Path("results_backup_20260727_1157.jsonl"),
                   help="fichier .jsonl d'entrée "
                        "(défaut: results_backup_20260727_1157.jsonl)")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="dossier de sortie (défaut: même dossier que l'entrée)")
    args = p.parse_args()
    process(args.input, args.output_dir)


if __name__ == "__main__":
    main()