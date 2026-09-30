"""
Fusionne plusieurs CSV d'épisodes (RL) en gérant :
- l'harmonisation des colonnes (schémas différents entre fichiers)
- le chevauchement de steps entre runs (ex: reprise depuis checkpoint)
- le filtrage jusqu'à un step max
- la renumérotation des épisodes

Usage: python merge_episodes.py
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1) Liste des fichiers, DANS L'ORDRE CHRONOLOGIQUE / DE PRIORITÉ.
#    En cas de chevauchement de "step" entre deux fichiers, c'est le
#    fichier le PLUS BAS dans cette liste (donc le plus récent) qui
#    l'emporte sur la zone commune.
# ------------------------------------------------------------------
files_in_priority_order = [
    "runs/cm_gap_sac/20260629-nav2-wide-doors/episodes_v1_pre_logging_fix.csv",
    "runs/cm_gap_sac/20260629-nav2-wide-doors/episodes_run_v1_abandoned.csv",
    # "runs/cm_gap_sac/20260629-nav2-wide-doors/episodes_v1_TROISIEME.csv",  # <-- ajoute ton 3e fichier ici
]

STEP_MAX = 1_156_000
OUTPUT_PATH = "runs/cm_gap_sac/20260629-nav2-wide-doors/episodes_combined_clean.csv"


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source_file"] = Path(path).name
    return df


def main():
    combined = pd.DataFrame()

    for f in files_in_priority_order:
        df = load_csv(f)
        print(f"[lecture] {f} -> {len(df)} lignes, "
              f"steps [{df['step'].min()} - {df['step'].max()}], "
              f"colonnes: {list(df.columns)}")

        if not combined.empty:
            lo, hi = df["step"].min(), df["step"].max()
            before = len(combined)
            # On retire de ce qu'on a déjà accumulé toute la zone de steps
            # que ce nouveau fichier couvre (=> le nouveau fichier écrase
            # les anciens sur le chevauchement).
            overlap_mask = combined["step"].between(lo, hi)
            n_overlap = overlap_mask.sum()
            if n_overlap:
                print(f"  -> {n_overlap} lignes précédentes dans la zone de "
                      f"chevauchement [{lo} - {hi}] seront écrasées par ce fichier")
            combined = combined[~overlap_mask]

        # concat avec union des colonnes (colonnes manquantes -> NaN)
        combined = pd.concat([combined, df], ignore_index=True, sort=False)

    # ------------------------------------------------------------------
    # 2) Tri par step, filtre jusqu'à STEP_MAX
    # ------------------------------------------------------------------
    combined = combined.sort_values("step").reset_index(drop=True)

    before = len(combined)
    combined = combined[combined["step"] <= STEP_MAX].reset_index(drop=True)
    print(f"\nFiltre step <= {STEP_MAX}: {before} -> {len(combined)} lignes")

    # ------------------------------------------------------------------
    # 3) Nettoyage
    # ------------------------------------------------------------------
    # Sécurité : suppression de doublons exacts éventuels (même step ET même
    # source), au cas où un même fichier serait listé/concaténé deux fois
    combined = combined.drop_duplicates(subset=["step", "__source_file"], keep="first")

    # Renumérotation propre des épisodes dans l'ordre chronologique
    combined["episode"] = range(1, len(combined) + 1)

    # On garde une colonne indiquant la provenance (utile pour debug),
    # à retirer si tu veux un CSV strictement identique au format d'origine :
    # combined = combined.drop(columns=["__source_file"])

    # ------------------------------------------------------------------
    # 4) Export
    # ------------------------------------------------------------------
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{len(combined)} épisodes -> {OUTPUT_PATH}")
    print(f"steps: {combined['step'].min()} -> {combined['step'].max()}")
    print(f"outcomes:\n{combined['outcome'].value_counts()}")


if __name__ == "__main__":
    main()