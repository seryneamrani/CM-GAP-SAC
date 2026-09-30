"""
Bibliotheque de calcul des metriques d'evaluation pour LIMO Pro.

Toutes les fonctions prennent en entree des structures de donnees collectees
par evaluation_node.py et retournent un scalaire ou un dict de scalaires.
"""
import math
from typing import List, Tuple, Optional
import numpy as np


# =============================================================================
# METRIQUES ROBOTIQUE
# =============================================================================

def compute_distance_traveled(odom_positions: List[Tuple[float, float]]) -> float:
    """
    Distance totale parcourue (somme des deplacements entre samples odom).
    
    Args:
        odom_positions: liste de (x, y) en metres
    
    Returns:
        distance en metres
    """
    if len(odom_positions) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(odom_positions)):
        dx = odom_positions[i][0] - odom_positions[i-1][0]
        dy = odom_positions[i][1] - odom_positions[i-1][1]
        dist += math.sqrt(dx*dx + dy*dy)
    return dist


def compute_optimal_distance(start: Tuple[float, float], 
                              goal: Tuple[float, float]) -> float:
    """
    Distance euclidienne optimale (borne inferieure A* en espace libre).
    Utilisee pour calculer SPL et TDI.
    """
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    return math.sqrt(dx*dx + dy*dy)


def compute_spl(success: bool, 
                actual_length: float, 
                optimal_length: float) -> float:
    """
    Success-weighted Path Length (Anderson et al. 2018).
    SPL = success * (optimal / max(actual, optimal))
    
    Returns 1.0 pour trajet optimal reussi, 0.0 pour echec.
    """
    if not success or actual_length <= 0:
        return 0.0
    return optimal_length / max(actual_length, optimal_length)


def compute_tdi(actual_length: float, optimal_length: float) -> float:
    """
    Trajectory Deviation Index : (actual - optimal) / optimal.
    
    0.0 = trajet optimal, 0.5 = 50% plus long que l'optimal.
    """
    if optimal_length <= 0:
        return float('nan')
    return (actual_length - optimal_length) / optimal_length


def compute_smoothness_jerk(cmd_vel_history: List[Tuple[float, float, float]],
                             dt: float = 0.1) -> float:
    """
    Smoothness via integrale du jerk (derivee de l'acceleration).
    
    Args:
        cmd_vel_history: liste de (linear_x, angular_z, timestamp)
        dt: pas de temps moyen entre samples
    
    Returns:
        jerk moyen quadratique (m/s^3 + rad/s^3)
    """
    if len(cmd_vel_history) < 4:
        return float('nan')
    
    velocities = np.array([(v[0], v[1]) for v in cmd_vel_history])
    
    # Acceleration = derivee 1ere
    accelerations = np.diff(velocities, axis=0) / dt
    
    # Jerk = derivee 2eme
    jerks = np.diff(accelerations, axis=0) / dt
    
    # Norme moyenne du jerk
    jerk_magnitudes = np.linalg.norm(jerks, axis=1)
    return float(np.mean(jerk_magnitudes))


def compute_ttc_min(odom_history: List[Tuple[float, float, float, float]],
                     scan_distances: List[float],
                     speed_threshold: float = 0.05) -> float:
    """
    Time-To-Collision minimum sur l'episode.
    TTC = distance_obstacle / vitesse_relative_d_approche
    
    Args:
        odom_history: (x, y, vx_linear, timestamp)
        scan_distances: distance min au plus proche obstacle a chaque step
        speed_threshold: en dessous, robot considere a l'arret (TTC = inf)
    
    Returns:
        TTC minimum (secondes), inf si jamais en mouvement
    """
    if len(odom_history) != len(scan_distances) or len(odom_history) == 0:
        return float('inf')
    
    ttcs = []
    for (x, y, v, _), d in zip(odom_history, scan_distances):
        if abs(v) < speed_threshold or d <= 0:
            continue
        ttcs.append(d / abs(v))
    
    if not ttcs:
        return float('inf')
    return float(min(ttcs))


def compute_d_min(scan_distances: List[float]) -> float:
    """Distance minimum a un obstacle pendant tout l'episode."""
    if not scan_distances:
        return float('nan')
    valid = [d for d in scan_distances if d > 0 and d < float('inf')]
    return min(valid) if valid else float('nan')


def count_collisions(scan_distances: List[float], 
                      threshold: float = 0.05) -> int:
    """
    Compte les episodes de collision (distance min < threshold).
    Une collision = passage continu sous threshold; on compte les transitions.
    """
    if not scan_distances:
        return 0
    
    in_collision = False
    n_collisions = 0
    for d in scan_distances:
        if d < threshold and not in_collision:
            n_collisions += 1
            in_collision = True
        elif d >= threshold:
            in_collision = False
    return n_collisions


def count_replanifications(plan_history: List[float]) -> int:
    """
    Compte le nombre de replanifications detectees via les timestamps de /plan.
    Chaque nouveau message /plan apres un delai > 0.5s = une replanif.
    """
    if len(plan_history) < 2:
        return 0
    n = 0
    for i in range(1, len(plan_history)):
        if plan_history[i] - plan_history[i-1] > 0.5:
            n += 1
    return n


# =============================================================================
# METRIQUES PERCEPTION (utilisable seulement pour A2 et A4)
# =============================================================================

def compute_detection_latency_percentiles(
    image_timestamps: List[float],
    detection_timestamps: List[float]
) -> dict:
    """
    Latence entre acquisition image et publication detection.
    
    Returns dict avec p50, p95, p99 en millisecondes.
    """
    if not image_timestamps or not detection_timestamps:
        return {'p50_ms': float('nan'), 'p95_ms': float('nan'), 'p99_ms': float('nan')}
    
    # Apparier chaque detection avec son image (par stamp le plus proche)
    latencies_ms = []
    for det_t in detection_timestamps:
        # On suppose que image_timestamps est trie
        idx = np.searchsorted(image_timestamps, det_t)
        if 0 < idx <= len(image_timestamps):
            img_t = image_timestamps[idx-1]
            latencies_ms.append((det_t - img_t) * 1000.0)
    
    if not latencies_ms:
        return {'p50_ms': float('nan'), 'p95_ms': float('nan'), 'p99_ms': float('nan')}
    
    arr = np.array(latencies_ms)
    return {
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
    }


# =============================================================================
# AGREGATION POUR UN RUN
# =============================================================================

def aggregate_run_metrics(run_data: dict) -> dict:
    """
    Calcule toutes les metriques d'un run a partir des donnees collectees.
    
    Args:
        run_data: dict avec cles suivantes :
            - run_id, architecture, world, pair_id, repetition_id
            - success (bool), reason (str)
            - start_pose, goal_pose : (x, y, yaw)
            - odom_positions : List[(x,y)]
            - odom_history_full : List[(x, y, v_lin, t)]
            - cmd_vel_history : List[(lin, ang, t)]
            - scan_distances : List[float]
            - plan_timestamps : List[float]
            - duration_seconds : float
    
    Returns:
        dict avec toutes les metriques
    """
    actual_dist = compute_distance_traveled(run_data['odom_positions'])
    optimal_dist = compute_optimal_distance(
        run_data['start_pose'][:2], run_data['goal_pose'][:2]
    )
    
    metrics = {
        # Identifiants
        'run_id': run_data['run_id'],
        'architecture': run_data['architecture'],
        'world': run_data['world'],
        'pair_id': run_data['pair_id'],
        'repetition_id': run_data['repetition_id'],
        
        # Resultat
        'success': int(run_data['success']),
        'failure_reason': run_data.get('reason', ''),
        
        # Robotique
        'duration_s': run_data['duration_seconds'],
        'distance_m': actual_dist,
        'optimal_distance_m': optimal_dist,
        'spl': compute_spl(run_data['success'], actual_dist, optimal_dist),
        'tdi': compute_tdi(actual_dist, optimal_dist),
        'collisions': count_collisions(run_data['scan_distances']),
        'replanifications': count_replanifications(run_data['plan_timestamps']),
        'd_min_obstacle_m': compute_d_min(run_data['scan_distances']),
        'jerk_mean': compute_smoothness_jerk(run_data['cmd_vel_history']),
        'ttc_min_s': compute_ttc_min(
            run_data['odom_history_full'], run_data['scan_distances']
        ),
    }
    
    # Metriques perception (si disponibles)
    if 'image_timestamps' in run_data and 'detection_timestamps' in run_data:
        latency = compute_detection_latency_percentiles(
            run_data['image_timestamps'], run_data['detection_timestamps']
        )
        metrics.update({
            f'perception_latency_{k}': v for k, v in latency.items()
        })
    
    return metrics
