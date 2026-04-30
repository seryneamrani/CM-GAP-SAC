"""
sdf_to_map.py - Extrait une carte d'occupation 2D depuis un monde Gazebo SDF.

Strategie :
  1. Lance Gazebo en mode headless avec le monde
  2. Pendant ce temps, fait tourner un robot virtuel LiDAR avec 360 deg
  3. Capture les retours et construit la carte
  
ALTERNATIVE PLUS SIMPLE (utilisee ici) :
  1. Parse le SDF directement
  2. Recupere les modeles avec leurs poses et collision shapes
  3. Rasterise en grille 2D
  
Usage:
  python3 sdf_to_map.py <world.sdf> <output_name> [--resolution 0.05] [--padding 1.0]
  
Genere : <output_name>.pgm + <output_name>.yaml dans le dossier courant
"""
import sys
import os
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def parse_pose(pose_text):
    """Parse '<pose>x y z roll pitch yaw</pose>' -> (x, y, z, roll, pitch, yaw)."""
    if pose_text is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parts = pose_text.strip().split()
    vals = [float(p) for p in parts[:6]]
    while len(vals) < 6:
        vals.append(0.0)
    return tuple(vals)


def parse_size(size_text):
    """Parse '<size>x y z</size>' -> (x, y, z)."""
    parts = size_text.strip().split()
    return tuple(float(p) for p in parts[:3])


def get_model_pose(model_elem):
    """Recupere la pose du modele dans le monde."""
    pose_elem = model_elem.find('pose')
    if pose_elem is not None:
        return parse_pose(pose_elem.text)
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def get_model_collisions(model_elem):
    """Recupere toutes les collision shapes du modele avec leur pose locale."""
    collisions = []
    for link in model_elem.findall('.//link'):
        link_pose_elem = link.find('pose')
        link_pose = parse_pose(link_pose_elem.text if link_pose_elem is not None else None)
        
        for collision in link.findall('collision'):
            col_pose_elem = collision.find('pose')
            col_pose = parse_pose(col_pose_elem.text if col_pose_elem is not None else None)
            
            geom = collision.find('geometry')
            if geom is None:
                continue
            
            box = geom.find('box')
            cylinder = geom.find('cylinder')
            
            if box is not None:
                size_elem = box.find('size')
                if size_elem is not None:
                    sx, sy, sz = parse_size(size_elem.text)
                    collisions.append({
                        'type': 'box',
                        'size': (sx, sy, sz),
                        'link_pose': link_pose,
                        'col_pose': col_pose,
                    })
            elif cylinder is not None:
                radius_elem = cylinder.find('radius')
                length_elem = cylinder.find('length')
                if radius_elem is not None and length_elem is not None:
                    r = float(radius_elem.text)
                    h = float(length_elem.text)
                    collisions.append({
                        'type': 'cylinder',
                        'radius': r,
                        'height': h,
                        'link_pose': link_pose,
                        'col_pose': col_pose,
                    })
    return collisions


def transform_point(point, pose):
    """Applique une transformation 2D (x, y, yaw) a un point."""
    x, y = point
    px, py, _, _, _, yaw = pose
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    nx = px + x * cos_y - y * sin_y
    ny = py + x * sin_y + y * cos_y
    return (nx, ny)


def compose_poses(pose_a, pose_b):
    """Compose pose_a o pose_b (pose B exprimee dans le repere de A) -> pose monde."""
    ax, ay, az, _, _, ayaw = pose_a
    bx, by, bz, broll, bpitch, byaw = pose_b
    cos_a = math.cos(ayaw)
    sin_a = math.sin(ayaw)
    nx = ax + bx * cos_a - by * sin_a
    ny = ay + bx * sin_a + by * cos_a
    nz = az + bz
    return (nx, ny, nz, broll, bpitch, ayaw + byaw)


def collect_obstacles(sdf_root):
    """
    Parse le SDF et retourne une liste d'obstacles 2D (footprint au sol).
    Chaque obstacle = liste de points formant un polygone XY.
    """
    obstacles = []
    
    # Tous les modeles dans le monde
    world = sdf_root.find('world')
    if world is None:
        # Peut-etre que la racine est directement le world
        world = sdf_root
    
    models = world.findall('.//model') if world is not None else []
    
    print(f"Modeles trouves: {len(models)}")
    
    for model in models:
        model_name = model.get('name', '?')
        
        # Skip le robot lui-meme et le sol
        if 'limo' in model_name.lower():
            continue
        if model_name.lower() in ('ground_plane', 'ground', 'sun'):
            continue
        
        # Pose du modele dans le monde
        model_pose = get_model_pose(model)
        
        # Collision shapes
        collisions = get_model_collisions(model)
        if not collisions:
            continue
        
        for col in collisions:
            # Pose totale = model_pose o link_pose o col_pose
            total_pose = compose_poses(model_pose, col['link_pose'])
            total_pose = compose_poses(total_pose, col['col_pose'])
            
            # Si le shape est trop bas (sol) ou trop haut (plafond), on ignore
            tx, ty, tz, _, _, _ = total_pose
            if col['type'] == 'box':
                sx, sy, sz = col['size']
                # Verifier que ca touche le plan du robot (z entre 0 et 1m)
                z_min = tz - sz/2
                z_max = tz + sz/2
                if z_max < 0.05 or z_min > 1.0:
                    continue
                
                # Footprint = rectangle aux 4 coins
                hw = sx / 2
                hh = sy / 2
                corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                world_corners = [transform_point(c, total_pose) for c in corners]
                obstacles.append(('polygon', world_corners))
                
            elif col['type'] == 'cylinder':
                r = col['radius']
                h = col['height']
                z_min = tz - h/2
                z_max = tz + h/2
                if z_max < 0.05 or z_min > 1.0:
                    continue
                obstacles.append(('circle', (tx, ty, r)))
        
        print(f"  Modele: {model_name} -> {len(collisions)} collision shapes")
    
    return obstacles


def rasterize_obstacles(obstacles, resolution=0.05, padding=2.0):
    """
    Rasterize les obstacles en grille 2D.
    Retourne (image, origin_x, origin_y, resolution).
    Image: 0=libre (blanc), 100=occupe (noir), 50=inconnu (gris)
    Format Nav2 : 254=libre, 0=occupe, 205=inconnu
    """
    if not obstacles:
        # Carte vide minimaliste 10x10m
        x_min, x_max = -5.0, 5.0
        y_min, y_max = -5.0, 5.0
    else:
        # Calculer bounding box
        xs = []
        ys = []
        for obs_type, obs_data in obstacles:
            if obs_type == 'polygon':
                for x, y in obs_data:
                    xs.append(x)
                    ys.append(y)
            elif obs_type == 'circle':
                cx, cy, r = obs_data
                xs.extend([cx - r, cx + r])
                ys.extend([cy - r, cy + r])
        
        x_min = min(xs) - padding
        x_max = max(xs) + padding
        y_min = min(ys) - padding
        y_max = max(ys) + padding
    
    width_m = x_max - x_min
    height_m = y_max - y_min
    width_px = int(math.ceil(width_m / resolution))
    height_px = int(math.ceil(height_m / resolution))
    
    print(f"Dimensions carte: {width_m:.1f}m x {height_m:.1f}m  ({width_px}px x {height_px}px)")
    print(f"Origin: ({x_min:.2f}, {y_min:.2f})")
    
    # Initialise a 254 (libre)
    img = np.full((height_px, width_px), 254, dtype=np.uint8)
    
    def world_to_pixel(x, y):
        px = int((x - x_min) / resolution)
        py = int((y - y_min) / resolution)
        # Important: en image, l'axe Y est inverse (top-down)
        py = height_px - 1 - py
        return (px, py)
    
    # Rasterize chaque obstacle
    from PIL import Image as PImage
    from PIL import ImageDraw
    
    pil_img = PImage.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    for obs_type, obs_data in obstacles:
        if obs_type == 'polygon':
            pixels = [world_to_pixel(x, y) for x, y in obs_data]
            draw.polygon(pixels, fill=0)
        elif obs_type == 'circle':
            cx, cy, r = obs_data
            cx_px, cy_px = world_to_pixel(cx, cy)
            r_px = int(r / resolution)
            draw.ellipse(
                [(cx_px - r_px, cy_px - r_px), (cx_px + r_px, cy_px + r_px)],
                fill=0
            )
    
    img = np.array(pil_img)
    return img, x_min, y_min, resolution


def save_map(img, origin_x, origin_y, resolution, output_name):
    """Sauve l'image PGM + le YAML Nav2."""
    pgm_path = f"{output_name}.pgm"
    yaml_path = f"{output_name}.yaml"
    
    # Sauve PGM
    Image.fromarray(img, mode='L').save(pgm_path)
    print(f"PGM sauve: {pgm_path}")
    
    # Sauve YAML
    yaml_content = f"""image: {os.path.basename(pgm_path)}
mode: trinary
resolution: {resolution}
origin: [{origin_x:.3f}, {origin_y:.3f}, 0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"YAML sauve: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Extrait carte 2D d\'un monde SDF Gazebo')
    parser.add_argument('sdf_path', help='Chemin vers le fichier .sdf du monde')
    parser.add_argument('output_name', help='Nom de sortie (sans extension)')
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='Resolution en m/pixel (defaut 0.05)')
    parser.add_argument('--padding', type=float, default=2.0,
                        help='Marge autour des obstacles en metres (defaut 2.0)')
    args = parser.parse_args()
    
    if not os.path.exists(args.sdf_path):
        print(f"ERREUR : Fichier introuvable : {args.sdf_path}")
        sys.exit(1)
    
    print(f"Parsing : {args.sdf_path}")
    tree = ET.parse(args.sdf_path)
    root = tree.getroot()
    
    obstacles = collect_obstacles(root)
    print(f"\nTotal obstacles 2D collectes : {len(obstacles)}")
    
    if not obstacles:
        print("ATTENTION : aucun obstacle trouve, carte vide generee.")
    
    img, ox, oy, res = rasterize_obstacles(
        obstacles, resolution=args.resolution, padding=args.padding
    )
    
    save_map(img, ox, oy, res, args.output_name)
    print("\nDone.")


if __name__ == '__main__':
    main()
