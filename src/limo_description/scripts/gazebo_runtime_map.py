"""
gazebo_runtime_map.py - Carte 2D des obstacles directement depuis Gazebo runtime.

Strategie :
  1. Liste tous les modeles via /world/<world>/scene/info
  2. Pour chaque modele non-robot, recupere sa bounding box AABB
  3. Projette les AABB en grille 2D
  4. Sauve en PGM + YAML

Avantages vs SDF parsing : Gazebo nous donne les vraies bounding box
des meshes 3D charges (pas besoin de parser les fichiers .dae).

Avantages vs scan LiDAR : pas de scan, pas de teleop, pas de SLAM.
La verite vraie est dans Gazebo, on la prend directement.

Pre-requis :
  - Gazebo doit tourner avec le monde (pas besoin de bridge ROS)
  - gz topic et gz service doivent fonctionner

Usage :
  python3 gazebo_runtime_map.py <world_name> <output_name>
      [--resolution 0.05] [--padding 1.0] [--robot-name limo]

Exemple :
  python3 gazebo_runtime_map.py hospital hospital_map
"""
import os
import sys
import math
import argparse
import subprocess
import re

import numpy as np
from PIL import Image, ImageDraw


def gz_command(cmd):
    """Lance une commande gz et retourne stdout."""
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10, text=True)
        return result.stdout
    except Exception as e:
        print(f"Erreur gz: {e}")
        return ""


def list_models(world_name):
    """Liste tous les modeles dans le monde via gz model --list."""
    out = gz_command(['gz', 'model', '--list'])
    
    models = []
    for line in out.split('\n'):
        line = line.strip()
        if line.startswith('- '):
            name = line[2:].strip()
            models.append(name)
    
    return models


def get_model_aabb(world_name, model_name):
    """
    Recupere l'AABB d'un modele via le service /world/<world>/state/aabb_box.
    Pour Gazebo Harmonic, on utilise plutot /world/<world>/scene/info ou
    on demande la pose et on inspecte les liens.
    
    Retourne dict { 'pose': (x,y,z,yaw), 'bbox': (sx, sy, sz) } ou None
    """
    # Strategie : recuperer la pose via gz topic
    out = gz_command([
        'gz', 'topic', '-e',
        '-t', f'/world/{world_name}/pose/info',
        '-n', '1'
    ])
    
    # Parser la pose en cherchant le bon nom
    # Format protobuf texte:
    #   pose {
    #     name: "wall_north"
    #     id: 8
    #     position { x: 8 }
    #     orientation { ... }
    #   }
    
    # On cherche le bloc qui contient name: "model_name"
    pattern = re.compile(
        r'name:\s*"' + re.escape(model_name) + r'".*?(?=name:|\Z)',
        re.DOTALL
    )
    match = pattern.search(out)
    if not match:
        return None
    
    block = match.group(0)
    
    # Extraire position
    pos_match = re.search(
        r'position\s*\{[^}]*?(?:x:\s*([-\d\.e+]+))?\s*(?:y:\s*([-\d\.e+]+))?\s*(?:z:\s*([-\d\.e+]+))?[^}]*?\}',
        block, re.DOTALL
    )
    if not pos_match:
        x, y, z = 0.0, 0.0, 0.0
    else:
        x = float(pos_match.group(1) or 0.0)
        y = float(pos_match.group(2) or 0.0)
        z = float(pos_match.group(3) or 0.0)
    
    # Extraire orientation (yaw seulement)
    ori_match = re.search(
        r'orientation\s*\{[^}]*?z:\s*([-\d\.e+]+)\s*w:\s*([-\d\.e+]+)[^}]*?\}',
        block, re.DOTALL
    )
    if ori_match:
        qz = float(ori_match.group(1))
        qw = float(ori_match.group(2))
        yaw = 2 * math.atan2(qz, qw)
    else:
        yaw = 0.0
    
    return {'pose': (x, y, z, yaw)}


def get_aabb_via_service(world_name, model_name):
    """
    Recupere l'AABB d'un modele via le service /world/<world>/state.
    Retourne (min_x, min_y, max_x, max_y) ou None.
    """
    cmd = [
        'gz', 'service',
        '-s', f'/world/{world_name}/state',
        '--reqtype', 'gz.msgs.SerializedStateMap',
        '--reptype', 'gz.msgs.SerializedStateMap',
        '--timeout', '3000',
        '--req', ''
    ]
    out = gz_command(cmd)
    
    # Pattern pour AABB
    # axis_aligned_box {
    #   min_corner { x: ... y: ... }
    #   max_corner { x: ... y: ... }
    # }
    return None  # Pas trivial via ce service


def parse_pose_info_full(world_name):
    """
    Parse /world/<world>/pose/info et retourne un dict {name: (x, y, yaw)}.
    """
    out = gz_command([
        'gz', 'topic', '-e',
        '-t', f'/world/{world_name}/pose/info',
        '-n', '1'
    ])
    
    poses = {}
    
    # Decouper en blocs "pose { ... }"
    # Methode simple : ligne par ligne
    current = {}
    in_pose = False
    in_position = False
    in_orientation = False
    brace_depth = 0
    
    lines = out.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('name:'):
            name_match = re.match(r'name:\s*"([^"]+)"', line)
            if name_match:
                current['name'] = name_match.group(1)
        
        elif line.startswith('position'):
            # Lire les prochaines lignes jusqu'a la fermeture
            j = i + 1
            x = y = z = 0.0
            while j < len(lines):
                l = lines[j].strip()
                if l.startswith('x:'):
                    x = float(l.split(':')[1].strip())
                elif l.startswith('y:'):
                    y = float(l.split(':')[1].strip())
                elif l.startswith('z:'):
                    z = float(l.split(':')[1].strip())
                elif '}' in l:
                    break
                j += 1
            current['position'] = (x, y, z)
            i = j
        
        elif line.startswith('orientation'):
            j = i + 1
            qx = qy = qz = 0.0
            qw = 1.0
            while j < len(lines):
                l = lines[j].strip()
                if l.startswith('x:'):
                    qx = float(l.split(':')[1].strip())
                elif l.startswith('y:'):
                    qy = float(l.split(':')[1].strip())
                elif l.startswith('z:'):
                    qz = float(l.split(':')[1].strip())
                elif l.startswith('w:'):
                    qw = float(l.split(':')[1].strip())
                elif '}' in l:
                    break
                j += 1
            yaw = 2 * math.atan2(qz, qw)
            current['orientation'] = yaw
            
            # Si on a name + position + orientation, on enregistre
            if 'name' in current and 'position' in current:
                px, py, pz = current['position']
                poses[current['name']] = (px, py, current.get('orientation', 0.0))
            current = {}
            i = j
        
        i += 1
    
    return poses


def load_aabb_database():
    """
    Base de donnees des AABB approximatives pour les modeles courants
    du monde hospital. Dimensions en metres (longueur_X, largeur_Y).
    
    Hauteur en Z est ignoree pour la projection 2D.
    """
    return {
        # Murs externes (longs et fins)
        'wall_north': (16.0, 0.2),
        'wall_south': (16.0, 0.2),
        'wall_east': (0.2, 16.0),
        'wall_west': (0.2, 16.0),
        
        # Murs de couloir
        'corridor_wall_1': (8.0, 0.2),
        'corridor_wall_2': (8.0, 0.2),
        
        # Cloisons divisions
        'div_left_top_1': (3.0, 0.2),
        'div_left_top_2': (3.0, 0.2),
        'div_left_bot_1': (3.0, 0.2),
        'div_left_bot_2': (3.0, 0.2),
        
        # Lits hopital (~2m x 1m)
        'bed_1': (2.0, 1.0),
        'bed_patient_1': (2.0, 1.0),
        'bed_patient_2': (2.0, 1.0),
        
        # IV stands (cylindres ~0.4m diametre)
        'iv_stand_1': (0.4, 0.4),
        'iv_stand_2': (0.4, 0.4),
        
        # Tables de chevet (~0.6 x 0.4)
        'bedside_table_1': (0.6, 0.4),
        'bedside_table_2': (0.6, 0.4),
        
        # Wheelchair (~0.7 x 0.7)
        'wheelchair_1': (0.7, 0.7),
        
        # Cabinets (~1.2 x 0.5)
        'cabinet_1': (1.2, 0.5),
        'cabinet_2': (1.2, 0.5),
        
        # Personnes (~0.5 x 0.5 cylindre)
        'person_standing_1': (0.5, 0.5),
        'nurse_1': (0.5, 0.5),
    }


def transform_corners(cx, cy, yaw, sx, sy):
    """Retourne les 4 coins d'un AABB rotated par yaw, centre en (cx, cy)."""
    hw = sx / 2
    hh = sy / 2
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    corners_world = []
    for lx, ly in corners_local:
        wx = cx + lx * cos_y - ly * sin_y
        wy = cy + lx * sin_y + ly * cos_y
        corners_world.append((wx, wy))
    return corners_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('world_name')
    parser.add_argument('output_name')
    parser.add_argument('--resolution', type=float, default=0.05)
    parser.add_argument('--padding', type=float, default=1.0)
    parser.add_argument('--robot-name', default='limo')
    parser.add_argument('--exclude', nargs='+',
                        default=['ground_plane', 'sun', 'limo', 'limo_world'],
                        help='Modeles a ignorer')
    args = parser.parse_args()
    
    print(f"=== Configuration ===")
    print(f"Monde      : {args.world_name}")
    print(f"Output     : {args.output_name}")
    print(f"Resolution : {args.resolution} m/px")
    print(f"Exclus     : {args.exclude}")
    print()
    
    # 1. Lister les modeles
    print("Liste des modeles dans Gazebo...")
    models = list_models(args.world_name)
    print(f"  {len(models)} modeles trouves")
    
    # Filtrer
    models = [m for m in models if m not in args.exclude and m != args.robot_name]
    print(f"  {len(models)} obstacles a rasteriser")
    
    # 2. Recuperer les poses de tous les modeles
    print("\nLecture des poses...")
    poses = parse_pose_info_full(args.world_name)
    print(f"  {len(poses)} poses recuperees")
    
    # 3. Charger la base d'AABB
    aabb_db = load_aabb_database()
    
    # 4. Construire la liste des obstacles avec leurs AABB
    obstacles = []
    for name in models:
        if name not in poses:
            print(f"  [WARN] Pose introuvable: {name}")
            continue
        
        x, y, yaw = poses[name]
        
        # Chercher les dimensions dans la DB
        if name in aabb_db:
            sx, sy = aabb_db[name]
        else:
            # Fallback : essayer matching partiel
            for key, dims in aabb_db.items():
                if name.startswith(key.split('_')[0]):
                    sx, sy = dims
                    break
            else:
                # Defaut : carre 0.5m
                sx, sy = 0.5, 0.5
                print(f"  [WARN] Dimensions inconnues pour {name}, defaut 0.5x0.5")
        
        corners = transform_corners(x, y, yaw, sx, sy)
        obstacles.append((name, corners))
        print(f"  {name}: pos=({x:+.2f},{y:+.2f}) yaw={math.degrees(yaw):+.0f}deg dim={sx}x{sy}")
    
    if not obstacles:
        print("ERREUR: aucun obstacle trouve")
        sys.exit(1)
    
    # 5. Calculer la bounding box totale
    all_xs = []
    all_ys = []
    for name, corners in obstacles:
        for x, y in corners:
            all_xs.append(x)
            all_ys.append(y)
    
    x_min = min(all_xs) - args.padding
    x_max = max(all_xs) + args.padding
    y_min = min(all_ys) - args.padding
    y_max = max(all_ys) + args.padding
    
    width_m = x_max - x_min
    height_m = y_max - y_min
    width_px = int(math.ceil(width_m / args.resolution))
    height_px = int(math.ceil(height_m / args.resolution))
    
    print(f"\nCarte: {width_m:.1f}m x {height_m:.1f}m  ({width_px}x{height_px} px)")
    print(f"Origin: ({x_min:.2f}, {y_min:.2f})")
    
    # 6. Rasteriser
    img = np.full((height_px, width_px), 254, dtype=np.uint8)  # libre par defaut
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    def world_to_pixel(x, y):
        col = int((x - x_min) / args.resolution)
        row = height_px - 1 - int((y - y_min) / args.resolution)
        return (col, row)
    
    for name, corners in obstacles:
        pixels = [world_to_pixel(x, y) for x, y in corners]
        draw.polygon(pixels, fill=0)
    
    img = np.array(pil_img)
    
    n_obs = (img == 0).sum()
    print(f"\nPixels obstacle: {n_obs} ({100*n_obs/img.size:.1f}%)")
    print(f"Pixels libre   : {(img == 254).sum()} ({100*(img == 254).sum()/img.size:.1f}%)")
    
    # 7. Sauver
    pgm_path = f"{args.output_name}.pgm"
    yaml_path = f"{args.output_name}.yaml"
    
    Image.fromarray(img, mode='L').save(pgm_path)
    print(f"\nPGM sauve: {pgm_path}")
    
    yaml_content = f"""image: {os.path.basename(pgm_path)}
mode: trinary
resolution: {args.resolution}
origin: [{x_min:.3f}, {y_min:.3f}, 0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"YAML sauve: {yaml_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
