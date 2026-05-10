"""
Social-LSTM Lite -- VERSION OPTIMISEE
=====================================
Version allegee de Social-LSTM (Alahi et al., CVPR 2016) pour deploiement
temps-reel sur robot mobile (AgileX LIMO Pro, RTX A1000).

OPTIMISATIONS V2 par rapport au code initial :
  * SocialPoolingLite entierement vectorise (suppression de la double boucle
    Python + des .item() qui forcaient la synchro GPU/CPU a chaque pas).
  * Operation scatter (index_add_) au lieu de loop sur les voisins.
  * ~50-100x plus rapide en pratique sur GPU.

Allegements (architecture inchangee) :
  * hidden_size : 128 -> 32
  * 1 seule couche LSTM
  * Social pooling grid 4x4
  * embedding_dim : 64 -> 16

Auteur : Seryne Amrani -- LITAN, ESTIN
"""

import torch
import torch.nn as nn


class SocialPoolingLite(nn.Module):
    """
    Social pooling vectorise.

    Pour chaque agent i, agrege par cellule du grid (grid_size x grid_size)
    autour de i les hidden states des voisins j != i situes a moins de
    neighborhood/2 metres en x et y.
    """

    def __init__(self, hidden_size: int = 32, grid_size: int = 4,
                 neighborhood: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.cell_size = neighborhood / grid_size

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """
        hidden_states : (N, hidden_size)
        positions     : (N, 2)
        retourne      : (N, grid_size*grid_size*hidden_size)
        """
        N = positions.size(0)
        H = self.hidden_size
        G = self.grid_size
        n_cells = G * G
        device = positions.device

        if N <= 1:
            return torch.zeros(N, n_cells * H, device=device,
                               dtype=hidden_states.dtype)

        half = self.neighborhood / 2

        # rel[i, j] = pos[j] - pos[i]  (ou est j vu depuis i)
        rel = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 2)

        # Masque : voisins dans la fenetre, exclure soi-meme
        in_nb = (rel.abs() < half).all(dim=-1)                  # (N, N)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        in_nb = in_nb & ~eye

        # Indices de cellule (clamp en cas d'epsilon flottant aux bords)
        cell_x = ((rel[..., 0] + half) / self.cell_size).long().clamp(0, G - 1)
        cell_y = ((rel[..., 1] + half) / self.cell_size).long().clamp(0, G - 1)
        flat_cell = cell_x * G + cell_y                          # (N, N)

        # Recupere les paires valides (vectorise, GPU-only)
        valid_pairs = in_nb.nonzero(as_tuple=False)              # (K, 2)
        if valid_pairs.size(0) == 0:
            return torch.zeros(N, n_cells * H, device=device,
                               dtype=hidden_states.dtype)

        valid_i = valid_pairs[:, 0]
        valid_j = valid_pairs[:, 1]
        cells = flat_cell[valid_i, valid_j]                      # (K,)
        h_nb = hidden_states[valid_j]                            # (K, H)

        # Scatter add dans un tenseur plat (N*n_cells, H) puis reshape
        flat_idx = valid_i * n_cells + cells                     # (K,)
        social_flat = torch.zeros(N * n_cells, H, device=device,
                                  dtype=hidden_states.dtype)
        social_flat.index_add_(0, flat_idx, h_nb)

        return social_flat.view(N, n_cells * H)


class SocialLSTMLite(nn.Module):
    """Encoder-decoder LSTM avec social pooling vectorise."""

    def __init__(self,
                 input_dim: int = 2,
                 embedding_dim: int = 16,
                 hidden_size: int = 32,
                 grid_size: int = 4,
                 neighborhood: float = 2.0,
                 pred_len: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.pred_len = pred_len

        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
        self.social_pool = SocialPoolingLite(
            hidden_size=hidden_size,
            grid_size=grid_size,
            neighborhood=neighborhood,
        )
        social_dim = grid_size * grid_size * hidden_size
        self.lstm_cell = nn.LSTMCell(embedding_dim + social_dim, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self,
                obs_traj: torch.Tensor,
                pred_len: int = None) -> torch.Tensor:
        if pred_len is None:
            pred_len = self.pred_len

        obs_len, N, _ = obs_traj.size()
        device = obs_traj.device
        dtype = obs_traj.dtype

        h = torch.zeros(N, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(N, self.hidden_size, device=device, dtype=dtype)

        # Encoder
        for t in range(obs_len):
            pos_t = obs_traj[t]
            emb_t = self.spatial_embedding(pos_t)
            social_t = self.social_pool(h, pos_t)
            x = torch.cat([emb_t, social_t], dim=-1)
            h, c = self.lstm_cell(x, (h, c))

        # Decoder : prediction autoregressive
        cur_pos = obs_traj[-1].clone()
        preds = []
        for _ in range(pred_len):
            emb = self.spatial_embedding(cur_pos)
            social = self.social_pool(h, cur_pos)
            x = torch.cat([emb, social], dim=-1)
            h, c = self.lstm_cell(x, (h, c))
            offset = self.output_layer(h)
            cur_pos = cur_pos + offset
            preds.append(cur_pos.unsqueeze(0))

        return torch.cat(preds, dim=0)  # (pred_len, N, 2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(weights_path: str, device: str = "cuda") -> SocialLSTMLite:
    """Helper pour charger le modèle optimisé V2."""
    # On utilise les paramètres par défaut de ton entraînement
    model = SocialLSTMLite(
        embedding_dim=16, 
        hidden_size=32, 
        grid_size=4, 
        neighborhood=2.0, 
        pred_len=12
    )
    
    # Chargement des poids
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Si c'est un dictionnaire complet (checkpoint), on extrait l'état
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device).eval()
    return model


if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = SocialLSTMLite(
        embedding_dim=16, hidden_size=32, grid_size=4,
        neighborhood=2.0, pred_len=12,
    ).to(device)
    print(f"Parametres entrainables : {count_parameters(model):,}")

    # Test correctness
    obs = torch.randn(8, 5, 2, device=device)
    pred = model(obs)
    print(f"Test forward : {tuple(obs.shape)} -> {tuple(pred.shape)}")

    # Benchmark forward
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(obs)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        N_ITER = 200
        for _ in range(N_ITER):
            model(obs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / N_ITER * 1000
    print(f"Latence forward (5 agents, eval) : {dt:.2f} ms")

    # Benchmark forward + backward
    model.train()
    obs = torch.randn(8, 5, 2, device=device, requires_grad=False)
    gt = torch.randn(12, 5, 2, device=device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        loss = crit(model(obs), gt)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        opt.zero_grad()
        loss = crit(model(obs), gt)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / N_ITER * 1000
    print(f"Latence train step (5 agents, fwd+bwd+step) : {dt:.2f} ms")
