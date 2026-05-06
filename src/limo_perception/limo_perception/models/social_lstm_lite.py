"""
Social-LSTM Lite
================
Version allégée de Social-LSTM (Alahi et al., CVPR 2016) pour déploiement
temps-réel sur robot mobile (AgileX LIMO Pro, RTX A1000).

Allègements par rapport au modèle original :
- hidden_size : 128 -> 32
- 1 seule couche LSTM (au lieu de 2)
- Social pooling grid 4x4 (au lieu de 8x8)
- embedding_dim : 64 -> 16
- Horizon de prédiction court (8-16 frames)

Entrée  : historique de positions (x, y) sur obs_len frames pour N agents
Sortie  : trajectoires futures (x, y) sur pred_len frames pour chaque agent

Auteur : Seryne Amrani — LITAN, ESTIN
"""

import torch
import torch.nn as nn


class SocialPoolingLite(nn.Module):
    """
    Social pooling allégé : grid 4x4 autour de chaque agent, agrégation
    des hidden states des voisins par cellule.
    """

    def __init__(self, hidden_size: int = 32, grid_size: int = 4,
                 neighborhood: float = 2.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.neighborhood = neighborhood  # mètres
        self.cell_size = neighborhood / grid_size

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """
        hidden_states : (N, hidden_size) — hidden state courant de chaque agent
        positions     : (N, 2)           — position (x, y) courante
        retourne      : (N, grid_size*grid_size*hidden_size) — tenseur social
        """
        N = positions.size(0)
        device = positions.device
        social_tensor = torch.zeros(
            N, self.grid_size, self.grid_size, self.hidden_size, device=device
        )

        if N <= 1:
            return social_tensor.view(N, -1)

        # Distances relatives entre tous les agents
        rel = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 2)

        # Pour chaque agent i, on regarde ses voisins j != i
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx, dy = rel[i, j, 0].item(), rel[i, j, 1].item()
                # j est-il dans le voisinage de i ?
                if abs(dx) >= self.neighborhood / 2 or abs(dy) >= self.neighborhood / 2:
                    continue
                # Indice de cellule (centre du grid)
                cx = int((dx + self.neighborhood / 2) / self.cell_size)
                cy = int((dy + self.neighborhood / 2) / self.cell_size)
                cx = max(0, min(self.grid_size - 1, cx))
                cy = max(0, min(self.grid_size - 1, cy))
                social_tensor[i, cx, cy] += hidden_states[j]

        return social_tensor.view(N, -1)


class SocialLSTMLite(nn.Module):
    """
    Social-LSTM Lite encoder-decoder avec social pooling.
    """

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

        # Embedding spatial
        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)

        # Social pooling
        self.social_pool = SocialPoolingLite(
            hidden_size=hidden_size,
            grid_size=grid_size,
            neighborhood=neighborhood,
        )
        social_dim = grid_size * grid_size * hidden_size

        # LSTM cell prend embedding + tensor social
        self.lstm_cell = nn.LSTMCell(embedding_dim + social_dim, hidden_size)

        # Tête de sortie : Gaussienne 2D (mux, muy, sx, sy, corr)
        # Pour version simplifiée on prédit juste (mux, muy)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self,
                obs_traj: torch.Tensor,
                pred_len: int = None) -> torch.Tensor:
        """
        obs_traj : (obs_len, N, 2) — trajectoires observées
        retourne : (pred_len, N, 2) — trajectoires prédites (offsets relatifs)
        """
        if pred_len is None:
            pred_len = self.pred_len

        obs_len, N, _ = obs_traj.size()
        device = obs_traj.device

        # Init hidden state
        h = torch.zeros(N, self.hidden_size, device=device)
        c = torch.zeros(N, self.hidden_size, device=device)

        # Encoder : on déroule sur l'historique observé
        for t in range(obs_len):
            pos_t = obs_traj[t]                   # (N, 2)
            emb_t = self.spatial_embedding(pos_t)  # (N, embedding_dim)
            social_t = self.social_pool(h, pos_t)  # (N, social_dim)
            x = torch.cat([emb_t, social_t], dim=-1)
            h, c = self.lstm_cell(x, (h, c))

        # Decoder : on prédit pred_len pas en avant
        # On utilise la dernière position observée comme point de départ
        preds = []
        last_pos = obs_traj[-1]  # (N, 2)
        cur_pos = last_pos.clone()

        for _ in range(pred_len):
            emb = self.spatial_embedding(cur_pos)
            social = self.social_pool(h, cur_pos)
            x = torch.cat([emb, social], dim=-1)
            h, c = self.lstm_cell(x, (h, c))
            offset = self.output_layer(h)         # (N, 2) -- delta
            cur_pos = cur_pos + offset
            preds.append(cur_pos.unsqueeze(0))

        return torch.cat(preds, dim=0)             # (pred_len, N, 2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check
    model = SocialLSTMLite(
        embedding_dim=16,
        hidden_size=32,
        grid_size=4,
        neighborhood=2.0,
        pred_len=12,
    )
    print(f"Paramètres entraînables : {count_parameters(model):,}")

    # Test forward : 8 frames observées, 5 agents
    obs = torch.randn(8, 5, 2)
    pred = model(obs, pred_len=12)
    print(f"Entrée  : {obs.shape}")
    print(f"Sortie  : {pred.shape}")
