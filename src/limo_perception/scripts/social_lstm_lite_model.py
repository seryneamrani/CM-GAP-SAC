#!/usr/bin/env python3
"""
social_lstm_lite_model.py -- Architecture Social-LSTM Lite.

Doit etre identique a celle utilisee pour l'entrainement.
Sert au trajectory_predictor_node a charger le state_dict.

Architecture :
    Input  : (B, T_obs=8, 4)    -- (x, y, vx, vy) normalise
    Output : (B, T_pred=6, 2)   -- (dx, dy) deltas relatifs a la
                                    derniere observation
    ~30 000 params au total.
"""
import torch
import torch.nn as nn


class SocialLSTMLite(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        encoder_hidden: int = 64,
        decoder_hidden: int = 32,
        output_dim: int = 2,
        pred_len: int = 6,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim

        # Encodeur : LSTM 64 unites
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=encoder_hidden,
            num_layers=1,
            batch_first=True,
        )
        # Bridge encoder -> decoder
        self.bridge_h = nn.Linear(encoder_hidden, decoder_hidden)
        self.bridge_c = nn.Linear(encoder_hidden, decoder_hidden)

        # Decodeur : LSTM 32 unites
        self.decoder = nn.LSTM(
            input_size=output_dim,  # input = derniere prediction
            hidden_size=decoder_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Tete de regression
        self.head = nn.Linear(decoder_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T_obs, 4) -- positions et vitesses normalisees
        Returns:
            out : (B, T_pred, 2) -- deltas relatifs a la derniere obs
        """
        B = x.size(0)
        # Encode
        _, (h_enc, _) = self.encoder(x)        # h_enc : (1, B, 64)
        h0 = self.bridge_h(h_enc)              # (1, B, 32)
        c0 = self.bridge_c(h_enc)              # (1, B, 32)

        # Decode autoregressif (teacher-free)
        outputs = []
        prev = torch.zeros(B, 1, self.output_dim, device=x.device)
        h, c = h0, c0
        for _ in range(self.pred_len):
            out, (h, c) = self.decoder(prev, (h, c))   # out: (B, 1, 32)
            delta = self.head(out)                      # (B, 1, 2)
            outputs.append(delta)
            prev = delta

        return torch.cat(outputs, dim=1)        # (B, T_pred, 2)


def load_model(weights_path: str, device: str = "cuda") -> SocialLSTMLite:
    """Charge un state_dict ou un modele entier."""
    model = SocialLSTMLite()
    obj = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        # state_dict ou checkpoint
        sd = obj.get("model_state_dict", obj.get("state_dict", obj))
        model.load_state_dict(sd)
    else:
        # modele entier
        model = obj
    model.to(device).eval()
    return model


if __name__ == "__main__":
    # Test rapide
    m = SocialLSTMLite()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Params entrainables : {n:,}")
    x = torch.randn(2, 8, 4)
    y = m(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {y.shape}")