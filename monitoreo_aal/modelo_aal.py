"""
CNN-GRU que predice el recorrido del próximo segundo a partir de la
posición observada por la infraestructura, y puntajes de anomalía.
"""
import numpy as np
import torch
import torch.nn as nn
from sim_aal import CFG, POI, SALA


# ---------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------

def ventanas(eps, cfg=CFG, stride=1):
    """Pares (historial observado, desplazamiento futuro verdadero)."""
    L, H = cfg['history'], cfg['horizon']
    X, Y = [], []
    for e in eps:
        obs, body = e['obs'], e['body']
        for t in range(L - 1, len(obs) - H, stride):
            X.append(obs[t - L + 1:t + 1])
            Y.append(body[t + 1:t + 1 + H] - obs[t])
    return np.array(X, np.float32), np.array(Y, np.float32)


class Rasterizador(nn.Module):
    """Mapa cenital 20x20: ocupante (dinámico) y lugares habituales (fijo)."""

    def __init__(self, G=CFG['grid'], sigma=0.20):
        super().__init__()
        c = (torch.arange(G, dtype=torch.float32) + 0.5) / G * float(SALA[0])
        self.register_buffer('c', c)
        self.sigma = sigma
        gx, gy = torch.meshgrid(c, c, indexing='ij')
        poi = torch.tensor(POI, dtype=torch.float32)
        lay = torch.exp(-((gx[None] - poi[:, 0, None, None]) ** 2 +
                          (gy[None] - poi[:, 1, None, None]) ** 2) / (2 * 0.3 ** 2)).sum(0)
        self.register_buffer('layout', lay.clamp(max=1.0))

    def forward(self, pos):                      # (B, L, 2) -> (B, L, 2, G, G)
        dx = pos[..., 0, None] - self.c          # (B, L, G)
        dy = pos[..., 1, None] - self.c
        occ = torch.exp(-dx[..., :, None] ** 2 / (2 * self.sigma ** 2)
                        - dy[..., None, :] ** 2 / (2 * self.sigma ** 2))
        lay = self.layout.expand_as(occ)
        return torch.stack([occ, lay], dim=2)


class CNNGRU(nn.Module):
    def __init__(self, H=CFG['horizon'], hid=64):
        super().__init__()
        self.H = H
        self.ras = Rasterizador()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 5 * 5, 64), nn.ReLU())
        self.gru = nn.GRU(64 + 4, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 2 * H))

    def forward(self, pos):                      # pos observadas (B, L, 2) en m
        B, L, _ = pos.shape
        mapas = self.ras(pos).flatten(0, 1)
        emb = self.cnn(mapas).view(B, L, -1)
        vel = torch.diff(pos, dim=1, prepend=pos[:, :1]) / CFG['dt']
        coords = torch.cat([pos / float(SALA[0]) - 0.5, 0.3 * vel], dim=-1)
        h, _ = self.gru(torch.cat([emb, coords], dim=-1))
        return self.head(h[:, -1]).view(B, self.H, 2)   # desplazamiento desde pos[:, -1]


def entrenar(modelo, Xtr, Ytr, Xva, Yva, epocas=15, lote=256, lr=2e-3, seed=0, log=print):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    Xtr, Ytr = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xva, Yva = torch.from_numpy(Xva), torch.from_numpy(Yva)
    hist = []
    for ep in range(epocas):
        modelo.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), lote):
            idx = perm[i:i + lote]
            loss = nn.functional.smooth_l1_loss(modelo(Xtr[idx]), Ytr[idx], beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        va = error_medio(modelo, Xva, Yva)
        hist.append((tot / len(Xtr), va))
        log(f'época {ep + 1:2d}  pérdida {tot / len(Xtr):.4f}  error val a 1 s {100 * va:.1f} cm')
    return hist


@torch.no_grad()
def predecir(modelo, X, lote=2048):
    modelo.eval()
    X = torch.as_tensor(X)
    return torch.cat([modelo(X[i:i + lote]) for i in range(0, len(X), lote)]).numpy()


def error_medio(modelo, X, Y):
    """Error euclidiano medio en el último paso del horizonte (1 s), en m."""
    P = predecir(modelo, X)
    return float(np.linalg.norm(P[:, -1] - np.asarray(Y)[:, -1], axis=-1).mean())


# ---------------------------------------------------------------------
# Predicciones por episodio y puntajes de anomalía
# ---------------------------------------------------------------------

def predicciones_modelo(modelo, e, cfg=CFG):
    """pred[t, h] = posición absoluta predicha en t para t+h+1 (NaN si no hay historial)."""
    L, H = cfg['history'], cfg['horizon']
    obs = e['obs']
    T = len(obs)
    X = np.stack([obs[t - L + 1:t + 1] for t in range(L - 1, T)]).astype(np.float32)
    pred = np.full((T, H, 2), np.nan)
    pred[L - 1:] = obs[L - 1:, None] + predecir(modelo, X)
    return pred


def predicciones_cv(e, cfg=CFG):
    """Extrapolación a velocidad constante desde observaciones suavizadas."""
    H, dt = cfg['horizon'], cfg['dt']
    obs = e['obs']
    T = len(obs)
    pred = np.full((T, H, 2), np.nan)
    k = np.arange(1, H + 1)[:, None]
    for t in range(7, T):
        ancla = obs[t - 2:t + 1].mean(0)               # representa t-1
        v = (ancla - obs[t - 7:t - 4].mean(0)) / (5 * dt)
        pred[t] = ancla + v * (k + 1) * dt
    return pred


def residuo(pred, obs, cfg=CFG):
    """r_t: discrepancia media entre lo observado en t y lo que se predijo para t."""
    H = cfg['horizon']
    T = len(obs)
    r = np.full(T, np.nan)
    for t in range(H, T):
        p = np.array([pred[t - h, h - 1] for h in range(1, H + 1)])
        r[t] = np.linalg.norm(p - obs[t], axis=-1).mean()
    return r


def suavizar(x, n=10):
    """Media móvil causal de n muestras (1 s)."""
    out = np.full_like(x, np.nan)
    for t in range(n - 1, len(x)):
        w = x[t - n + 1:t + 1]
        if not np.isnan(w).any():
            out[t] = w.mean()
    return out


def puntaje_inmovilidad(e, cfg=CFG, v_umbral=0.12, d_poi=0.6):
    """
    Regla de referencia: segundos consecutivos de inmovilidad lejos de
    cualquier lugar habitual. No usa aprendizaje.
    """
    obs, dt = e['obs'], cfg['dt']
    T = len(obs)
    s = np.full(T, np.nan)
    cont = 0.0
    for t in range(14, T):
        a, b = obs[t - 4:t + 1].mean(0), obs[t - 14:t - 9].mean(0)
        v = np.linalg.norm(a - b) / 1.0
        lejos = np.min(np.linalg.norm(POI - a, axis=1)) > d_poi
        cont = cont + dt if (v < v_umbral and lejos) else 0.0
        s[t] = cont
    return s
