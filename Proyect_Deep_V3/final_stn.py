"""STN (Zhang & Zhang, 2023) — arquitectura copiada TAL CUAL de STN_PM25_Fontibon_replica.ipynb
(Secciones 3.1-3.6), con una única adición: ProbAttention puede guardar su matriz de atención
(`capture=True`) para la auditoría. Más utilidades de entrenamiento por regímenes de transferencia."""
import math, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    """Position embedding senoidal (el paper lo menciona explícitamente)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Proyección de las c_in variables a d_model con una Conv1D causal-circular."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=3,
            padding=1, padding_mode="circular", bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class TimeFeatureEmbedding(nn.Module):
    """Embedding de marcas temporales (hora, día, día-semana, mes) normalizadas.

    El paper NO lo menciona; queda desactivado por defecto (`use_time_features=False`)
    para mantener la réplica fiel. Actívalo solo como experimento extra y
    documéntalo como desviación respecto del paper.
    """

    def __init__(self, d_model: int, n_time_feats: int = 4):
        super().__init__()
        self.embed = nn.Linear(n_time_feats, d_model, bias=False)

    def forward(self, x_mark):
        return self.embed(x_mark)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05, use_time_features=False, n_time_feats=4):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.use_time_features = use_time_features
        if use_time_features:
            self.temporal_embedding = TimeFeatureEmbedding(d_model, n_time_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        out = self.value_embedding(x) + self.position_embedding(x)
        if self.use_time_features and x_mark is not None:
            out = out + self.temporal_embedding(x_mark)
        return self.dropout(out)

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1
            )

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """Máscara causal restringida a las top-u queries seleccionadas."""

    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ]
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    """Atención estándar O(L^2). El paper la usa en la atención CRUZADA del decoder."""

    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.05):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        scale = self.scale or 1.0 / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class ProbAttention(nn.Module):
    """Atención dispersa multi-head (ec. 5-7 del paper): medida de esparsidad
    aproximada por muestreo de keys + selección de las top-u queries."""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.05):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Muestreo aleatorio de keys (K~ de la ec. 7)
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # M_sparse = max - media  (aprox. de la K-L divergence contra la uniforme)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), sample_k)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V, "self-attention enmascarada requiere L_Q == L_V"
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        if getattr(self, "capture", False):          # [Entrega final] para auditar mapas de atención
            self.last_attn = attn.detach().cpu(); self.last_index = index.detach().cpu()
        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        return context_in

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * int(math.ceil(math.log(max(L_K, 2))))
        u = self.factor * int(math.ceil(math.log(max(L_Q, 2))))
        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1.0 / math.sqrt(D)
        scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context = self._update_context(context, values, scores_top, index, L_Q)
        return context.transpose(2, 1).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out)

class ConvLayer(nn.Module):
    """Destilación: Conv1D (kernel 3) + ELU + max-pooling stride 2 — tal cual el paper."""

    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in, out_channels=c_in, kernel_size=3,
            padding=1, padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """Bloque de atención dispersa: MHSA disperso + FFN, residual + LayerNorm."""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.05, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x, attn_mask=attn_mask)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderStack(nn.Module):
    """Las DOS ramas paralelas del encoder del paper.

    Rama 0: secuencia completa, e_layers bloques + destilación entre bloques.
    Rama 1: recibe ½·Lx (el paper: "el primer camino con un bloque recibe
            entradas a la mitad") con un bloque menos.
    Las salidas se concatenan en el eje temporal.
    """

    def __init__(self, encoders, inp_lens):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        x_stack = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_stack.append(encoder(x[:, -inp_len:, :], attn_mask=attn_mask))
        return torch.cat(x_stack, dim=-2)

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.05, activation="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask))
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class STN(nn.Module):
    """Sparse attention-based Transformer Network.

    Args:
        enc_in / dec_in: nº de variables de entrada (PM2.5 + meteorología).
        c_out: nº de series a predecir (1 = solo PM2.5).
        seq_len: lookback (paper: 24).
        label_len: nº de start tokens que ve el decoder (no está en el paper).
        out_len: horizonte de predicción (1 para h1, 6/12/24/48 para multi-paso).
        two_branch_encoder: True reproduce el encoder de dos ramas del paper.
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=2048, dropout=0.05, activation="gelu",
                 distil=True, mix=True, two_branch_encoder=True,
                 use_time_features=False, n_time_feats=4):
        super().__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.seq_len = seq_len

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout, use_time_features, n_time_feats)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout, use_time_features, n_time_feats)

        def make_encoder(n_layers):
            return Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            ProbAttention(False, factor, attention_dropout=dropout),
                            d_model, n_heads, mix=False,
                        ),
                        d_model, d_ff, dropout=dropout, activation=activation,
                    )
                    for _ in range(n_layers)
                ],
                [ConvLayer(d_model) for _ in range(n_layers - 1)] if distil else None,
                norm_layer=nn.LayerNorm(d_model),
            )

        if two_branch_encoder and e_layers >= 2:
            # Rama 0: secuencia completa con e_layers bloques.
            # Rama 1: media secuencia (inp_len = L/2) con e_layers-1 bloques.
            encoders = [make_encoder(e_layers), make_encoder(max(e_layers - 1, 1))]
            self.encoder = EncoderStack(encoders, inp_lens=[0, 1])
        else:
            self.encoder = make_encoder(e_layers)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout),
                        d_model, n_heads, mix=mix,
                    ),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout),
                        d_model, n_heads, mix=False,
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_dec, x_mark_enc=None, x_mark_dec=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        dec_out = self.projection(dec_out)
        return dec_out[:, -self.pred_len:, :]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ═════════════════════════ utilidades de entrenamiento ═════════════════════════
def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def sync():
    if DEVICE.type == "cuda": torch.cuda.synchronize()

def construir_modelo(n_feat, cfg, pred_len=1):
    return STN(enc_in=n_feat, dec_in=n_feat, c_out=1, seq_len=cfg["seq_len"], label_len=cfg["label_len"],
               out_len=pred_len, factor=cfg["factor"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
               e_layers=cfg["e_layers"], d_layers=cfg["d_layers"], d_ff=cfg["d_ff"],
               dropout=cfg["dropout"], two_branch_encoder=True).to(DEVICE)


def aplicar_regimen(model, regimen, estado_fuente=None):
    """Devuelve (grupos_de_parametros_a_optimizar, info).
    scratch   : todo aleatorio, todo entrenable.
    frozen    : cuerpo de Beijing congelado; solo se entrenan las capas de entrada (nº de variables distinto).
    ft_partial: encoder congelado; se afinan decoder + proyección + capas nuevas.
    ft_full   : todo entrenable, con tasa baja para el cuerpo preentrenado y tasa normal para lo nuevo."""
    sd = model.state_dict()
    cargados = {}
    if regimen != "scratch":
        cargados = {k: v for k, v in estado_fuente.items() if k in sd and sd[k].shape == v.shape}
        model.load_state_dict(cargados, strict=False)
    nombres_param = [n for n, _ in model.named_parameters()]
    es_nuevo = {n: (n not in cargados) for n in nombres_param}
    def entrenable(n):
        if regimen in ("scratch", "ft_full"): return True
        if es_nuevo[n]: return True
        if regimen == "ft_partial": return n.startswith(("decoder", "projection"))
        return False                                                   # frozen
    nuevos, cuerpo = [], []
    for n, p in model.named_parameters():
        p.requires_grad = entrenable(n)
        if p.requires_grad: (nuevos if es_nuevo[n] else cuerpo).append(p)
    info = {"regimen": regimen,
            "tensores_modelo": len(sd), "tensores_cargados": len(cargados),
            "params_totales": int(sum(p.numel() for p in model.parameters())),
            "params_cargados": int(sum(sd[k].numel() for k in cargados if k in dict(model.named_parameters()))),
            "params_entrenables": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "no_cargados": [n for n in nombres_param if es_nuevo[n]] if regimen != "scratch" else []}
    return nuevos, cuerpo, info


def _bn_congelados_a_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) and all(not p.requires_grad for p in m.parameters()):
            m.eval()


@torch.no_grad()
def predecir(model, enc, dec, fwd, batch=1024):
    model.eval(); out = []
    for i in range(0, len(enc), batch):
        out.append(fwd(model, enc[i:i + batch], dec[i:i + batch]).reshape(-1).float().cpu())
    return torch.cat(out).numpy()


def fit(model, nuevos, cuerpo, tr, va, fwd, lr_new, lr_body, epochs, patience, batch, seed, log=print):
    """tr, va = (enc, dec, y) tensores en DEVICE. Early stopping por MSE de validación + ReduceLROnPlateau
    (igual que la réplica). Devuelve (historia, mejor_epoca, segundos)."""
    grupos = [{"params": nuevos, "lr": lr_new}] + ([{"params": cuerpo, "lr": lr_body}] if cuerpo else [])
    opt = torch.optim.Adam(grupos)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    crit = nn.MSELoss()
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    todos = nuevos + cuerpo
    mejor, mejor_estado, mejor_ep, sin, hist = float("inf"), None, 0, 0, []
    sync(); t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train(); _bn_congelados_a_eval(model)
        perm = torch.randperm(len(tr[0]), generator=g).to(DEVICE)
        ls, gn = [], []
        for i in range(0, len(perm) - batch + 1, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(fwd(model, tr[0][b], tr[1][b]), tr[2][b])
            loss.backward()
            gn.append(float(torch.nn.utils.clip_grad_norm_(todos, 1.0)))
            opt.step(); ls.append(loss.item())
        pv = predecir(model, va[0], va[1], fwd)
        vl = float(np.mean((pv - va[2].reshape(-1).cpu().numpy()) ** 2))
        sched.step(vl)
        hist.append({"epoca": ep, "train_mse": float(np.mean(ls)), "val_mse": vl, "grad_norm": float(np.mean(gn))})
        log(f"  ép {ep:3d} | train {hist[-1]['train_mse']:.5f} | val {vl:.5f} | |grad| {hist[-1]['grad_norm']:.3f}")
        if vl < mejor - 1e-6:
            mejor, mejor_ep, sin = vl, ep, 0
            mejor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            sin += 1
            if sin >= patience: log(f"  early stopping en ép {ep} (mejor {mejor_ep})"); break
    sync(); seg = time.perf_counter() - t0
    model.load_state_dict(mejor_estado)
    return hist, mejor_ep, seg


class LSTMBase(nn.Module):
    """Línea base recurrente mínima sobre las mismas ventanas (no usa el decoder)."""
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        o, _ = self.lstm(x); return self.head(o[:, -1])[:, None, :]

fwd_stn = lambda m, e, d: m(e, d)
fwd_lstm = lambda m, e, d: m(e)
