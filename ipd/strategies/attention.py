"""EvolvedAttention: a 24-layer BERT-style encoder over the last 200 rounds, most recent first.

The window is a 200-token buffer padded with PAD and masked, matching the library's unpadded input.
Every act is a full forward pass (~5 GFLOP), so the bot is slow, most of all under round_robin's lax.switch.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ipd.bots import Bot
from ipd.game import C, D

MEMORY = 200
CLS, PAD = 0, 1
HEADS = 8
EPS = 1e-12
# token of (my move, their move): CD 2, DC 3, CC 4, DD 5
TOKEN = jnp.array([[4, 2], [3, 5]], jnp.int32)


def _layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + EPS) * w + b


def _encoder_layer(h, lw, bias):
    n, d = h.shape
    split = lambda x: x.reshape(n, HEADS, d // HEADS).transpose(1, 0, 2)
    q, k, v = (split(h @ lw[f"{m}_w"] + lw[f"{m}_b"]) for m in ("q", "k", "v"))
    scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(jnp.float32(d // HEADS)) + bias
    att = (jax.nn.softmax(scores, -1) @ v).transpose(1, 0, 2).reshape(n, d)
    att = _layer_norm(att @ lw["o_w"] + lw["o_b"] + h, lw["ln1_w"], lw["ln1_b"])
    mid = jax.nn.gelu(att @ lw["i_w"] + lw["i_b"], approximate=False)
    return _layer_norm(mid @ lw["out_w"] + lw["out_b"] + att, lw["ln2_w"], lw["ln2_b"])


def _attention_act(p, tokens, o, k):
    new = TOKEN[jnp.maximum(o.my_last, 0), jnp.maximum(o.their_last, 0)]
    tokens = jnp.where(o.t > 0, jnp.concatenate([new[None], tokens[:-1]]), tokens)
    ids = jnp.concatenate([jnp.array([CLS], jnp.int32), tokens])
    h = _layer_norm(p["state_emb"][ids] + p["pos_emb"], p["emb_ln_w"], p["emb_ln_b"])
    bias = jnp.where(ids != PAD, 0.0, jnp.finfo(jnp.float32).min)
    h, _ = lax.scan(lambda h, lw: (_encoder_layer(h, lw, bias), None), h, p["layers"])
    pooled = jnp.tanh(h[0] @ p["pool_w"] + p["pool_b"])
    logit = pooled @ p["act_w"] + p["act_b"]
    return jnp.where(jax.nn.sigmoid(logit) < 0.5, C, D), tokens


def from_axelrod():
    """Reads the trained weights from axelrod's data/model_attention.pth (lazy import, needs torch)."""
    from axelrod.load_data_ import load_attention_model_weights

    w = {k: v.numpy() for k, v in load_attention_model_weights().items()}
    names = {"q": "attention.self.query", "k": "attention.self.key", "v": "attention.self.value",
             "o": "attention.output.dense", "i": "intermediate.dense", "out": "output.dense"}
    n_layers = 1 + max(int(k.split(".")[2]) for k in w if k.startswith("encoder.layer."))
    layer = lambda i, key: w[f"encoder.layer.{i}.{key}"]
    layers = {}
    for short, full in names.items():  # torch Linear stores [out, in]
        layers[f"{short}_w"] = np.stack([layer(i, full + ".weight").T for i in range(n_layers)])
        layers[f"{short}_b"] = np.stack([layer(i, full + ".bias") for i in range(n_layers)])
    for short, full in {"ln1": "attention.output.LayerNorm", "ln2": "output.LayerNorm"}.items():
        layers[f"{short}_w"] = np.stack([layer(i, full + ".weight") for i in range(n_layers)])
        layers[f"{short}_b"] = np.stack([layer(i, full + ".bias") for i in range(n_layers)])
    params = {
        "state_emb": w["embeddings.game_state_embeddings.weight"],
        "pos_emb": w["embeddings.position_embeddings.weight"][: MEMORY + 1],
        "emb_ln_w": w["embeddings.LayerNorm.weight"], "emb_ln_b": w["embeddings.LayerNorm.bias"],
        "layers": layers,
        "pool_w": w["pooler.dense.weight"].T, "pool_b": w["pooler.dense.bias"],
        "act_w": w["action.weight"][0], "act_b": w["action.bias"][0],
    }
    return Bot("EvolvedAttention", lambda p, k: jnp.full(MEMORY, PAD, jnp.int32), _attention_act,
               params=jax.tree.map(jnp.asarray, params), memory_depth=MEMORY)


evolved_attention = from_axelrod()
STRATEGIES = [evolved_attention]
