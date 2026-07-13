from __future__ import annotations

import re


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def classify_tensor(tensor_name: str) -> tuple[int | None, str]:
    """
    purpose: Hugging Face 계열 tensor_name에서 layer index와 module type을 추출한다.
    input: safetensors 헤더에서 읽은 tensor_name 문자열.
    processing: `layers.{N}.` 패턴으로 layer_idx를 찾고 attention/MLP/norm/lm_head 패턴을 분류한다.
    return/side effects: `(layer_idx, module_type)` tuple을 반환하며 외부 상태는 변경하지 않는다.
    """

    name = tensor_name.lower()
    layer_match = _LAYER_RE.search(name)
    layer_idx = int(layer_match.group(1)) if layer_match else None
    if "lm_head" in name or "output_projection" in name:
        return layer_idx, "lm_head"
    if "embed_tokens" in name or "word_embeddings" in name or "wte" in name:
        return layer_idx, "embedding"
    if "q_proj" in name or "query" in name or ".wq" in name:
        return layer_idx, "attn_q"
    if "k_proj" in name or "key" in name or ".wk" in name:
        return layer_idx, "attn_k"
    if "v_proj" in name or "value" in name or ".wv" in name:
        return layer_idx, "attn_v"
    if "o_proj" in name or "out_proj" in name or ".wo" in name:
        return layer_idx, "attn_o"
    if "gate_proj" in name or "gate" in name or "w1" in name:
        return layer_idx, "mlp_gate"
    if "up_proj" in name or "dense_h_to_4h" in name or "w3" in name:
        return layer_idx, "mlp_up"
    if "down_proj" in name or "dense_4h_to_h" in name or "w2" in name:
        return layer_idx, "mlp_down"
    if "norm" in name or "ln_" in name or "layernorm" in name:
        return layer_idx, "norm"
    return layer_idx, "other"


def classify_tensor_name(tensor_name: str) -> str:
    """
    purpose: 기존 호출부 호환을 위해 tensor_name의 module type만 반환한다.
    input: safetensors 헤더에서 읽은 tensor_name 문자열.
    processing: `classify_tensor`를 호출하고 module_type만 추출한다.
    return/side effects: module_type 문자열을 반환하며 외부 상태는 변경하지 않는다.
    """

    return classify_tensor(tensor_name)[1]
