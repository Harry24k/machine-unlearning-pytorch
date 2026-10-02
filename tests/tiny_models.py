"""Tiny random-init models for CPU tests (no download: only the cached sshleifer/tiny-gpt2 tokenizer).

make_tiny_llava(out_dir)   LLaVA = 2-layer CLIP (32px, patch 8 -> 16 image tokens) + 2-layer Llama, saved in HF
                           format with a LlavaProcessor + chat template.  Loads via AutoModelForImageTextToText.
make_tiny_llm(out_dir)     2-layer Llama text model + GPT-2 tokenizer with a chat template.
"""
from __future__ import annotations

import os

import torch

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: "
    "{% if m['content'] is string %}{{ m['content'] }}{% else %}"
    "{% for c in m['content'] %}{% if c['type'] == 'image' %}<image>{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}{% endfor %}"
    "{% endif %}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE
    return tok


def make_tiny_llm(out_dir: str, seed: int = 0) -> str:
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    tok = _tokenizer()
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                      pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    model = LlamaForCausalLM(cfg)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def make_tiny_llava(out_dir: str, seed: int = 0) -> str:
    from transformers import (CLIPImageProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                              LlavaForConditionalGeneration, LlavaProcessor)
    torch.manual_seed(seed)
    tok = _tokenizer()
    tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    image_token_id = tok.convert_tokens_to_ids("<image>")
    vcfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                            image_size=32, patch_size=8, projection_dim=32)
    tcfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                       num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                       pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    cfg = LlavaConfig(vision_config=vcfg, text_config=tcfg, image_token_index=image_token_id,
                      vision_feature_select_strategy="default", vision_feature_layer=-2,
                      pad_token_id=tok.pad_token_id)
    model = LlavaForConditionalGeneration(cfg)
    ip = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}, do_center_crop=True,
                            do_resize=True, do_normalize=True)
    proc = LlavaProcessor(image_processor=ip, tokenizer=tok, patch_size=8, vision_feature_select_strategy="default",
                          image_token="<image>", num_additional_image_tokens=1, chat_template=CHAT_TEMPLATE)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    proc.save_pretrained(out_dir)
    return out_dir


def random_image(seed: int = 0, size: int = 40):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def make_tiny_clip(out_dir: str, seed: int = 0) -> str:
    """2-layer CLIP (32px / patch 8, hidden 32) with the GPT-2 tokenizer -> CLIPRobModel.from_pretrained works."""
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTextConfig, CLIPVisionConfig
    torch.manual_seed(seed)
    tok = _tokenizer()
    tcfg = CLIPTextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                          max_position_embeddings=64, projection_dim=16, pad_token_id=tok.pad_token_id,
                          bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    vcfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=32,
                            patch_size=8, projection_dim=16)
    cfg = CLIPConfig(text_config=tcfg.to_dict(), vision_config=vcfg.to_dict(), projection_dim=16)
    model = CLIPModel(cfg)
    ip = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    proc = CLIPProcessor(image_processor=ip, tokenizer=tok)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    proc.save_pretrained(out_dir)
    return out_dir
