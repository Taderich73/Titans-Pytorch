#!/usr/bin/env python3
"""Convert native Titans checkpoints to HuggingFace format.

Usage:
    python scripts/convert_to_hf.py --checkpoint checkpoints/final.pt --output-dir ./hf_model
    python scripts/convert_to_hf.py --checkpoint checkpoints/final.pt --output-dir ./hf_model \
        --tokenizer gpt2 --push-to-hub user/repo-name
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from titans import TITANS_SCHEMA_VERSION
from titans.checkpoint import load_checkpoint
from titans.config import TitansConfig
from titans.hf.configuration import TitansMACConfig
from titans.hf.modeling import TitansMACForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Registry of model types -> (config_cls, model_cls)
# Extend this dict when adding MAG/MAL/LMM variants.
MODEL_REGISTRY: dict[str, tuple[type, type]] = {
    "mac": (TitansMACConfig, TitansMACForCausalLM),
}


CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)


def remap_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add ``model.`` prefix to native checkpoint keys.

    Native keys:  ``embed.weight``, ``blocks.0.memory...``
    HF keys:      ``model.embed.weight``, ``model.blocks.0.memory...``
    """
    return {f"model.{k}": v for k, v in state_dict.items()}


def convert_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    model_type: str = "mac",
    tokenizer_name: str | None = None,
    torch_dtype: str = "float32",
    add_chat_template: bool = False,
    push_to_hub: str | None = None,
    upload_model_code: bool = False,
) -> None:
    """Convert a native Titans checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to native .pt or .safetensors checkpoint.
        output_dir: Directory to write the HF model files to.
        model_type: Model variant key (default: "mac").
        tokenizer_name: HF tokenizer to include (e.g. "gpt2"). None to skip.
        torch_dtype: Dtype string for config metadata.
        add_chat_template: If True, add ChatML special tokens and template.
        push_to_hub: Hub repo ID to push to. None to skip.
        upload_model_code: If True, upload Python source for trust_remote_code.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type {model_type!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    config_cls, model_cls = MODEL_REGISTRY[model_type]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path, weights_only=False)

    # 2. Extract config
    if "config" in ckpt:
        titans_config = TitansConfig.from_dict(ckpt["config"])
    elif "titans_config" in ckpt:
        titans_config = TitansConfig.from_dict(ckpt["titans_config"])
    else:
        raise ValueError(
            "Checkpoint does not contain config metadata. "
            "Pass config values via CLI args."
        )

    # 3. Create HF config
    hf_config = config_cls.from_titans_config(titans_config)
    hf_config.torch_dtype = torch_dtype
    hf_config.architectures = [model_cls.__name__]
    hf_config.auto_map = {
        "AutoConfig": f"titans.hf.configuration.{config_cls.__name__}",
        "AutoModelForCausalLM": f"titans.hf.modeling.{model_cls.__name__}",
    }
    # Stamp the schema version explicitly so the emitted config.json
    # carries it even when an older TitansConfig was read from disk (the
    # kwarg default does this too, but being explicit keeps the
    # convert-script contract obvious when the output is inspected).
    hf_config.titans_schema_version = TITANS_SCHEMA_VERSION

    # 4. Remap state dict keys
    remapped = remap_state_dict_keys(ckpt["model"])

    # 5. Save config
    hf_config.save_pretrained(output_dir)
    logger.info(f"Saved config.json to {output_dir}")

    # 6. Save model weights as safetensors
    from safetensors.torch import save_file

    # Handle tied weights: clone shared tensors to avoid safetensors errors
    seen: dict[int, str] = {}
    prepared: dict[str, torch.Tensor] = {}
    for k, v in remapped.items():
        data_ptr = v.data_ptr()
        if data_ptr in seen:
            prepared[k] = v.clone().contiguous()
        else:
            seen[data_ptr] = k
            prepared[k] = v.contiguous()

    sf_path = output_path / "model.safetensors"
    save_file(prepared, sf_path)
    logger.info(f"Saved model.safetensors ({sf_path.stat().st_size / 1e6:.1f} MB)")

    # 7. Save tokenizer (optional)
    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if add_chat_template:
            tokenizer.add_special_tokens({
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            })
            tokenizer.chat_template = CHATML_TEMPLATE

        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved tokenizer files to {output_dir}")

    # 8. Save generation config
    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.8,
        top_k=50,
        top_p=1.0,
        do_sample=True,
        eos_token_id=titans_config.vocab_size - 1,
        pad_token_id=titans_config.vocab_size - 1,
    )
    gen_config.save_pretrained(output_dir)
    logger.info(f"Saved generation_config.json to {output_dir}")

    # 9. Push to Hub
    if push_to_hub is not None:
        import os

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(push_to_hub, exist_ok=True)

        if upload_model_code:
            src_dir = Path(__file__).resolve().parent.parent / "src" / "titans" / "hf"
            for py_file in src_dir.glob("*.py"):
                api.upload_file(
                    path_or_fileobj=str(py_file),
                    path_in_repo=py_file.name,
                    repo_id=push_to_hub,
                )

        api.upload_folder(
            folder_path=output_dir,
            repo_id=push_to_hub,
        )
        logger.info(f"Pushed to Hub: {push_to_hub}")

    logger.info("Conversion complete.")


def main() -> None:
    """Parse arguments and run conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Titans checkpoints to HuggingFace format."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to native .pt or .safetensors checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for HF model files",
    )
    parser.add_argument(
        "--model-type", type=str, default="mac",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model variant (default: mac)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="HF tokenizer name (e.g. gpt2). Omit to skip.",
    )
    parser.add_argument(
        "--torch-dtype", type=str, default="float32",
        help="Dtype for config metadata",
    )
    parser.add_argument(
        "--add-chat-template", action="store_true",
        help="Add ChatML special tokens and template to tokenizer",
    )
    parser.add_argument(
        "--push-to-hub", type=str, default=None,
        help="Hub repo ID to push to (e.g. user/model-name)",
    )
    parser.add_argument(
        "--upload-model-code", action="store_true",
        help="Upload Python source files for trust_remote_code",
    )
    args = parser.parse_args()

    convert_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_type=args.model_type,
        tokenizer_name=args.tokenizer,
        torch_dtype=args.torch_dtype,
        add_chat_template=args.add_chat_template,
        push_to_hub=args.push_to_hub,
        upload_model_code=args.upload_model_code,
    )


if __name__ == "__main__":
    main()
