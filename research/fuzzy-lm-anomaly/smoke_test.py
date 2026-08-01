"""Smoke test: load SmolLM2-360M-Instruct, generate, and inspect hidden-state shapes.

Confirms the capture path works on 12GB before we build the full harness.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    cfg = model.config
    print(f"model      : {MODEL_ID}")
    print(f"params     : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"layers     : {cfg.num_hidden_layers}")
    print(f"hidden     : {cfg.hidden_size}")
    print(f"vocab      : {cfg.vocab_size}")

    msgs = [{"role": "user", "content": "Who won the 1997 Nobel Prize in Interpretive Dance?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to("cuda")

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=48,
            do_sample=False,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

    gen_ids = out.sequences[0, enc["input_ids"].shape[1]:]
    print("\n--- generation ---")
    print(tok.decode(gen_ids, skip_special_tokens=True))

    print("\n--- shapes ---")
    print(f"n_new_tokens        : {len(gen_ids)}")
    print(f"hidden_states steps : {len(out.hidden_states)}")
    print(f"layers per step     : {len(out.hidden_states[0])}")
    print(f"step0 layer0 shape  : {tuple(out.hidden_states[0][0].shape)}  (prompt pass)")
    print(f"step1 layer0 shape  : {tuple(out.hidden_states[1][0].shape)}  (single new token)")
    print(f"scores steps        : {len(out.scores)}, shape {tuple(out.scores[0].shape)}")
    print(f"\nvram alloc: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
