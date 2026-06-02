"""
Qwen3-VL inference test using the same clinical prompts as inference_gemeni.py.

Usage:
    python -m qwen.test_inference
    python -m qwen.test_inference --model 8b-thinking --video_type mesh --samples 3
    python -m qwen.test_inference --model 30b-thinking --video_type keypoints --samples 5

Notes on Instruct vs Thinking:
    - Instruct variants answer directly. We use the Qwen-recommended sampling
      params (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5).
    - Thinking variants emit a <think>...</think> trace before the answer.
      They MUST NOT use greedy decoding (it causes repetition loops); we use
      the Qwen-recommended params (temp 1.0, top_p 0.95, top_k 20).
    Whether a run is "thinking" is derived from the chosen model, not a flag,
    so the sampling config can never drift out of sync with the checkpoint.
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen.dataset import NeuroDxVideoDataset, build_qwen_messages
from qwen.prompts import SYSTEM_PROMPT, build_user_prompt

MODELS = {
    "8b":           "Qwen/Qwen3-VL-8B-Instruct",
    "8b-thinking":  "Qwen/Qwen3-VL-8B-Thinking",
    "30b":          "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "30b-thinking": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "32b":          "Qwen/Qwen3-VL-32B-Instruct",
    "32b-thinking": "Qwen/Qwen3-VL-32B-Thinking",
}
DEFAULT_MODEL = "8b"
RESULTS_DIR = Path("results")

# Qwen-recommended sampling configs (from the official model cards / repo).
SAMPLING = {
    "instruct": dict(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
    ),
    "thinking": dict(
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
    ),
}


def is_thinking_model(model_id: str) -> bool:
    return "thinking" in model_id.lower()


def extract_json(text: str):
    """Pull the last JSON object out of a multi-stage response."""
    matches = list(re.finditer(r"\{[\s\S]*\}", text))
    if not matches:
        return None, "no JSON block found"
    try:
        return json.loads(matches[-1].group(0)), None
    except json.JSONDecodeError as e:
        return None, str(e)


def _extract_videos_and_fps(messages: list[dict]) -> tuple[list[str], list[float]]:
    paths, fps_list = [], []
    for msg in messages:
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "video":
                paths.append(part["video"])
                fps_list.append(part.get("fps", 1.0))
    return paths, fps_list


def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.eval()
    print(f"Loaded on: {next(model.parameters()).device}")
    return model, processor


def run_inference(model, processor, messages: list[dict], thinking: bool) -> tuple[str, str | None]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    video_paths, fps_list = _extract_videos_and_fps(messages)
    fps_kwarg = fps_list[0] if len(fps_list) == 1 else fps_list if fps_list else None

    inputs = processor(
        text=[text],
        videos=video_paths or None,
        fps=fps_kwarg,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Thinking traces need a much larger budget; the answer is appended after them.
    max_new_tokens = 8192 if thinking else 2048
    gen_kwargs = SAMPLING["thinking" if thinking else "instruct"]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    full_text = processor.decode(generated, skip_special_tokens=False)

    think_content = None
    if "<think>" in full_text and "</think>" in full_text:
        think_start = full_text.index("<think>") + len("<think>")
        think_end = full_text.index("</think>")
        think_content = full_text[think_start:think_end].strip()
        response = full_text[think_end + len("</think>"):].strip()
        # Strip any trailing special tokens left after skip_special_tokens=False.
        response = re.sub(r"<\|[^|]*\|>", "", response).strip()
    else:
        # No think block (Instruct model, or Thinking model that didn't emit one).
        response = processor.decode(generated, skip_special_tokens=True).strip()

    return response, think_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS),
                        help="Model shorthand: " + ", ".join(f"{k}={v}" for k, v in MODELS.items()))
    parser.add_argument("--video_type", default="clip",
                        choices=["clip", "mesh", "keypoints", "mesh_multiview"])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--save", action="store_true", help="Save results to results/")
    args = parser.parse_args()

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)

    model_id = MODELS[args.model]
    thinking = is_thinking_model(model_id)
    print(f"Mode: {'THINKING' if thinking else 'INSTRUCT'}  "
          f"(sampling: {SAMPLING['thinking' if thinking else 'instruct']})")

    model, processor = load_model(model_id)
    user_prompt = build_user_prompt(args.video_type)

    ds = NeuroDxVideoDataset("data/final", video_type=args.video_type)
    samples = list(ds)[:args.samples]

    sep = "=" * 70
    rows = []

    for i, sample in enumerate(samples):
        print(f"\n{sep}")
        print(f"Sample {i+1}/{len(samples)}: {sample['sample_id']}")
        print(f"  Label : {sample['label_name'].upper()} (class {sample['label']})")
        print(f"  Video : {sample['video_path']}")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += build_qwen_messages(
            sample, user_prompt, fps=args.fps, max_frames=args.max_frames
        )
        selected_fps = messages[-1]["content"][0]["fps"]
        print(f"  FPS   : {selected_fps:.2f}\n")

        response, think = run_inference(model, processor, messages, thinking=thinking)
        parsed, parse_err = extract_json(response)

        if think:
            wrapped = textwrap.fill(think[:500] + ("…" if len(think) > 500 else ""),
                                    width=80, initial_indent="    ", subsequent_indent="    ")
            print(f"[Thinking (truncated)]\n{wrapped}\n")

        # Print Stage 1 + 2 (prose before the JSON block)
        prose = re.split(r"\{[\s\S]*\}", response)[0].strip()
        if prose:
            print("[Stages 1 & 2]")
            for line in prose.splitlines():
                print("  " + line)

        # Print parsed JSON summary
        if parsed:
            print("\n[Stage 3 — Parsed]")
            print(f"  primary_diagnosis  : {parsed.get('primary_diagnosis')}")
            print(f"  diagnosis_confidence: {parsed.get('diagnosis_confidence')}")
            print(f"  evidence_tally     : {parsed.get('evidence_tally')}")
            print(f"  probabilities      : {parsed.get('probabilities')}")
            print(f"  primary_evidence   : {parsed.get('primary_evidence', '')[:120]}")
            counter = parsed.get("counter_evidence", [])
            if counter:
                print(f"  counter_evidence   : {counter[0][:100]}")
        else:
            print(f"\n[JSON parse failed: {parse_err}]")
            print(response[-800:])

        # Accumulate summary row
        true_label = sample["label_name"].upper()
        if parsed:
            probs = parsed.get("probabilities") or {}
            tally = parsed.get("evidence_tally") or {}
            predicted = (parsed.get("primary_diagnosis") or "").upper()
            rows.append({
                "sample_id": sample["sample_id"],
                "true_label": true_label,
                "predicted": predicted,
                "correct": predicted == true_label,
                "confidence": parsed.get("diagnosis_confidence", ""),
                "prob_MS": probs.get("MS", ""),
                "prob_PD": probs.get("PD", ""),
                "prob_Stroke": probs.get("Stroke", ""),
                "tally_MS": tally.get("MS", ""),
                "tally_PD": tally.get("PD", ""),
                "tally_Stroke": tally.get("Stroke", ""),
            })
        else:
            rows.append({
                "sample_id": sample["sample_id"],
                "true_label": true_label,
                "predicted": "PARSE_ERROR",
                "correct": False,
                "confidence": "",
                "prob_MS": "", "prob_PD": "", "prob_Stroke": "",
                "tally_MS": "", "tally_PD": "", "tally_Stroke": "",
            })

        if args.save:
            clip_id = sample["sample_id"]
            out = RESULTS_DIR / f"{clip_id}__{args.video_type}__{args.model}.json"
            out.write_text(json.dumps({
                "sample_id": sample["sample_id"],
                "label": sample["label_name"],
                "video_type": args.video_type,
                "model": model_id,
                "thinking": thinking,
                "think_trace": think,
                "full_response": response,
                "parsed_json": parsed,
                "parse_error": parse_err,
            }, indent=2))
            print(f"\n  Saved: {out}")

    # --- Summary table ---
    print(f"\n{sep}")
    print("SUMMARY TABLE")
    print(sep)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if not df.empty:
        n_correct = df["correct"].sum()
        print(f"\nAccuracy: {n_correct}/{len(df)} correct")

    if args.save and not df.empty:
        csv_out = RESULTS_DIR / f"summary__{args.video_type}__{args.model}.csv"
        df.to_csv(csv_out, index=False)
        print(f"Summary saved: {csv_out}")

    print(f"\n{sep}")


if __name__ == "__main__":
    main()