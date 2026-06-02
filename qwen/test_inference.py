"""
Qwen3-VL inference test using the same clinical prompts as inference_gemeni.py.

Usage:
    python -m qwen.test_inference
    python -m qwen.test_inference --video_type mesh --samples 3
    python -m qwen.test_inference --video_type keypoints --thinking --samples 5
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

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
RESULTS_DIR = Path("results")


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

    max_new_tokens = 4096 if thinking else 2048
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    full_text = processor.decode(generated, skip_special_tokens=False)

    think_content = None
    if "<think>" in full_text and "</think>" in full_text:
        think_start = full_text.index("<think>") + len("<think>")
        think_end = full_text.index("</think>")
        think_content = full_text[think_start:think_end].strip()
        response = full_text[think_end + len("</think>"):].strip()
    else:
        response = processor.decode(generated, skip_special_tokens=True).strip()

    return response, think_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--video_type", default="clip",
                        choices=["clip", "mesh", "keypoints", "mesh_multiview"])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--save", action="store_true", help="Save results to results/")
    args = parser.parse_args()

    if args.save:
        RESULTS_DIR.mkdir(exist_ok=True)

    model, processor = load_model(args.model)
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

        response, think = run_inference(model, processor, messages, thinking=args.thinking)
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
            out = RESULTS_DIR / f"{clip_id}__{args.video_type}.json"
            out.write_text(json.dumps({
                "sample_id": sample["sample_id"],
                "label": sample["label_name"],
                "video_type": args.video_type,
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
        csv_out = RESULTS_DIR / f"summary__{args.video_type}.csv"
        df.to_csv(csv_out, index=False)
        print(f"Summary saved: {csv_out}")

    print(f"\n{sep}")


if __name__ == "__main__":
    main()
