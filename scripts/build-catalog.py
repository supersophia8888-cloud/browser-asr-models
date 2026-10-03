#!/usr/bin/env python3
"""
Build data/browser-asr-models.csv and .json from the Hugging Face API.

Every number in the output is fetched live from huggingface.co — nothing is
hand-typed. Run this again to refresh the catalog.

    python scripts/build-catalog.py

Why the sizes look the way they do: a browser does not download a whole model
repository. It downloads one encoder file plus one merged decoder file for the
quantisation it picked. The "download" column is exactly those two files added
together, which is the number a user actually pays for over the network.
"""
import csv
import json
import os
import re
import urllib.request
from collections import defaultdict

REPOS = {
    "whisper-tiny":            "onnx-community/whisper-tiny",
    "whisper-base":            "onnx-community/whisper-base",
    "whisper-small":           "onnx-community/whisper-small",
    "whisper-large-v3-turbo":  "onnx-community/whisper-large-v3-turbo",
    "whisper-tiny.en":         "onnx-community/whisper-tiny.en",
    "whisper-base.en":         "onnx-community/whisper-base.en",
    "whisper-small.en":        "onnx-community/whisper-small.en",
    "moonshine-tiny":          "onnx-community/moonshine-tiny-ONNX",
    "moonshine-base":          "onnx-community/moonshine-base-ONNX",
}

# Quantisations a browser build can realistically pick. Everything else in the
# repo (fp32, bnb4, ...) is left out on purpose: too large to ship to a tab.
KEEP = {"int8", "uint8", "quantized", "q4", "q4f16", "fp16"}

API = "https://huggingface.co/api/models/{}?blobs=true"


def fetch(repo):
    req = urllib.request.Request(API.format(repo), headers={"User-Agent": "browser-asr-models-catalog"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def quantisations(info):
    """{quantisation: encoder_bytes + decoder_merged_bytes}"""
    files = {s["rfilename"]: (s.get("size") or 0) for s in info.get("siblings", [])}
    grouped = defaultdict(dict)
    for path, size in files.items():
        name = path.split("/")[-1]
        if not name.endswith(".onnx"):
            continue
        m = re.match(r"(encoder_model|decoder_model_merged)(?:_([a-z0-9]+))?\.onnx$", name)
        if not m:
            continue
        grouped[m.group(2) or "fp32"][m.group(1)] = size
    out = {}
    for quant, parts in grouped.items():
        if "encoder_model" in parts and "decoder_model_merged" in parts:
            out[quant] = parts["encoder_model"] + parts["decoder_model_merged"]
    return out


def main():
    rows = []
    for label, repo in REPOS.items():
        info = fetch(repo)
        for quant, total in quantisations(info).items():
            if quant not in KEEP:
                continue
            rows.append({
                "model": label,
                "hf_repo": repo,
                "quantisation": quant,
                "download_mib": round(total / 1048576, 1),
                "download_bytes": total,
                "hf_downloads_30d": info.get("downloads"),
                "hf_likes": info.get("likes"),
                "hf_last_modified": str(info.get("lastModified", ""))[:10],
                "source": f"https://huggingface.co/api/models/{repo}?blobs=true",
            })

    rows.sort(key=lambda r: (r["model"], r["download_bytes"]))

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "browser-asr-models.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(data_dir, "browser-asr-models.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"wrote {len(rows)} rows across {len(REPOS)} models")


if __name__ == "__main__":
    main()
