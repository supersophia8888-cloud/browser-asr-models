# Browser ASR model sizes

A machine-readable catalog of speech-recognition models that can run **inside a web browser** — ONNX exports used with `transformers.js` / `onnxruntime-web` — with the download size a real visitor actually pays for.

The question this answers: *if I ship Whisper in a browser tab, how many megabytes will a visitor download before they get a transcript?*

## The data

- [`data/browser-asr-models.csv`](data/browser-asr-models.csv) — 48 rows, 9 models
- [`data/browser-asr-models.json`](data/browser-asr-models.json) — same rows as JSON

Every number is fetched live from the Hugging Face API by [`scripts/build-catalog.py`](scripts/build-catalog.py). Nothing is hand-typed, and each row carries the exact API URL it came from in the `source` column. Re-run the script to refresh.

```bash
python scripts/build-catalog.py
```

## How `download_mib` is calculated

A browser does not download a whole model repository. For a given quantisation it fetches exactly two files:

```
encoder_model<quant>.onnx  +  decoder_model_merged<quant>.onnx
```

`download_mib` is those two files added together, in MiB (1 MiB = 1,048,576 bytes). This is deliberately **not** the sum of everything in the repository — that number is typically 10–30× larger and is not what a visitor experiences.

Only quantisations a browser build can realistically ship are included: `int8`, `uint8`, `quantized`, `q4`, `q4f16`, `fp16`. The `fp32` and `bnb4` variants present in most of these repositories are left out.

## What is in the catalog

| model | smallest quantisation | size | multilingual |
|---|---|---|---|
| moonshine-tiny | int8 | 26.8 MiB | no (English only) |
| whisper-tiny | int8 | 39.0 MiB | yes |
| moonshine-base | int8 | 60.0 MiB | no (English only) |
| whisper-base | int8 | 73.3 MiB | yes |
| whisper-small | int8 | 237.5 MiB | yes |
| whisper-large-v3-turbo | q4f16 | 537.4 MiB | yes |

`int8` / `uint8` / `quantized` are three names for the same 8-bit quantisation and resolve to near-identical byte counts; the catalog keeps all three because that is how they are actually named in the repositories.

The `.en` repositories (`whisper-tiny.en`, `whisper-base.en`, `whisper-small.en`) are English-only builds. Their file sizes are essentially identical to the multilingual ones — the English-only models are *faster and more accurate on English*, not smaller.

## What is not in the catalog

**Accuracy.** Download size is a fact you can read off an API; accuracy is not — it depends on your audio, your language and your quantisation. Measured word-error-rate numbers on a fixed 8-clip set, with the scoring scripts and the raw transcripts, are published separately at <https://clipquill.com/video-to-text-accuracy/> — that page is the reference for accuracy, this repository is the reference for size.

Also out of scope: models that need a GPU with more memory than a browser tab can get, and anything requiring a server round-trip.

## License

MIT for the script; the catalog data in `data/` is CC0-1.0. Hugging Face model files carry their own licenses — check the individual repository before shipping a model.
