# Gemini vs Emotify (GEMS) — Audio Emotion Distribution Estimation

This script batch-processes `.opus` music clips and asks a Gemini model (via **OpenRouter**) to estimate **GEMS** emotion **mean** and **std** (as if `N` listeners gave binary 0/1 votes). It then compares Gemini outputs against your **ground-truth** (Emotify/human) statistics and generates two comparison CSVs.

---

## What it does

For each `data/raw/song_*.opus`:

1. Loads **listener count** `N` for that song (hardcoded dictionary)
2. Sends audio + a strict JSON-only prompt to Gemini
3. Saves predictions incrementally to `gemini_output.json`
4. After processing, creates:
   - `means_comparison.csv` (Gemini vs Emotify vs Diff for **means**)
   - `stds_comparison.csv` (Gemini vs Emotify vs Diff for **stds**)

**Diff is computed as:** `Gemini − Emotify` (rounded to 4 decimals)

---

## Requirements

- Python **3.10+**
- An OpenRouter-compatible API key in environment variable:
  - `OPENROUTER_API_KEY`

Python packages:
- `pandas`
- `requests`

Install:
```bash
pip install pandas requests
```

---

## Project layout (expected)

```
project/
├─ script.py                  # (your code)
├─ truth.json                 # Ground-truth stats (Emotify)
├─ gemini_output.json         # (created) model outputs
├─ means_comparison.csv       # (created)
├─ stds_comparison.csv        # (created)
└─ data/
   └─ raw/
      ├─ song_1.opus
      ├─ song_2.opus
      └─ ...
```

---

## Environment setup

### macOS / Linux
```bash
export OPENROUTER_API_KEY="YOUR_KEY"
```

### Windows PowerShell
```powershell
$env:OPENROUTER_API_KEY="YOUR_KEY"
```

---

## Ground truth format: `truth.json`

`truth.json` must be a dictionary keyed by song id (e.g., `"song_1"`), where each emotion contains:

- `mean`: float in `[0, 1]`
- `std`: float in `[0, 0.5]` (as in your dataset)

Example:
```json
{
  "song_1": {
    "amazement": { "mean": 0.1458, "std": 0.3529 },
    "solemnity": { "mean": 0.3333, "std": 0.4714 },
    "tenderness": { "mean": 0.2083, "std": 0.4061 },
    "nostalgia": { "mean": 0.2917, "std": 0.4545 },
    "calmness": { "mean": 0.625, "std": 0.4841 },
    "power": { "mean": 0.0208, "std": 0.1428 },
    "joyful_activation": { "mean": 0.0833, "std": 0.2764 },
    "tension": { "mean": 0.0625, "std": 0.2421 },
    "sadness": { "mean": 0.3125, "std": 0.4635 }
  }
}
```

**Important:** The script expects these exact 9 emotion keys:
- `amazement`
- `solemnity`
- `tenderness`
- `nostalgia`
- `calmness`
- `power`
- `joyful_activation`
- `tension`
- `sadness`

---

## Listener counts

Listener counts are embedded in the script as a dictionary:
```python
listeners = {"song_1": 48, "song_2": 47, ...}
```

This `N` is injected into the prompt per song.

---

## Model + API

The script calls:
- `OPENROUTER_URL = "https://llmfoundry.straive.com/openrouter/v1/chat/completions"`
- `MODEL_NAME = "google/gemini-3-pro-preview"`

You can change `MODEL_NAME` to any OpenRouter-available Gemini model.

---

## Audio format note (IMPORTANT)

Your files are `.opus`, but the request payload sets:
```json
{"format": "audio/ogg"}
```

Depending on your gateway/provider, this may still work (since Opus is commonly stored in an Ogg container), **but** if you hit errors, update the format to match your actual container, for example:

- If your `.opus` files are Ogg/Opus: keep `"audio/ogg"`
- If your provider expects explicit Opus: try `"audio/opus"`
- If you convert to `.ogg`: keep `"audio/ogg"`

If you need conversion examples (ffmpeg):
```bash
# Convert OPUS to OGG (Opus codec inside Ogg container)
ffmpeg -i input.opus -c copy output.ogg

# Or decode/re-encode if needed
ffmpeg -i input.opus -c:a libopus -b:a 96k output.ogg
```

---

## How to run

From the project root:

```bash
python script.py
```

What you should see:
- It loads `truth.json`
- It optionally loads `gemini_output.json` (if it exists) to resume
- It processes unprocessed songs in `data/raw/`
- It writes outputs

---

## Outputs

### 1) `gemini_output.json` (incremental)
Saved after each successful API call to support resume if interrupted.

Shape:
```json
{
  "song_1": {
    "amazement": {"mean": 0.25, "std": 0.433},
    "...": {}
  }
}
```

### 2) `means_comparison.csv`
Rows are grouped per song:
- `source=gemini`  (Gemini means)
- `source=emotify` (Truth means)
- `source=diff`    (Gemini − Emotify)

Columns:
- `song, metric, source, amazement, solemnity, ... sadness`

### 3) `stds_comparison.csv`
Same pattern, but for std.

---

## Resume behavior

If `gemini_output.json` exists, the script will:
- Load it
- Skip any already processed songs
- Continue where it left off

Delete `gemini_output.json` if you want a clean re-run.

---

## Troubleshooting

### `OPENROUTER_API_KEY not set`
Set the environment variable and rerun.

### API status != 200
The script raises:
```python
raise Exception(f"API error: {response.status_code}")
```
To debug quickly, print `response.text` in the error path.

### Model returns non-JSON
The script attempts a regex extract of the first `{...}` block.
If output is still malformed, consider adding:
- retries on JSON decode failure
- a stricter system message
- response_format / JSON mode (if your gateway supports it)

---

## Notes / Future improvements (optional)

- Add retries with exponential backoff for transient API failures
- Add rate limiting (sleep) to avoid throttling
- Validate that all 9 emotions are present in the model output before saving
- Compute summary metrics (MAE, bias per emotion, top-k alignment) directly in the script

---

## License / Usage

Internal evaluation / benchmarking utility.
