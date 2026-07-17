# Eval Reproducibility (webeval)

> **Note:** This document covers the `webeval/` pipeline that produced the
> published Fara-7B benchmark numbers (WebVoyager, Online-Mind2Web,
> WebTailBench). It predates the Fara-1.5 release and will be updated as the
> repo is refactored to the Fara-1.5 evaluation stack. For the trajectory
> on-disk format, see [`webeval/README.md`](../webeval/README.md).

Agentic evaluations on live websites present unique challenges due to day-to-day changes. We implement several measures to ensure reliable and comparable evaluations:

**BrowserBase Integration**
We employ BrowserBase to manage browser session hosting, enabling reliable browser instance management.

**Time-sensitive Task Updates**
Tasks in benchmarks like WebVoyager can become stale or impossible. We:
- Removed ~48 impossible tasks from the original WebVoyager benchmark
- Updated ~50 tasks with future dates to keep them achievable
- Example: *"Search for a hotel in Bali from Jan 1 to Jan 4, 2024"* → *"Search for a hotel in Bali from Jan 1 to Jan 4, 2026"*
- Our updated WebVoyager benchmark is available at `webeval/data/webvoyager/WebVoyager_data_08312025.jsonl`

**Environment Error Handling**
Browser errors (connection drops, page timeouts) are handled robustly:
- Trajectories are retried up to 5 times when environment errors occur
- Complete yet incorrect trajectories are never retried
- Each retry starts with a fresh browser session, with no retained state

**Step Budget**
Each trajectory is capped at a maximum of 100 actions across all online benchmarks. Trajectories exceeding this budget without choosing to stop are considered incorrect.

## WebEval Package Installation

```bash
conda create --name fara_webeval python=3.12
conda activate fara_webeval

# Install fara package (with vllm extras for GPU hosting)
pip install -e .[vllm]

# Install webeval
cd webeval
pip install -e .

# Install playwright
playwright install
```

The webeval package no longer depends on `autogen-core` / `autogen-ext` —
all chat completion clients are vendored under `webeval/src/webeval/oai_clients/`
(see `GracefulRetryClient`, `OpenAIClientWrapper`, `AzureOpenAIClientWrapper`,
etc.). You no longer need to clone or install the autogen submodule.

> **Always activate the `fara_webeval` env before running any of the eval
> scripts below.** It pins `vllm==0.10.0` + `torch==2.7.1`; running with
> a newer `vllm` (≥ 0.19) under `torch` ≥ 2.10 trips a known
> `torch._dynamo.symbolic_convert` crash during CUDA-graph capture, which
> can be worked around with `--enforce_eager` but at a meaningful
> throughput cost. Stick to the pinned env.

## Running Evaluations

Navigate to the scripts directory:

```bash
cd webeval/scripts
```

Make sure you set a valid OpenAI GPT-4o endpoint in `endpoint_configs_gpt4o/dev` in order to run the WebVoyager LLM-as-a-judge! 

**Option 1: Self-hosted vLLM**

```bash
python webvoyager.py --model_url /path/where/you/want/to/download/model/ --model_port 5000 --eval_oai_config ../endpoint_configs_gpt4o/dev/ --out_url /path/to/save/eval/files --device_id 0,1 --processes 1 --run_id 1 --max_rounds 100
python om2w.py --model_url /path/where/you/want/to/download/model/ --model_port 5000 --eval_oai_config ../endpoint_configs_o4/dev/ --eval_model o4-mini --out_url /path/to/save/eval/files --device_id 0,1 --processes 1 --run_id 1 --max_rounds 100

# WebTailBench almost always needs --browserbase: a meaningful share of
# the benchmark's task websites (airlines, retailers, ticketing, …)
# block bot traffic from a vanilla playwright browser. Without
# --browserbase you'll see a high rate of trajectories that abort on
# Page.goto / navigation / captcha errors. Set BROWSERBASE_API_KEY and
# BROWSERBASE_PROJECT_ID in the environment first.
export BROWSERBASE_API_KEY=<your_browserbase_api_key>
export BROWSERBASE_PROJECT_ID=<your_browserbase_project_id>

# --success controls which Universal Verifier signal counts as the
# top-line score: ``outcome`` (default; binary outcome_success — what the
# Fara numbers in the README are reported against), ``process``
# (rubric_is_success := rubric_score >= --rubric_score_threshold; a more
# lenient gate, expect slightly higher numbers), or ``both``.
python webtailbench.py \
    --model_url /path/to/Fara/model_checkpoints \
    --model_port 5000 \
    --device_id 0,1 \
    --eval_oai_config ../../endpoint_configs/judge_active/prod \
    --judge_eval_model gpt-5.2 \
    --judge_o4_eval_model o4-mini \
    --rubric_score_threshold 0.8 \
    --success outcome \
    --out_url /path/to/Fara/eval \
    --processes 4 \
    --run_id 1 \
    --max_rounds 100 \
    --browserbase

python verify_trajectories.py \
    --input /path/to/Fara/eval/runs/.../<benchmark>/<run_id>/traj \
    --task-data ../path/to/om2w/Online_Mind2Web_06042025.json \
    --task-data-format om2w \
    --eval-config ../../endpoint_configs/judge_active/prod \
    --judge-model gpt-5.2 --o4mini-model o4-mini \
    --processes 8
```

**Option 2: Azure Foundry Deployment**

Deploy [Fara1.5-9B on Foundry endpoint(s)](https://ai.azure.com/catalog/models/Fara1.5-9B), then place endpoint URLs and keys in JSONs under `endpoint_configs/`:

```bash
python webvoyager.py --model_endpoint ../../endpoint_configs/ --eval_oai_config ../endpoint_configs_gpt4o/dev/ --out_url /path/to/save/eval/files --processes 1 --run_id 1_endpoint --max_rounds 100
python om2w.py --model_endpoint ../../endpoint_configs/ --eval_oai_config ../endpoint_configs_o4/dev/ --eval_model o4-mini --out_url /path/to/save/eval/files --processes 1 --run_id 1_endpoint --max_rounds 100
python webtailbench.py --model_endpoint ../../endpoint_configs/ --eval_oai_config ../../endpoint_configs/judge_active/prod --judge_eval_model gpt-5.2 --judge_o4_eval_model o4-mini --out_url /path/to/Fara/eval --processes 1 --run_id 1_endpoint --max_rounds 100
```

### Notes


- We use the same LLM-as-a-judge prompts and model (GPT-4o) as WebVoyager, hence the `--eval_oai_config` argument
- Set `--browserbase` for browser session management (requires exported API key and project ID environment variables)
- Avoid overloading a single vLLM deployment with more than ~10 concurrent processes due to known issues
- See debugging output in `fara/webeval/scripts/stdout.txt`

---

## Analyzing Evaluation Results

### Evaluation Output Structure

Evaluation results are stored under `--out_url` in folders organized by:
- Model name
- Dataset
- Username
- Run ID

Example path:
```
/runs/WebSurfer-fara-100-max_n_images-3/fara1.5-9b/<username>/WebVoyager_WebVoyager_data_08312025.jsonl/<run_id>
```

Each evaluation folder contains:
- `gpt_eval/` - LLM-as-a-judge evaluation results
- `traj/` - Per-task trajectory subdirectories containing:
  - `*-final_answer.json` (e.g., `Amazon--1_final_answer.json`) - `<no_answer>` indicates abortion or step budget exceeded
  - `scores/*_eval.json` - LLM judge scores (`gpt_eval.json` for WebVoyager, `WebJudge_Online_Mind2Web_eval-3.json` for Online-Mind2Web)
  - `web_surfer.log` - Action history and errors
  - `screenshot_X.png` - Screenshots captured before each action X
  - `times.json` - contains start and end times of task
  - `core.log` - contains high level logs like if the trajectory needs to be started or is already cached/done, the evaluation score, duration, and errors encountered

### Running Analysis

Use the analysis notebook to compute metrics:

```bash
cd webeval/scripts/analyze_eval_results/
jupyter notebook analyze.ipynb
```

The script:
- Identifies trajectories aborted mid-execution and diagnostic reasons
- Computes average scores across non-aborted trajectories
- Distinguishes between aborted trajectories (errors during sampling) and completed trajectories (with terminate() call or step budget exceeded)

To re-run failed tasks, execute the evaluation script again with the same `run_id` and `username` - it will skip non-aborted tasks.

<details>
<summary>Example WebVoyager GPT Eval Result</summary>

```json
{
  "score": 1.0,
  "gpt_response_text": "To evaluate the task, we need to verify if the criteria have been met:\n\n1. **Recipe Requirement**: A vegetarian lasagna recipe with zucchini and at least a four-star rating.\n\n2. **Search and Results**:\n   - The screenshots show that the search term used was \"vegetarian lasagna zucchini.\"\n   - Among the search results, \"Debbie's Vegetable Lasagna\" is prominently featured.\n   \n3. **Evaluation of the Recipe**:\n   - Rating: \"Debbie's Vegetable Lasagna\" has a rating of 4.7, which satisfies the requirement of being at least four stars.\n   - The presence of zucchini in the recipe is implied through the search conducted, though the screenshots do not explicitly show the ingredients list. However, the result response confirms the match to the criteria.\n\nGiven the information provided, the task seems to have fulfilled the requirement of finding a vegetarian lasagna recipe with zucchini and a four-star rating or higher. \n\n**Verdict: SUCCESS**"
}
```

</details>

<details>
<summary>Example of web_surfer.log (action history)</summary>

```json
{"message": "\nThought #1: To begin locating the closest GameStop store, I will click on the 'Trade-In' button, which typically leads to a store locator feature on retail websites.\nAction #1: executing tool 'left_click' with arguments {\"action\": \"left_click\", \"coordinate\": [1074, 37]}", "timestamp": "2025-11-25T22:23:42.788320", "type": "OtherEvent"}
{"message": "Observation#1: I clicked at coordinates (1083.0252100840337, 37.16517857142858).", "timestamp": "2025-11-25T22:23:47.616613", "type": "OtherEvent"}
{"message": "\nThought #2: Since access to Gamestop's website has been blocked, I will search for alternative sources of store location and hours for GameStop locations near zip code 90028. This could include cached or third-party website information.\nAction #2: executing tool 'web_search' with arguments {\"action\": \"web_search\", \"query\": \"GameStop store locator zip code 90028\"}", "timestamp": "2025-11-25T22:23:50.579702", "type": "OtherEvent"}
{"message": "Observation#2: I typed 'GameStop store locator zip code 90028' into the browser search bar.", "timestamp": "2025-11-25T22:23:55.119495", "type": "OtherEvent"}
{"message": "\nThought #3: The closest GameStop store to zip code 90028 is located at 5533 W Sunset Blvd, Los Angeles, CA 90028. It is currently closed and will open tomorrow at 11 AM. The contact number is (323) 466-3714.\nAction #3: executing tool 'terminate' with arguments {\"action\": \"terminate\", \"status\": \"success\"}", "timestamp": "2025-11-25T22:23:58.729038", "type": "OtherEvent"}
{"message": "Observation#3: The closest GameStop store to zip code 90028 is located at 5533 W Sunset Blvd, Los Angeles, CA 90028. It is currently closed and will open tomorrow at 11 AM. The contact number is (323) 466-3714.", "timestamp": "2025-11-25T22:24:02.379069", "type": "OtherEvent"}
```
</details>
