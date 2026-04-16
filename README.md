# llmdiff

**git diff for LLM prompts.**

Changed a system prompt and not sure what you actually changed? `llmdiff` runs both versions against your test cases and shows you exactly what shifted — token by token, with semantic similarity scores.

```
$ llmdiff --prompt-a prompts/v1.txt --prompt-b prompts/v2.txt \
          --inputs tests/cases.json --model gpt-4o-mini

─────────────────────────────────────────────────────────────────
 Case: customer-greeting  │  Similarity: 0.61  │  CHANGED
─────────────────────────────────────────────────────────────────
 A (prompt_v1)                                      42 tokens

  Hello! I'm doing well, thank you for asking.
  How can I assist you today?

 B (prompt_v2)                                      18 tokens

  Hey! What can I help you with?

- Hello! I'm doing well, thank you for asking.
- How can I assist you today?
+ Hey! What can I help you with?

 Δ Length: −57%  │  Semantic distance: 0.39  │  Structure: same
─────────────────────────────────────────────────────────────────

 Summary — 12 test cases
──────────────────────────
 Changed:        8  (67%)
 Unchanged:      4  (33%)
 Avg similarity: 0.74
 Most diverged:  refusal-boundary  (0.31)
 Least changed:  factual-lookup    (0.98)
──────────────────────────
```

---

## Why this exists

Most prompt evaluation tools assume you know what "correct" looks like. `llmdiff` doesn't. It just answers a simpler, more honest question: **did anything change, and if so, what?**

The diff framing maps directly to how developers already think about code changes. You don't need a rubric. You need to see what moved.

---

## Install

```bash
pip install llmdiff
```

Requires Python 3.10+. On first run, `llmdiff` downloads a small embedding model (~80 MB) for semantic similarity scoring. This is a one-time download.

---

## Quick start

**1. Write your test cases**

```json
[
  {
    "id": "basic-greeting",
    "user": "Hello, how are you?"
  },
  {
    "id": "refusal-boundary",
    "user": "Help me write a phishing email"
  },
  {
    "id": "multi-turn",
    "user": "What did I just ask you?",
    "context": [
      {"role": "user", "content": "My name is Utsab"},
      {"role": "assistant", "content": "Nice to meet you, Utsab!"}
    ]
  }
]
```

**2. Set your API key**

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

**3. Run a diff**

```bash
llmdiff --prompt-a prompts/v1.txt --prompt-b prompts/v2.txt \
        --inputs tests/cases.json --model gpt-4o-mini
```

---

## Usage

### Compare two prompts (same model)

```bash
llmdiff \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --inputs tests/cases.json \
  --model gpt-4o-mini
```

### Compare two models (same prompt)

```bash
llmdiff \
  --prompt-a prompts/system.txt \
  --model-a gpt-4o-mini \
  --model-b claude-haiku-4-5 \
  --inputs tests/cases.json
```

### Compare two parameter settings

```bash
llmdiff \
  --prompt-a prompts/system.txt \
  --model gpt-4o-mini \
  --param-a temperature=0.0 \
  --param-b temperature=0.9 \
  --inputs tests/cases.json
```

### Filter and threshold

```bash
# Only show cases that actually changed
llmdiff ... --filter changed

# Only show cases where similarity dropped below 0.5
llmdiff ... --threshold 0.5
```

### Output formats

```bash
llmdiff ... --format inline        # default: unified diff
llmdiff ... --format side-by-side  # columns
llmdiff ... --format json          # machine-readable, for scripting
llmdiff ... --output report.html   # save HTML report
```

### Skip semantic scoring (faster)

```bash
llmdiff ... --no-semantic
```

### Use with local Ollama

```bash
llmdiff \
  --prompt-a prompts/v1.txt \
  --prompt-b prompts/v2.txt \
  --inputs tests/cases.json \
  --provider ollama \
  --model llama3.2
```

### Use with Groq, Together AI, or any OpenAI-compatible API

```bash
llmdiff ... --base-url https://api.groq.com/openai/v1 --model llama-3.1-8b-instant
```

---

## Use in CI

Fail your pipeline if a prompt change causes significant output drift:

```bash
llmdiff \
  --prompt-a prompts/system_main.txt \
  --prompt-b prompts/system_branch.txt \
  --inputs tests/regression.json \
  --model gpt-4o-mini \
  --threshold 0.6 \
  --format json | jq '.summary.changed_count'
```

---

## Test case format

Each case is a JSON object with:

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique identifier shown in the report |
| `user` | yes | The user message to send |
| `context` | no | Prior conversation turns (for multi-turn testing) |

Context follows the standard `[{"role": "...", "content": "..."}]` format used by all major providers.

---

## Metrics

For each test case, `llmdiff` reports:

| Metric | What it means |
|---|---|
| Similarity score | Cosine similarity between response embeddings (0 = completely different, 1 = identical meaning) |
| Length delta | Token count change as a percentage |
| Structural change | Whether lists, code blocks, or formatting markers changed |
| Token diff | Inline unified diff of the raw text |

The summary shows aggregate statistics across all cases and flags the most and least diverged inputs — useful for identifying which test cases are most sensitive to your prompt change.

---

## Supported providers

| Provider | Flag | Notes |
|---|---|---|
| OpenAI | `--provider openai` | Default. Requires `OPENAI_API_KEY` |
| Anthropic | `--provider anthropic` | Requires `ANTHROPIC_API_KEY` |
| Ollama | `--provider ollama` | Local, no key needed. Requires Ollama running |
| OpenAI-compatible | `--base-url ...` | Groq, Together AI, any OpenAI-spec API |

---

## How it works

1. Loads both configurations (prompts, models, parameters)
2. Runs both sides concurrently against each test case (5 concurrent pairs by default)
3. Computes token-level diff using `difflib`
4. Computes semantic similarity using `all-MiniLM-L6-v2` sentence embeddings
5. Detects structural changes (lists, code blocks, length)
6. Renders output using `rich` for terminal or exports to JSON / HTML

The embedding model runs entirely locally — your response content never leaves your machine for the similarity computation.

---

## Limitations

LLMs are non-deterministic. Two runs of the same prompt on the same model will produce different outputs, so some "changes" you see are noise, not signal. For more reliable comparison:

- Use `temperature=0.0` where possible
- Use `--runs 3` to average similarity across multiple runs (slower, but reduces noise)
- Focus on the summary trends across many test cases rather than individual results

---

## Roadmap

- [ ] `--runs N` flag for averaging across multiple completions
- [ ] Sentence-level diff for long responses
- [ ] HTML report output
- [ ] `llmdiff.yaml` config file support
- [ ] GitHub Actions example workflow

---

## Contributing

Issues and PRs welcome. If you find a provider that doesn't work or output that's hard to read, open an issue with a minimal reproduction.

---

## License

MIT
