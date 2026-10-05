# r3con

[![arXiv](https://img.shields.io/badge/arXiv-2609.27173-b31b1b.svg)](https://arxiv.org/abs/2609.27173)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE)

**Context representation for large-scale reasoning.**

Ask a question whose evidence is scattered across many documents. `r3con` builds a
task-specific representation of that context, then reasons over it to produce an answer.

![How r3con works](https://raw.githubusercontent.com/michaeltheologitis/r3con/main/docs/r3con.png)

## Install

```bash
pip install r3context
```

Set the API key(s) for the provider(s) you use:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

You can also put them in a `.env` file in your working directory.

## Usage

### Command line

Pass a directory, glob, or list of files:

```bash
r3con run "Which supplier missed the most delivery windows?" ./reports \
  --model openai/gpt-6-luna
```

`--model` accepts any [LiteLLM](https://docs.litellm.ai/docs/providers) model string, for
example:

```
anthropic/claude-sonnet-5-5
gemini/gemini-3.8-flash
hosted_vllm/Qwen/Qwen3.5-35B-A3B
```

The default is `openai/gpt-6-luna`.

Directories are searched recursively. You can also use glob patterns or pass files directly:

```bash
r3con run "Who approved the Q3 overspend?" "./reports/*.pdf"
r3con run "Who approved the Q3 overspend?" "./reports/**/*.pdf"
r3con run "Who approved the Q3 overspend?" q1.md q2.md q3.md
```

Quote glob patterns so your shell passes them to `r3con` unchanged.

### Python

Load documents from a directory, glob, or list of files:

```python
from r3con import r3con

docs = r3con.read_documents("./reports")
result = r3con.run(
    "Which supplier missed the most delivery windows?",
    docs,
)
print(result)
```

`read_documents` supports `.txt`, `.md`, `.csv`, `.json`, `.html`, `.xml`, `.pdf`, and
other common text formats.

If you already have the documents in memory, pass them directly:

```python
result = r3con.run(
    "Which supplier missed the most delivery windows?",
    [doc_a, doc_b, doc_c],
)
```

`documents` is simply a list of strings, one per document.

## Notebooks

The examples use five short memos arranged so that no single document contains the full
answer:

- [`examples/quickstart.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/quickstart.ipynb) — an end-to-end example.
- [`examples/options.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/options.ipynb) — models, relevance rounds, generation
  parameters, run configuration, custom connections, and artifact paths.
- [`examples/vllm.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/vllm.ipynb) — using a self-hosted vLLM endpoint.

## Output

`r3con.run(...)` returns a result object containing the final answer and the representation
built during the run:

```python
result.answer  # str       — final answer
result.relevant_context  # list[str] — final relevance snippets
result.structured_context  # dict      — structured representation
result.run_dir  # Path      — directory containing run artifacts
```

`str(result)` returns the same value as `result.answer`.

## Run artifacts

Every run writes its intermediate artifacts to a directory under `./logs`. This includes the
relevance snippet produced for each document at every round, the generated schema, parsed
records, model calls, and the final reasoning transcript.

```
logs/<timestamp>_<hex>/          <- result.run_dir
├── splits.json                  <- only when a document was read in parts
├── notes.json                   <- only when notes were read again shorter
├── relevance/
│   ├── result.json
│   └── calls.json
├── structuring/
│   ├── schema/
│   │   ├── result.json
│   │   ├── calls.json
│   │   └── transcript.yaml
│   └── parsing/
│       ├── result.json
│       └── calls.json
└── reasoning/
    ├── result.json
    ├── calls.json
    └── transcript.yaml
```

## Evaluation

For benchmark results, baselines, and evaluation code, see
[r3con-evaluation](https://github.com/michaeltheologitis/r3con-evaluation).

## License

MIT — see [LICENSE](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE).

`src/r3con/runtime/python_executor.py` contains a modified version of the sandboxed
interpreter from [smolagents](https://github.com/huggingface/smolagents), licensed under
Apache-2.0 and © HuggingFace Inc. See [NOTICE](https://github.com/michaeltheologitis/r3con/blob/main/NOTICE) and
[LICENSE-APACHE-2.0](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE-APACHE-2.0).
