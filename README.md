# r3con

[![arXiv](https://img.shields.io/badge/arXiv-2609.27173-b31b1b.svg)](https://arxiv.org/abs/2609.27173)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE)

**Context representation for large-scale reasoning with smaller, cheaper models.**

Ask a question whose evidence is scattered across many documents. `r3con` builds a
task-specific representation of the context, then reasons over it to produce an answer.

![r3con app: a run on six memos builds its representation, then lights the answer's provenance back to the four memos it used](https://raw.githubusercontent.com/michaeltheologitis/r3con/main/docs/demo.gif)

## App

Run r3con in your browser:

```bash
uv tool install "r3context[app]"
r3con app
```

## Library

### Install

```bash
pip install r3context
```

Or `uv add r3context`.

Set the API key(s) for the provider(s) you use:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

You can also put them in a `.env` file in your working directory.

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

Quote glob patterns so your shell passes them to `r3con` unchanged. A directory or a
quoted glob reads only files inside that directory, or the glob's part before its first
wildcard, and logs and skips a link that points outside it; a file you name is read as
named.

### Python

Load documents from a directory, glob, or list of files:

```python
from r3con import r3con

docs = r3con.read_documents("./reports")
result = r3con.run(
    "Which supplier missed the most delivery windows?",
    docs,
)
print(result)  # "Ardent Logistics missed the most delivery windows: 8 in total."
```

`read_documents` reads PDF and Word (`.docx`) files, and text files such as `.txt`,
`.md`, `.csv`, `.json`, `.html` and `.xml`, and skips a folder's hidden files and folders;
a glob's wildcards skip them too, as in the shell.

If you already have the documents in memory, pass them directly:

```python
result = r3con.run(
    "Which supplier missed the most delivery windows?",
    [doc_a, doc_b, doc_c],
)
```

`documents` is simply a list of strings, one per document.

To ask another question over the same run, `r3con.ask` runs only the reasoning step
again:

```python
other = r3con.ask(result, "Which supplier delivered on time most often?")
print(other)
```

Pass `cancel=threading.Event()` to `run` or `ask` and set it to stop: no new model call is
sent and it raises `r3con.Cancelled`, keeping what it did. A call already sent, and
retries you set yourself (`completion=` or `params`), run their course.

## Notebooks

The examples use five short memos arranged so that no single document contains the full
answer:

- [`examples/quickstart.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/quickstart.ipynb) — an end-to-end example.
- [`examples/provenance.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/provenance.ipynb) — tracing an answer back to the documents and
  relevance snippets it came from.
- [`examples/options.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/options.ipynb) — models, relevance rounds, the reasoning
  path, generation parameters, run configuration, custom connections, and artifact paths.
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

r3con also records provenance, how the answer, each relevance snippet and each parse came
to be; see [`examples/provenance.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/provenance.ipynb).

## Run artifacts

Every run writes each stage's output and model calls, failed ones included, with their
tokens, cost, and any resends and the wait between them, to a folder under `./logs`.
API keys a provider's error echoes are masked before r3con writes or logs them; a
`completion=` wrapper that adds its own key registers it with
`r3con.masking.register_key(key)`.

```
logs/<timestamp>_<hex>/          <- result.run_dir
├── ids.json                     <- each document's id
├── events.jsonl                 <- each stage's start and finish, as they happened
├── splits.json                  <- only when a document was read in parts
├── relevance/
│   ├── result.json              <- each round's snippets
│   └── calls.json
├── structuring/
│   ├── schema/
│   │   ├── result.json
│   │   ├── calls.json
│   │   └── transcript.yaml
│   └── parsing/
│       ├── result.json          <- the parsed records
│       └── calls.json
├── reasoning/
│   ├── result.json              <- the answer
│   ├── calls.json
│   └── transcript.yaml
└── asks/<folder>/               <- one per r3con.ask: manifest.json, events.jsonl and reasoning/
```

## Evaluation

Paper: for benchmark results, baselines, and evaluation code, see
[r3con-evaluation](https://github.com/michaeltheologitis/r3con-evaluation).

## License

MIT — see [LICENSE](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE).

`src/r3con/runtime/python_executor.py` contains a modified version of the sandboxed
interpreter from [smolagents](https://github.com/huggingface/smolagents), licensed under
Apache-2.0 and © HuggingFace Inc. See [NOTICE](https://github.com/michaeltheologitis/r3con/blob/main/NOTICE) and
[LICENSE-APACHE-2.0](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE-APACHE-2.0).
