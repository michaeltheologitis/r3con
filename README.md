
# r3con

[![arXiv](https://img.shields.io/badge/arXiv-2609.27173-b31b1b.svg)](https://arxiv.org/abs/2609.27173)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE)

**Context representation for large-scale reasoning with smaller, cheaper models.**

Ask a question whose evidence is scattered across many documents. `r3con` builds a
task-specific representation of the context, then reasons over it to produce an answer.

![r3con app: a run on six memos builds its representation, then lights the answer's provenance back to the four memos it used; the sentences it quotes unfold under the answer, a file's name there opens its memo with the same sentences marked, and the panel's answer and summary nodes open their popups](https://raw.githubusercontent.com/michaeltheologitis/r3con/main/docs/demo.gif)

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

Pass a folder containing documents:

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

You can read an entire folder, select files using patterns, or specify individual files:

```bash
# All supported files in the reports folder and its subfolders
r3con run "Who approved the Q3 overspend?" ./reports

# Only PDF files directly inside reports
r3con run "Who approved the Q3 overspend?" "./reports/*.pdf"

# PDF files inside reports and its subfolders
r3con run "Who approved the Q3 overspend?" "./reports/**/*.pdf"

# Only these three files
r3con run "Who approved the Q3 overspend?" q1.md q2.md q3.md
```

Put file patterns containing `*` in quotes, as shown above. `*` matches filenames,
while `**` also lets you search subfolders.

When you select a folder or use a file pattern, r3con only reads files inside
that folder. If a link points to a file outside it, r3con skips it and records
that it was skipped. You can also specify individual files directly, like
`q1.md q2.md q3.md`, to read exactly those files.

### Python

Load documents from a directory, glob pattern, or list of files:

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
`.md`, `.csv`, `.json`, `.html` and `.xml`. It skips hidden files and folders
when reading a directory or using file patterns.

It returns the text of each document, using its path within the folder as its
name (`{"q3/ardent.pdf": "...", ...}`). The answer also identifies the documents
it quotes:

```python
quote = result.provenance["quotes"][0]
print(quote["name"], quote["text"])  # q3/ardent.pdf Ardent missed 8 delivery windows.
```

If you already have the documents in memory, pass them directly:

```python
result = r3con.run(
    "Which supplier missed the most delivery windows?",
    [doc_a, doc_b, doc_c],
)
```

`documents` can be a list of strings (one per document) or a dictionary mapping
document names to their text, as returned by `read_documents`.

`result.document_names` lists each document's name. If you pass a list, the
documents are named by their position: `"0"`, `"1"`, and so on. You can access
the original document using `documents[int(name)]`.

To ask another question over the same run, `r3con.ask` reuses the existing
representation and only runs the reasoning step again:

```python
other = r3con.ask(result, "Which supplier delivered on time most often?")
print(other)
```

You can stop `run` or `ask` using `cancel=threading.Event()`. When you set that
event, r3con stops sending new requests to the model and raises
`r3con.Cancelled`. Work already completed is kept. Requests already sent to
the model will still finish, including any retries you configured yourself
through `completion=` or `params`.

## Notebooks

The examples use five short memos arranged so that no single document contains the full
answer:

- [`examples/quickstart.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/quickstart.ipynb) — an end-to-end example.
- [`examples/provenance.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/provenance.ipynb) — checking an answer: the sentences behind it,
  and the relevance snippets and records it used.
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

r3con also records where the answer, each relevance snippet, and each parsed
record came from; see [`examples/provenance.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/provenance.ipynb).

## Run artifacts

Every run saves the output of each stage and records of its model calls, including
failed calls, to a folder under `./logs`. These records include token usage,
cost, retries, and how long r3con waited between retries.

If a provider includes an API key in an error message, r3con masks it before
saving or logging the error. If you use a custom `completion=` wrapper that
adds its own API key, register it with `r3con.masking.register_key(key)` so
r3con can mask that key too.

```text
logs/<timestamp>_<hex>/          <- result.run_dir
├── ids.json                     <- each document's id
├── names.json                   <- each document's name, by its id
├── events.jsonl                 <- each stage's start and finish, as they happened
├── splits.json                  <- only when a document was read in parts
├── relevance/
│   ├── result.json              <- each round's snippets and their quotes
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
Apache-2.0 and © HuggingFace Inc. See
[THIRD-PARTY-NOTICES](https://github.com/michaeltheologitis/r3con/blob/main/THIRD-PARTY-NOTICES).
