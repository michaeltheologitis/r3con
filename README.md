
# r3con

[![arXiv](https://img.shields.io/badge/arXiv-2609.27173-b31b1b.svg)](https://arxiv.org/abs/2609.27173)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE)

**Context representation for large-scale reasoning with smaller, cheaper models.**

Ask a question whose evidence is spread across many documents. `r3con` finds and
organizes the relevant information, then reasons over it to produce an answer.

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

Set the API key(s) for the AI provider(s) you use:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

You can also put them in a `.env` file in your working directory.

### Command line

Ask a question about documents in a folder:

```bash
r3con run "Which supplier missed the most delivery windows?" ./reports \
  --model openai/gpt-6-luna
```

`./reports` is the folder containing your documents. `--model` selects the AI
model to use.

You can use any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers),
for example:

```
anthropic/claude-sonnet-5-5
gemini/gemini-3.8-flash
hosted_vllm/Qwen/Qwen3.5-35B-A3B
```

The default is `openai/gpt-6-luna`.

You can read an entire folder, select files using glob patterns, or specify
individual files. A glob pattern uses `*` to match filenames and `**` to
include subfolders.

```bash
# All supported files in reports and its subfolders
r3con run "Who approved the Q3 overspend?" ./reports

# Only PDF files directly inside reports
r3con run "Who approved the Q3 overspend?" "./reports/*.pdf"

# PDF files inside reports and its subfolders
r3con run "Who approved the Q3 overspend?" "./reports/**/*.pdf"

# Only these three files
r3con run "Who approved the Q3 overspend?" q1.md q2.md q3.md
```

Keep glob patterns in quotes, as shown above. When reading a folder or using
a glob pattern, r3con skips links to files outside the selected folder.
Individual files can be specified directly.

### Python

Load documents from a folder, glob pattern, or list of files:

```python
from r3con import r3con

docs = r3con.read_documents("./reports")
result = r3con.run(
    "Which supplier missed the most delivery windows?",
    docs,
)
print(result)  # "Ardent Logistics missed the most delivery windows: 8 in total."
```

`read_documents` supports PDF, Word (`.docx`), `.txt`, `.md`, `.csv`, `.json`,
`.html`, and `.xml` files. It skips hidden files and folders when reading
directories or using glob patterns.

It returns each document's text along with its filename. If a file is inside
a subfolder, its name includes that subfolder.

For example, a file at `reports/q3/ardent.pdf` is named `q3/ardent.pdf`.
Here, `q3` is simply the subfolder name, and `ardent.pdf` is the filename.

If your documents are already stored as text in Python, pass them directly
without reading files:

```python
result = r3con.run(
    "Which supplier missed the most delivery windows?",
    [doc_a, doc_b, doc_c],
)
```

You can provide documents as a list:

```python
documents = [doc_a, doc_b, doc_c]
```

Or give them names:

```python
documents = {
    "report_a": doc_a,
    "report_b": doc_b,
    "report_c": doc_c,
}
```

Documents in a list are named by their position: `"0"`, `"1"`, `"2"`, and so on.

To ask another question without repeating the earlier processing, use
`r3con.ask`:

```python
other = r3con.ask(result, "Which supplier delivered on time most often?")
print(other)
```

This reuses the information organized during the first run and only
repeats the final reasoning step.

#### Provenance

`result.provenance["quotes"]` lists the sentences the answer is based on.
Each has the document's `name`, the quoted `text`, and its `start` and `end`
characters in that document:

```python
quote = result.provenance["quotes"][0]
print(quote["name"], quote["text"])  # q3/ardent.pdf Ardent missed 8 delivery windows.
```

[`examples/provenance.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/provenance.ipynb)
shows the rest of the provenance.

## Notebooks

The examples use five short memos, with the information spread across them
so no single document contains the complete answer:

- [`examples/quickstart.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/quickstart.ipynb) — a complete example of using r3con.
- [`examples/provenance.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/provenance.ipynb) — checking the original sentences, snippets, and records behind an answer.
- [`examples/options.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/options.ipynb) — configuring models, relevance rounds, reasoning, generation settings, custom connections, and saved files.
- [`examples/vllm.ipynb`](https://github.com/michaeltheologitis/r3con/blob/main/examples/vllm.ipynb) — using an AI model hosted on your own vLLM server.

## Output

`r3con.run(...)` returns the answer and the information created during the run:

```python
result.answer  # Final answer
result.relevant_context  # Relevant text found in the documents
result.structured_context  # Information extracted and organized from the documents
result.provenance  # What the answer is based on (see Provenance above)
result.run_dir  # Folder where this run's files are saved
```

## Run artifacts

Every run saves its results and processing records in a folder under `./logs`.

These include AI requests (even failed ones), token usage, API costs,
retries, and the time spent waiting between retries.

If an AI provider accidentally includes an API key in an error message,
r3con hides that key before saving the error.

If you use your own `completion=` function with a separate API key, register
that key using `r3con.masking.register_key(key)` so r3con can hide it too.

```text
logs/<timestamp>_<hex>/          <- result.run_dir
├── ids.json                     <- each document's ID
├── names.json                   <- each document's filename
├── events.jsonl                 <- when each processing step started and finished
├── splits.json                  <- how documents were split, if necessary
├── relevance/
│   ├── result.json              <- relevant text and supporting quotes
│   └── calls.json               <- AI requests made during relevance checking
├── structuring/
│   ├── schema/
│   │   ├── result.json          <- structure created for organizing information
│   │   ├── calls.json           <- AI requests
│   │   └── transcript.yaml      <- record of the process
│   └── parsing/
│       ├── result.json          <- information extracted from documents
│       └── calls.json           <- AI requests
├── reasoning/
│   ├── result.json              <- final answer
│   ├── calls.json               <- AI requests
│   └── transcript.yaml          <- record of the reasoning process
└── asks/<folder>/               <- saved results for each additional question
```

## Evaluation

For benchmark results, comparisons with other methods, and evaluation code,
see [r3con-evaluation](https://github.com/michaeltheologitis/r3con-evaluation).

## License

MIT — see [LICENSE](https://github.com/michaeltheologitis/r3con/blob/main/LICENSE).

`src/r3con/runtime/python_executor.py` includes modified code from
[smolagents](https://github.com/huggingface/smolagents), licensed under
Apache-2.0 and © HuggingFace Inc.

See [THIRD-PARTY-NOTICES](https://github.com/michaeltheologitis/r3con/blob/main/THIRD-PARTY-NOTICES).
