# r3con

**Reasoning over a corpus of documents — by building the right representation first.**

Ask a question whose evidence is scattered across many documents. `r3con` doesn't answer it
with a fixed pipeline: it constructs a representation *for that question*, then reasons over it.

![How r3con works](docs/r3con.png)

## Install

```bash
pip install r3context
```

Set the key your provider already expects (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …), or drop
a `.env` in your working directory.

## Use

**Command line** — point it at a folder, a glob, or files:

```bash
r3con run "Which supplier missed the most delivery windows?" ./reports
```

**Python, reading documents off disk:**

```python
from r3con import r3con

docs = r3con.read_documents("./reports")     # a folder, a glob, or a list of files
                                             # .txt .md .csv .json .html .xml … and .pdf
result = r3con.run("Which supplier missed the most delivery windows?", docs)
print(result)
```

**Python, documents you already have:**

```python
result = r3con.run("Which supplier missed the most delivery windows?", [doc_a, doc_b, doc_c])
```

`documents` is a list of strings — the documents themselves.

**Options:**

```python
r3con.run(question, docs,
          model="anthropic/claude-sonnet-4",    # any litellm model string
          relevance_rounds=3,                   # how many times each document is re-read
          params={"temperature": 0.7},          # passed straight to litellm
          completion=router.completion)         # your own connection, if you have one
```

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs the whole thing over five short
memos arranged so no single one holds the answer.

## What you get back

```python
result.answer               # the answer — str(result) gives you the same
result.relevant_context     # what each document contributed, aligned to your documents
result.structured_context   # the records the answer was computed from
result.run_dir              # the folder this run wrote (below)
```

## Logs

Every run writes a folder under `./logs` with everything it did — each document's relevance
snippet in every round, the schema it proposed, the parsed records, and the reasoning
transcript.

```
logs/<timestamp>_<hex>/          <- result.run_dir
├── relevance/            result.json · calls.json
├── structuring/
│   ├── schema/           result.json · calls.json · transcript.yaml
│   └── parsing/          result.json · calls.json
└── reasoning/            result.json · calls.json · transcript.yaml
```

## Licence

MIT — see [LICENSE](LICENSE). `src/r3con/runtime/python_executor.py` is a modified copy of the
sandboxed interpreter from [smolagents](https://github.com/huggingface/smolagents), Apache-2.0,
© HuggingFace Inc. See [NOTICE](NOTICE) and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
