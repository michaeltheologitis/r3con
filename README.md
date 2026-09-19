# r3con

**Just-in-time reasoning over a collection of documents.**

Ask a question whose evidence is scattered across many documents. `r3con` does not
answer it with a fixed pipeline — it *builds the apparatus for that question*, at the
moment you ask, and then runs it.

```bash
pip install r3con
```

```bash
r3con run "Which supplier missed the most delivery windows, and by how much?" ./reports
```

```python
from r3con import r3con

result = r3con.run("Which supplier missed the most delivery windows?", [doc_a, doc_b, doc_c])
print(result)                 # the answer
```

**`documents` is a list of strings — the documents themselves.** If yours are on disk,
`read_documents` turns a folder, a glob or a list of files into that list:

```python
docs = r3con.read_documents("./reports")          # or "reports/*.md", or [f1, f2]
result = r3con.run("Which supplier missed the most delivery windows?", docs)
```

Two jobs, two functions: nothing has to guess whether the string you passed was a
filename or a document.

A run hands back the views it derived the answer from, because those are often what you
actually wanted:

```python
result.answer         # the answer text (str(result) gives you the same)
result.relevance      # what each document was found to contribute, aligned to your documents
result.struct_data    # the structured parse, each record tagged with its source document
result.schema_code    # the schema that was proposed for this question
result.run_dir        # where the full artifacts were written
```

See [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a runnable version, with
a small corpus in `examples/memos/`. From a checkout:

```bash
uv sync          # .venv is ready to be a notebook kernel
```

then pick `.venv` as the notebook's interpreter.

## The three moves

The name is the method: **r³** — relevance, representation, reasoning.

**1 · Surfacing relevance.** Relevance is not a property a document *has*. The same
filing is decisive for one question and noise for another, and a document can be
uninterpretable until you know what the others say. So each document is read against
your question and written up as a **relevance state** — a note of what *this* document
contributes. Then every document is read again, this time given the *other* documents'
states. A passing mention becomes the missing link; a promising document turns out to
contribute nothing. Together the notes are the **corpus-wide relevance state**.

**2 · Structuring.** Prose is a poor substrate for the thing language models are least
reliable at — precise bookkeeping over many items. So the surfaced relevance is
committed to a form: a small Pydantic schema, proposed *for this question* from what the
documents were found to contain, then filled from every document in parallel. Deciding
the schema is itself a reasoning act — it picks the ontology the question lives in.

**3 · Reasoning.** The filled records are concatenated into one table, loaded into a
sandboxed Python runtime, and an agent reasons there — over the structured records and
the relevance states together, writing and running code until it commits an answer.

Two things follow from this shape. The whole collection is **never placed in a single
prompt**: raw text is read one document at a time, and information crosses document
boundaries through the relevance states. And nothing is **chunked** — each document is
read whole, so a document that does not fit in the model's context is an error, not a
silently degraded answer.

## Configuration

Everything that shapes the answer — model, seed, how many rounds of relevance to
surface, the prompt version of each stage, generation params — lives in one run
config, and its `label()` is the identity of a run:

```
default[model=gpt-5.6-luna,seed=42,rounds=2,prompts=(rel=v1,schema=v1,parse=v1,reason=v1)]
```

```bash
r3con run "..." ./docs --model openai/gpt-5.6-luna --relevance-rounds 3
```

Set your provider key the usual way (`OPENAI_API_KEY=…`, or a `.env` in the working
directory); `--base-url` points at a self-hosted endpoint.

## Connecting to a model

r3con calls [litellm](https://docs.litellm.ai) directly, so the connection works the way
litellm's does and nothing here is r3con-specific. Set the environment variable your
provider already expects and pass its model string:

```bash
export OPENAI_API_KEY=...          # or ANTHROPIC_API_KEY, GEMINI_API_KEY, …
r3con run "..." ./docs --model anthropic/claude-sonnet-4
```

A `.env` in your working directory is loaded for you. `--base-url` points at a self-hosted
endpoint (`--model hosted_vllm/Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8000/v1`).
There is no r3con-branded API-key variable, and no model registry: if litellm reaches it,
so do we.

**If you already own the connection, hand it over.** Anything with litellm's
`(model, messages, **kwargs)` shape can replace the call — a configured `Router` for
fallbacks or load balancing, or your own wrapper for caching, logging, or an internal
gateway:

```python
import litellm
from r3con import r3con

router = litellm.Router(model_list=[...], routing_strategy="simple-shuffle")

r3con.run("...", docs, model="my-model-group", completion=router.completion)
```

Generation parameters go in one place — `params`, passed straight through to
`litellm.completion`:

```python
r3con.run("...", docs, params={"temperature": 0.7, "extra_body": {"top_k": 20}})
```

## What it leaves behind

Every run writes a folder under `./logs` containing exactly what it did: the relevance
state of each document in each round, the schema that was proposed (and any repair
attempts), the parsed records with the document each came from, and the full reasoning
transcript. The answer alone rarely tells you whether to trust it; these do.

```
logs/<timestamp>_<hex>/
├── relevance/              result.json · calls.json
├── structuring/
│   ├── schema/             result.json · calls.json · transcript.yaml
│   └── parsing/            result.json · calls.json
└── reasoning/            result.json · calls.json · transcript.yaml
```

## Security: this package runs model-generated code

Both halves of the method involve executing Python that a language model wrote, on your
machine, in a process conditioned on **your documents' contents**:

- **Stage 2** `exec`s the proposed schema in-process to validate it. There is no sandbox
  around that step.
- **Stage 3** runs the agent's code in a restricted AST
  interpreter — an import allowlist, blocked dunder access, and operation/loop/wall-clock
  caps — which raises the bar considerably but is not a security boundary you should bet
  a production secret on.

Treat documents as untrusted input, and don't run `r3con` against documents you don't
trust in a process that holds credentials you can't afford to lose. Running it in a
container or a dedicated virtualenv is the sane default for anything sensitive.

## Licence

MIT — see [LICENSE](LICENSE).

`src/r3con/runtime/python_executor.py` is a modified copy of the sandboxed interpreter
from [smolagents](https://github.com/huggingface/smolagents), Apache-2.0, © HuggingFace
Inc. See [NOTICE](NOTICE) and [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0).
