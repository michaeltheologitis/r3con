import json
import zipfile

import pydantic
import pytest

from evals import JUDGE_MODEL, loong

LOCAL_HEADER = b"PK\x03\x04"
CENTRAL_HEADER = b"PK\x01\x02"
FLAGS_OFFSET = {LOCAL_HEADER: 6, CENTRAL_HEADER: 8}
UTF8_NAMES_FLAG = 0x800
FLAGS_SIZE = 2


@pytest.mark.parametrize(
    ("task_id", "documents"),
    [
        ("loong-financial", ["《Annual》\nACME's 2024 revenue was 7 million.\n\n"]),
        (
            "loong-financial-years",
            ["《Annual》\nACME's 2023 revenue was 6 million.\n\n"],
        ),
        ("loong-paper", ["Sparse Notes\n# Sparse Notes\nWe study notes.\n\n"]),
    ],
)
def test_each_document_is_found_and_titled_as_upstream_does(
    loong_cache, task_id, documents
):
    assert loong.inputs(loong.rows()[task_id])[1] == documents


def test_legal_verdicts_are_withheld_only_when_the_question_asks_to_match_them(
    loong_cache,
):
    rows = loong.rows()

    assert loong.inputs(rows["loong-legal"])[1] == [
        "《判决文书1》\nBody A. Verdict A.\n\n",
        "《判决文书2》\nBody B. Verdict B.\n\n",
    ]
    assert loong.inputs(rows["loong-verdicts"])[1] == [
        "《判决文书1》\nBody A.\n\n",
        "《判决文书2》\nBody B.\n\n",
    ]


@pytest.mark.parametrize(
    ("task_id", "question"),
    [
        ("loong-financial", "Answer from the reports.\n\nWhat was ACME's revenue?"),
        ("loong-paper", "List the citations."),
    ],
)
def test_r3con_is_asked_the_instruction_then_the_question(
    loong_cache, task_id, question
):
    assert loong.inputs(loong.rows()[task_id])[0] == question


def test_chinese_file_names_survive_the_pool_extraction(tmp_path, monkeypatch):
    cache = tmp_path / "loong"
    report = "doc/financial/report_000001-2024-平安银行-2024年第一季度报告.txt"
    files = {report: "平安银行的营收。", "doc/legal/legal.json": "{}"}
    write_zip_without_utf8_flag(cache / "doc.zip", files)
    row = {"id": "zh", "type": "financial", "level": 1, "doc": ["平安银行"]}
    row |= {"instruction": "营收是多少？", "question": ""}
    (cache / "loong.jsonl").write_text(json.dumps(row), encoding="utf-8")
    monkeypatch.setattr(loong, "CACHE", cache)

    documents = loong.inputs(loong.rows()["zh"])[1]

    assert documents == ["《2024年第一季度报告》\n平安银行的营收。\n\n"]


def write_zip_without_utf8_flag(path, files: dict[str, str]) -> None:
    """A zip like Loong's pool: UTF-8 file names, without the flag that says so."""
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    data = bytearray(path.read_bytes())
    for header, offset in FLAGS_OFFSET.items():
        start = data.find(header)
        while start != -1:
            at = slice(start + offset, start + offset + FLAGS_SIZE)
            flags = int.from_bytes(data[at], "little") & ~UTF8_NAMES_FLAG
            data[at] = flags.to_bytes(FLAGS_SIZE, "little")
            start = data.find(header, start + 1)
    path.write_bytes(bytes(data))


def test_the_judge_rates_the_answer_against_the_gold_alone(loong_cache, judging_llm):
    row = loong.rows()["loong-paper"]

    assert loong.judge(row, "Sparse Notes") == (40, "Checked the gold.")
    [request] = judging_llm.requests
    gold = json.dumps(row["answer"], ensure_ascii=False, indent=2)
    assert request["model"] == JUDGE_MODEL
    assert request["messages"] == [
        {"role": "system", "content": loong.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "[Question]\nList the citations.\n\n\n\n"
            f"[Gold Answer]\n{gold}\n\n"
            "[The Start of Assistant's Predicted Answer]\nSparse Notes\n"
            "[The End of Assistant's Predicted Answer]",
        },
    ]
    assert not {"seed", "temperature"} & request.keys()


def test_a_rating_outside_1_to_100_is_refused(loong_cache, judging_llm):
    judging_llm.replies(json.dumps({"rationale": "Better than gold.", "rating": 101}))

    with pytest.raises(pydantic.ValidationError, match="less than or equal to 100"):
        loong.judge(loong.rows()["loong-paper"], "Sparse Notes")


def test_an_empty_answer_rates_worst_without_a_call(loong_cache, judging_llm):
    assert loong.judge(loong.rows()["loong-paper"], "  \n") == (1, None)
    assert judging_llm.requests == []


def test_metrics_are_the_mean_rating_and_the_share_rated_100():
    assert loong.metrics([100, 40, 100, 1]) == {"avg_score": 60.25, "perfect_rate": 0.5}
