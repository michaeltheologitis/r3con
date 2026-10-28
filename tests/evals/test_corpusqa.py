import pytest

from evals import JUDGE_MODEL, corpusqa


def test_r3con_is_asked_the_question_then_the_instruction_over_each_document(
    corpusqa_cache, corpusqa_instruction
):
    task, documents = corpusqa.inputs(corpusqa.rows()["sales_en_1"])

    assert task == f"How many units were sold?\n\n{corpusqa_instruction}"
    assert documents == ["Store A sold 40 units.", "Store B sold 2 units."]


@pytest.mark.parametrize(
    ("response", "extracted"),
    [
        ("Adding the stores: 40 + 2.\nThe answer is: 42", "42"),
        ('The answer is: ["A", "B"]  ', '["A", "B"]'),
        ("The answer is: 12.34%\nDone.", "12.34%"),
        ("  42 units  ", "42 units"),
        ("The answer is: ", ""),
        ("", ""),
    ],
)
def test_extract_answer_takes_the_line_after_the_contract_else_the_reply(
    response, extracted
):
    assert corpusqa.extract_answer(response) == extracted


@pytest.mark.parametrize(("task_id", "score"), [("sales_en_1", 1), ("sales_en_2", 0)])
def test_the_judge_compares_the_extracted_answer_with_the_gold(
    corpusqa_cache, judging_llm, task_id, score
):
    row = corpusqa.rows()[task_id]

    assert corpusqa.judge(row, "40 + 2.\nThe answer is: 42") == (score, "Checked.")
    [request] = judging_llm.requests
    assert request["model"] == JUDGE_MODEL
    assert request["messages"] == [
        {"role": "system", "content": corpusqa.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"\nProblem: {row['question']}\nAnswer 1: 42\n"
            f"Answer 2: {row['answer']}\n",
        },
    ]
    assert not {"seed", "temperature"} & request.keys()


def test_an_answer_with_nothing_after_the_contract_scores_0_without_a_call(
    corpusqa_cache, judging_llm
):
    row = corpusqa.rows()["sales_en_1"]
    assert corpusqa.judge(row, "Adding up.\nThe answer is: ") == (0, None)
    assert judging_llm.requests == []


def test_accuracy_is_the_share_judged_equivalent():
    assert corpusqa.metrics([1, 0, 1, 1]) == {"accuracy": 0.75}
