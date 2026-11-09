import pytest

from tests.app.build_sample import APP_JS, app_block, built_blocks


@pytest.mark.parametrize("name", ["SAMPLE", "RECORDED_ASKS"])
def test_app_js_holds_the_block_its_run_folder_builds(name: str) -> None:
    app = APP_JS.read_text(encoding="utf-8")
    assert app_block(app, name) == built_blocks()[name]
