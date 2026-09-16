from pathlib import Path

from scenecut_tracking.evaluation import evaluate_mot


def _write(path: Path, rows: list[str]):
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_perfect_tracking_scores_are_perfect(tmp_path):
    ground_truth = tmp_path / "gt.txt"
    prediction = tmp_path / "pred.txt"
    rows = [
        "1,1,10,10,20,30,1,1,1",
        "2,1,12,10,20,30,1,1,1",
        "3,1,14,10,20,30,1,1,1",
    ]
    _write(ground_truth, rows)
    _write(prediction, rows)

    metrics = evaluate_mot(ground_truth, prediction)

    assert metrics["HOTA"] == 100.0
    assert metrics["AssA"] == 100.0
    assert metrics["IDF1"] == 100.0
    assert metrics["MOTA"] == 100.0
    assert metrics["IDs"] == 0
    assert metrics["Frag"] == 0


def test_identity_switch_is_counted(tmp_path):
    ground_truth = tmp_path / "gt.txt"
    prediction = tmp_path / "pred.txt"
    _write(ground_truth, [
        "1,1,10,10,20,30,1,1,1",
        "2,1,12,10,20,30,1,1,1",
        "3,1,14,10,20,30,1,1,1",
    ])
    _write(prediction, [
        "1,1,10,10,20,30,1,1,1",
        "2,2,12,10,20,30,1,1,1",
        "3,2,14,10,20,30,1,1,1",
    ])

    metrics = evaluate_mot(ground_truth, prediction)

    assert metrics["IDs"] == 1
    assert metrics["IDF1"] < 100.0

