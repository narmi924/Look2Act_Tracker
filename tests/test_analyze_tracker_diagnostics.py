import json
import sys

import pandas as pd

from scripts.analyze_tracker_diagnostics import (
    analyze_file,
    bool_ratio,
    point_extent,
    point_jitter,
    print_summary,
    main,
)


def test_bool_ratio_parses_common_true_values():
    df = pd.DataFrame({"valid": ["True", "false", "1", "yes", ""]})

    assert bool_ratio(df, "valid") == 0.75  # blank is unknown, not False


def test_point_extent_and_jitter():
    df = pd.DataFrame(
        {
            "raw_x": [0.0, 3.0, 6.0],
            "raw_y": [0.0, 4.0, 8.0],
        }
    )

    extent = point_extent(df, "raw_x", "raw_y")
    jitter = point_jitter(df, "raw_x", "raw_y")

    assert extent["width"] == 6.0
    assert extent["height"] == 8.0
    assert jitter["mean"] == 5.0
    assert jitter["p95"] == 5.0


def test_analyze_file_summarizes_diagnostic_csv(tmp_path):
    csv_path = tmp_path / "diag.csv"
    pd.DataFrame(
        {
            "backend": ["classic", "classic"],
            "valid": [True, False],
            "face_detected": [True, True],
            "fps": [30.0, 20.0],
            "raw_x": [0.1, 0.2],
            "raw_y": [0.3, 0.4],
            "calibrated_x": [100.0, 110.0],
            "calibrated_y": [200.0, 210.0],
            "error": ["", "no face"],
        }
    ).to_csv(csv_path, index=False)

    summary = analyze_file(csv_path)

    assert summary["rows"] == 2
    assert summary["backend_counts"] == {"classic": 2}
    assert summary["valid_ratio"] == 0.5
    assert summary["error_counts"] == {"no face": 1}


def test_old_csv_missing_face_and_fps_is_unknown_in_json_and_terminal(tmp_path, capsys):
    csv_path = tmp_path / 'legacy.csv'
    pd.DataFrame({'backend': ['classic'], 'valid': [False]}).to_csv(csv_path, index=False)
    summary = analyze_file(csv_path)
    print_summary(summary)
    output = capsys.readouterr().out
    assert summary['face_detected_ratio'] is None
    assert summary['face_detected_counts'] == dict(true=0, denominator=0, unknown_rows=1, ratio=None)
    assert summary['fps']['count'] == 0
    assert 'face=unknown' in output
    assert 'fps=unknown' in output
    assert 'face=0.000' not in output


def test_mixed_observed_and_empty_rows_use_known_denominators(tmp_path, capsys):
    csv_path = tmp_path / 'mixed.csv'
    pd.DataFrame({'backend': ['classic'] * 3, 'source_valid': [True, False, None],
                  'face_detected': [True, False, None], 'fps': [30., 0., None]}).to_csv(csv_path, index=False)
    summary = analyze_file(csv_path)
    print_summary(summary)
    output = capsys.readouterr().out
    assert summary['valid_counts'] == dict(true=1, denominator=2, unknown_rows=1, ratio=.5)
    assert summary['face_detected_counts'] == dict(true=1, denominator=2, unknown_rows=1, ratio=.5)
    assert summary['fps']['count'] == 2 and summary['fps']['min'] == 0.
    assert summary['fps']['denominator'] == 3 and summary['fps']['unavailable_count'] == 1
    assert 'face=0.500 (1/2; unknown=1)' in output
    assert 'fps mean=15.0' in output and '(n=2/3)' in output


def test_all_measured_face_false_is_real_zero_ratio(tmp_path, capsys):
    csv_path = tmp_path / 'no-face.csv'
    pd.DataFrame({'face_detected': [False, False, None]}).to_csv(csv_path, index=False)
    summary = analyze_file(csv_path)
    print_summary(summary)
    output = capsys.readouterr().out
    assert summary['face_detected_ratio'] == 0.
    assert summary['face_detected_counts'] == dict(true=0, denominator=2, unknown_rows=1, ratio=0.)
    assert 'face=0.000 (0/2; unknown=1)' in output


def test_measured_zero_fps_is_reported_as_zero(tmp_path, capsys):
    csv_path = tmp_path / 'zero-fps.csv'
    pd.DataFrame({'fps': [0.]}).to_csv(csv_path, index=False)
    summary = analyze_file(csv_path)
    print_summary(summary)
    output = capsys.readouterr().out
    assert summary['fps']['count'] == 1 and summary['fps']['mean'] == 0.
    assert 'fps mean=0.0' in output and 'fps=unknown' not in output


def test_cli_json_and_terminal_use_same_missing_value_semantics(tmp_path, monkeypatch, capsys):
    csv_path = tmp_path / 'missing.csv'
    output_path = tmp_path / 'summary.json'
    pd.DataFrame({'backend': ['classic'], 'source_valid': [False]}).to_csv(csv_path, index=False)
    monkeypatch.setattr(sys, 'argv', ['analyze_tracker_diagnostics.py', str(csv_path), '--output', str(output_path)])
    assert main() == 0
    saved = json.loads(output_path.read_text(encoding='utf-8'))['files'][0]
    shown = capsys.readouterr().out
    assert saved['face_detected_ratio'] is None
    assert saved['face_detected_counts']['denominator'] == 0
    assert saved['fps']['status'] == 'unavailable'
    assert 'face=unknown' in shown and 'fps=unknown' in shown
