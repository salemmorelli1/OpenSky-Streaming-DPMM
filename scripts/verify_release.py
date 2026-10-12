#!/usr/bin/env python3
"""Verify the committed methods-only release and generated publication files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from importlib.metadata import version
from pathlib import Path

from pypdf import PdfReader

from opensky_streaming_dpmm import __version__
from opensky_streaming_dpmm.contracts import ENDPOINT_PROTOCOL
from build_site import HTML


ROOT = Path(__file__).resolve().parents[1]


def verify(report: Path) -> None:
    status = json.loads((ROOT / "data/project_status.json").read_text(encoding="utf-8"))
    empirical = json.loads(
        (ROOT / "data/empirical_status.json").read_text(encoding="utf-8")
    )
    assert (
        status["version"]
        == empirical["release"]
        == __version__
        == version("opensky-streaming-dpmm")
    )
    assert status["endpoint_protocol"] == ENDPOINT_PROTOCOL
    assert status["planned_blocks"] == 100 and status["planned_executions"] == 600
    assert status["completed_blocks"] == status["completed_executions"] == 0
    assert status["empirical_results_available"] is False
    assert empirical["public_empirical_summary_available"] is False
    assert (
        empirical["formal_blocks_complete"]
        == empirical["factorial_executions_complete"]
        == 0
    )
    assert not (ROOT / "data/empirical_summary.json").exists(), (
        "pending release contains a stale empirical summary"
    )
    assert status["authenticated_live_api_verified"] is True
    assert empirical["connectivity_check_date"] == "2026-08-29"
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    assert html == HTML, "generated index.html does not match its source"
    for phrase in (
        "No operational SIGINT claim",
        "Execution pipeline ready",
        "No public empirical summary before 600 unique successful rows",
        "levels=['0.0','0.15','0.3']",
        "completeEmpiricalSummary",
        "aggregate fingerprint mismatch",
    ):
        assert phrase in html, f"missing site boundary: {phrase}"
    with (ROOT / "data/experiment_design.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        design = list(csv.DictReader(stream))
    assert (
        len(design) == 6
        and sum(int(row["planned_executions"]) for row in design) == 600
    )
    assert {float(row["dropout_rate"]) for row in design} == {0.0, 0.15, 0.30}
    pdf = PdfReader(report)
    assert len(pdf.pages) == 26
    text = re.sub(
        r"\s+", " ", " ".join(page.extract_text() or "" for page in pdf.pages)
    )
    for phrase in (
        "Paired Block-Contrast Model",
        "mean pre-snapshot log density",
        "VI uses final-posterior maximum responsibilities",
        "SMC records ESS before resampling",
        "one-pass streaming approximation",
        "Synthetic validation cannot unlock the public benchmark",
    ):
        assert phrase in text, f"stale report text: {phrase}"
    assert "inverse of A1" not in text
    print(
        f"Release {__version__} verified: 0/600 empirical cells; 26-page methods report; site gate locked."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "report/OpenSky_Streaming_DPMM_APA_Report.pdf",
    )
    args = parser.parse_args()
    verify(args.report)


if __name__ == "__main__":
    main()
