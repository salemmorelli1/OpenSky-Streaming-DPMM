import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from opensky_streaming_dpmm.empirical import analyze_results
from test_empirical import analysis_fixture


ROOT = Path(__file__).resolve().parents[1]


def test_browser_gate_and_zero_dropout_plot_use_actual_analyzer_output(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed only for the generated site JavaScript check")
    lock_path, _, paths, _, results, status = analysis_fixture(
        tmp_path, 100, "confirmatory"
    )
    summary_path = tmp_path / "aggregate.json"
    summary = analyze_results(
        results, lock_path, summary_path, status, block_paths=paths
    )
    state = json.loads(status.read_text())
    assert (
        state["empirical_summary_sha256"]
        == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    )
    html = runpy.run_path(str(ROOT / "scripts/build_site.py"))["HTML"]
    snippet = html[
        html.index("let empiricalSummary=null;") : html.index(
            "fetch('data/project_status.json')"
        )
    ]
    payload = {"source": snippet, "summary": summary, "status": state}
    program = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const payload=JSON.parse(fs.readFileSync(0,'utf8'));
let plotted;
const elements={empiricalEndpoint:{value:'mean_prequential_log_score',addEventListener(){}},empiricalDirection:{}};
const context={document:{getElementById:id=>elements[id]},Plotly:{react:(id,data)=>{plotted=data}},
               layout:{xaxis:{},yaxis:{}},config:{},cyan:'#00ffff',amber:'#ffcc00',grid:'#222222'};
vm.createContext(context);vm.runInContext(payload.source,context);
assert(context.completeEmpiricalSummary(payload.status,payload.summary));
const small=JSON.parse(JSON.stringify(payload.summary));small.blocks=2;small.executions=12;
assert(!context.completeEmpiricalSummary(payload.status,small));
const stale=JSON.parse(JSON.stringify(payload.status));stale.analysis_lock_sha256='f'.repeat(64);
assert(!context.completeEmpiricalSummary(stale,payload.summary));
const pending=JSON.parse(JSON.stringify(payload.status));pending.empirical_results_available=false;
assert(!context.completeEmpiricalSummary(pending,payload.summary));
const invalid=JSON.parse(JSON.stringify(payload.summary));
invalid.analysis.mean_prequential_log_score.paired_architecture_by_dropout['0.0'].estimate=null;
assert(!context.completeEmpiricalSummary(payload.status,invalid));
vm.runInContext('empiricalSummary='+JSON.stringify(payload.summary)+';renderEmpirical()',context);
assert.equal(plotted[0].y[0],payload.summary.analysis.mean_prequential_log_score.paired_architecture_by_dropout['0.0'].estimate);
assert.equal(plotted[0].x[0],0);assert(plotted[0].y.every(Number.isFinite));
"""
    result = subprocess.run(
        [node, "-e", program],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
