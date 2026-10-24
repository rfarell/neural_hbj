"""A tiny train -> checkpoint -> reload -> evaluate pass for each problem."""

import math

import pytest

from motion_control import evaluate as mc_evaluate
from motion_control.problem import MotionControl
from neural_hjb.experiment import collect_runs, load_run, train_run
from resource_allocation import evaluate as ra_evaluate
from resource_allocation.problem import ResourceAllocation


@pytest.mark.parametrize(
    "problem, make_problem, evaluate",
    [
        (MotionControl(state_dim=3), MotionControl, mc_evaluate.evaluate),
        (ResourceAllocation(), ResourceAllocation, ra_evaluate.evaluate),
    ],
)
@pytest.mark.parametrize("variant", ["full", "no_hjb"])
def test_train_reload_evaluate(tmp_path, problem, make_problem, evaluate, variant):
    path = train_run(problem, variant, seed=0, model_dir=tmp_path, stages=2, epochs_per_stage=2,
                     batch_size=16, rollout_steps=4, log_every=None)
    assert collect_runs(tmp_path) == {(variant, problem.tag, 0): path}
    assert path.with_suffix(".csv").exists()

    reloaded, policy, value, ckpt = load_run(path, make_problem)
    assert reloaded.config() == problem.config()
    assert ckpt["variant"] == variant
    metrics = evaluate(reloaded, policy, value)
    assert all(math.isfinite(v) for v in metrics.values())
