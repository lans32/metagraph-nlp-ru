import time

from metagraph_nlp.profiling import PipelineMetrics, StageMetrics, measure_stage


def test_measure_stage_records_wall_time():
    metrics = PipelineMetrics()
    with measure_stage("test", metrics) as sm:
        time.sleep(0.01)
        sm.output_count = 42
    assert len(metrics.stages) == 1
    assert metrics.stages[0].wall_seconds >= 0.005
    assert metrics.stages[0].output_count == 42
    assert metrics.total_wall_seconds > 0


def test_measure_stage_records_memory():
    metrics = PipelineMetrics()
    with measure_stage("alloc", metrics):
        _ = [0] * 100_000
    assert metrics.stages[0].peak_memory_kb > 0


def test_pipeline_metrics_to_dict():
    metrics = PipelineMetrics()
    metrics.stages.append(StageMetrics(stage="a", wall_seconds=0.1234, peak_memory_kb=512.5, output_count=3))
    metrics.total_wall_seconds = 0.1234

    d = metrics.to_dict()
    assert d["total_wall_seconds"] == 0.1234
    assert len(d["stages"]) == 1
    assert d["stages"][0]["stage"] == "a"
    assert d["stages"][0]["output_count"] == 3
