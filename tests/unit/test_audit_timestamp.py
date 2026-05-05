from datetime import datetime, timezone

from metagraph_nlp.provenance.audit import AuditLog


def test_record_adds_iso_timestamp():
    log = AuditLog()
    log.record("test_stage", "test_rule")

    assert len(log.events) == 1
    ts = log.events[0].timestamp
    assert ts is not None
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None


def test_timestamp_is_utc():
    log = AuditLog()
    before = datetime.now(timezone.utc)
    log.record("s", "r")
    after = datetime.now(timezone.utc)

    ts = datetime.fromisoformat(log.events[0].timestamp)
    assert before <= ts <= after
