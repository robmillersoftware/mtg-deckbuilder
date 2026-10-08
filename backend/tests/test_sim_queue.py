"""The simulation queue has its own worker; the API needs to know whether it is up."""

from types import SimpleNamespace

from app.core import queue as q


def test_sim_worker_running_only_counts_sim_queue_workers(monkeypatch):
    monkeypatch.setattr(q.Worker, "all", lambda connection: [SimpleNamespace(queue_names=lambda: ["spellbook_jobs"])])
    assert not q.sim_worker_running()
    monkeypatch.setattr(q.Worker, "all", lambda connection: [SimpleNamespace(queue_names=lambda: ["spellbook_sim"])])
    assert q.sim_worker_running()


def test_sim_worker_running_is_false_when_redis_fails(monkeypatch):
    def down(connection):
        raise ConnectionError("redis down")
    monkeypatch.setattr(q.Worker, "all", down)
    assert not q.sim_worker_running()
