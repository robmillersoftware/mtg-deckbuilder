"""RQ entry point for Forge simulation runs (consumed by the sim-worker)."""

from uuid import UUID

from app.jobs.rq_tasks import _run_async


def run_simulation(run_id: str) -> None:
    from app.services.sim_runs import execute
    _run_async(execute(UUID(run_id)))
