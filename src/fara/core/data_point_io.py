"""Incremental DataPoint writer and reader.

DataPointWriter persists a DataPoint incrementally to a directory:
  metadata.json        - DataPointMetadata: run_id, created_at, stats
  task.json            - Task (written once)
  solver_log/
    status.json        - SolverStatus + Outcome (overwritten on change)
    events.jsonl       - Observation/Action events (append-only)
  verification.jsonl   - VerificationResult entries (append-only)

DataPointReader reconstructs a DataPoint from such a directory.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import TypeAdapter

from .data_point import (
    Action,
    DataPoint,
    DataPointMetadata,
    Event,
    ObservationEvent,
    Outcome,
    SolverLog,
    SolverStatus,
    Task,
    VerificationResult,
    VerificationResultEvent,
)

_event_adapter = TypeAdapter(Event)
_verification_adapter = TypeAdapter(VerificationResultEvent)


def validate_verification_result(payload: Any) -> VerificationResultEvent:
    """Validate a JSON-shaped payload into a :class:`VerificationResultEvent`."""
    return _verification_adapter.validate_python(payload)


_TASK_FILE = "task.json"
_METADATA_FILE = "metadata.json"
_VERIFICATION_FILE = "verification.jsonl"
_SOLVER_LOG_DIR = "solver_log"
_EVENTS_FILE = "events.jsonl"
_STATUS_FILE = "status.json"


def _write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tmp + rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class DataPointWriter:
    """Incrementally persists a DataPoint to a directory.

    Use the factory methods to create instances:
        DataPointWriter.create(output_dir, task)  - start a new data point
        DataPointWriter.resume(output_dir)         - continue from existing directory
    """

    def __init__(self, output_dir: Path, data_point: DataPoint) -> None:
        self._output_dir = output_dir
        self._solver_log_dir = self._output_dir / _SOLVER_LOG_DIR
        self._events_path = self._solver_log_dir / _EVENTS_FILE
        self._data_point = data_point

    @classmethod
    def create(
        cls,
        output_dir: str | os.PathLike,
        task: Task,
        run_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> "DataPointWriter":
        """Create a new DataPointWriter for a fresh data point.

        Raises FileExistsError if a data point already exists in output_dir.
        """
        out = Path(output_dir)
        if out.exists() and (out / _TASK_FILE).exists():
            raise FileExistsError(
                f"Data point already exists in {out} ({_TASK_FILE} found). "
                "Use DataPointWriter.resume() to continue an existing data point, "
                "or remove the directory first."
            )
        metadata_kwargs: Dict[str, Any] = {}
        if run_id is not None:
            metadata_kwargs["run_id"] = run_id
        if created_at is not None:
            metadata_kwargs["created_at"] = created_at

        dp = DataPoint(task=task, metadata=DataPointMetadata(**metadata_kwargs))
        writer = cls(out, dp)
        writer.finalize()
        return writer

    @classmethod
    def resume(cls, output_dir: str | os.PathLike) -> "DataPointWriter":
        """Resume writing to an existing data point directory."""
        out = Path(output_dir)
        dp = DataPointReader.read(out)
        return cls(out, dp)

    @property
    def data_point(self) -> DataPoint:
        return self._data_point

    def add_event(self, event: Union["ObservationEvent", Action]) -> DataPoint:
        # Write to disk first so the log captures every event,
        # including ones that cause the in-memory model to reject.
        self._solver_log_dir.mkdir(parents=True, exist_ok=True)
        with open(self._events_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
            f.flush()

        if isinstance(event, Action):
            self._data_point.solver_log.add_action(event)
        else:
            self._data_point.solver_log.add_observation(event)
        return self._data_point

    def add_verification_result(self, result: VerificationResult) -> DataPoint:
        self._data_point.verification[result.verifier_name] = result
        with open(self._output_dir / _VERIFICATION_FILE, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
            f.flush()
        return self._data_point

    def finalize(self) -> DataPoint:
        """Write all non-streaming state: task, metadata, solver status/outcome."""
        dp = self._data_point
        self._output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self._output_dir / _TASK_FILE, dp.task.to_dict())
        _write_json(self._output_dir / _METADATA_FILE, dp.metadata.to_dict())

        self._solver_log_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            self._solver_log_dir / _STATUS_FILE,
            {
                "status": dp.solver_log.status.value,
                "outcome": dp.solver_log.outcome.to_dict()
                if dp.solver_log.outcome
                else None,
            },
        )
        return dp


class DataPointReader:
    """Reconstructs a DataPoint from a directory written by DataPointWriter."""

    @staticmethod
    def read(input_dir: str | os.PathLike) -> DataPoint:
        d = Path(input_dir)

        # Task (required)
        task = Task.from_dict(_load_json(d / _TASK_FILE))

        # Metadata
        meta_path = d / _METADATA_FILE
        metadata = (
            DataPointMetadata.from_dict(_load_json(meta_path))
            if meta_path.exists()
            else DataPointMetadata()
        )

        # Solver log
        solver_log = SolverLog()
        solver_log_dir = d / _SOLVER_LOG_DIR

        events_path = solver_log_dir / _EVENTS_FILE
        if events_path.exists():
            with open(events_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    event = _event_adapter.validate_python(json.loads(line))
                    if isinstance(event, Action):
                        solver_log.add_action(event)
                    else:
                        solver_log.add_observation(event)

        status_path = solver_log_dir / _STATUS_FILE
        if status_path.exists():
            status_data = _load_json(status_path)
            solver_log.status = SolverStatus(status_data["status"])
            if status_data.get("outcome") is not None:
                solver_log.outcome = Outcome.from_dict(status_data["outcome"])

        # Verification
        verification: Dict[str, VerificationResult] = {}
        verif_path = d / _VERIFICATION_FILE
        if verif_path.exists():
            with open(verif_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    result = _verification_adapter.validate_python(json.loads(line))
                    verification[result.verifier_name] = result

        return DataPoint(
            task=task,
            solver_log=solver_log,
            verification=verification,
            metadata=metadata,
        )


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)
