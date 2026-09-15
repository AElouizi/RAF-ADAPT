"""Mémoire partagée : Supabase si disponible, sinon JSON local (même contrat)."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bvc_recommender.agents.contracts import JobStatus
from bvc_recommender.config import (
    PROJECT_ROOT,
    TABLE_AGENT_BACKTEST_MONTHLY,
    TABLE_AGENT_EVALUATION_KPIS,
    TABLE_AGENT_EVENTS,
    TABLE_AGENT_JOBS,
    TABLE_AGENT_LOGS,
    TABLE_AGENT_PORTFOLIO_HOLDINGS,
    TABLE_AGENT_RECOMMENDATIONS,
    TABLE_AGENT_RESULT_REGISTRY,
)
from bvc_recommender.data.loader import get_service_role_client, get_supabase_client

logger = logging.getLogger(__name__)

LOCAL_DIR = PROJECT_ROOT / "bvc_recommender" / "data" / "agent_memory"
BATCH = 400
_LOCAL_ONLY = False


def set_local_only(enabled: bool) -> None:
    global _LOCAL_ONLY
    _LOCAL_ONLY = bool(enabled)


def _write_client():
    if _LOCAL_ONLY or os.getenv("BVC_AGENTS_LOCAL_ONLY", "").strip().lower() in {"1", "true", "yes"}:
        return None
    try:
        sr = get_service_role_client()
        if sr is not None:
            return sr
        return get_supabase_client()
    except Exception as exc:
        logger.warning("Client Supabase indisponible : %s", exc)
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def _append_jsonl(name: str, row: dict[str, Any]) -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCAL_DIR / name
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(name: str) -> list[dict[str, Any]]:
    path = LOCAL_DIR / name
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def create_job(
    *,
    agent_name: str,
    task_type: str,
    workflow_id: str,
    period_start: str,
    period_end: str,
    configuration: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    job_id = _new_id()
    row = {
        "id": job_id,
        "workflow_id": workflow_id,
        "agent_name": agent_name,
        "task_type": task_type,
        "period_start": period_start,
        "period_end": period_end,
        "configuration": configuration,
        "status": JobStatus.PENDING.value,
        "started_at": None,
        "completed_at": None,
        "error_message": None,
        "metadata": metadata or {},
        "created_at": _now(),
    }
    _append_jsonl("jobs.jsonl", row)
    client = _write_client()
    if client is not None:
        payload = dict(row)
        payload["metadata"] = metadata or {}
        try:
            client.table(TABLE_AGENT_JOBS).insert(payload).execute()
        except Exception as exc:
            logger.warning("Insert agent_jobs échoué (local conservé) : %s", exc)
            add_log(job_id, workflow_id, agent_name, "WARNING", f"Supabase jobs: {exc}")
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    fields = {k: v for k, v in fields.items() if v is not None or k == "error_message"}
    patch = dict(fields)
    patch["id"] = job_id
    patch["updated_at"] = _now()
    _append_jsonl("job_updates.jsonl", patch)
    client = _write_client()
    if client is None:
        return
    try:
        data = {k: v for k, v in fields.items()}
        if "metadata" in data and not isinstance(data["metadata"], (dict, list)):
            data["metadata"] = {"value": data["metadata"]}
        client.table(TABLE_AGENT_JOBS).update(data).eq("id", job_id).execute()
    except Exception as exc:
        logger.warning("Update agent_jobs échoué : %s", exc)


def add_log(
    job_id: str | None,
    workflow_id: str | None,
    agent_name: str,
    level: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    row = {
        "id": _new_id(),
        "job_id": job_id,
        "workflow_id": workflow_id,
        "agent_name": agent_name,
        "level": level,
        "message": message,
        "payload": payload or {},
        "created_at": _now(),
    }
    _append_jsonl("logs.jsonl", row)
    client = _write_client()
    if client is None:
        return
    try:
        client.table(TABLE_AGENT_LOGS).insert(row).execute()
    except Exception as exc:
        logger.debug("Insert agent_logs échoué : %s", exc)


def emit_event(
    *,
    workflow_id: str,
    job_id: str | None,
    event_type: str,
    producer_agent: str,
    payload: dict[str, Any] | None = None,
) -> None:
    row = {
        "id": _new_id(),
        "workflow_id": workflow_id,
        "job_id": job_id,
        "event_type": event_type,
        "producer_agent": producer_agent,
        "payload": payload or {},
        "created_at": _now(),
    }
    _append_jsonl("events.jsonl", row)
    client = _write_client()
    if client is None:
        return
    try:
        client.table(TABLE_AGENT_EVENTS).insert(row).execute()
    except Exception as exc:
        logger.warning("Insert agent_events échoué : %s", exc)


def register_artifact(
    *,
    run_id: str,
    agent_name: str,
    configuration: str,
    period_start: str,
    period_end: str,
    model_version: str,
    parameters: dict[str, Any],
    artifact_path: str,
    checksum_sha256: str | None,
    status: str,
) -> None:
    row = {
        "id": _new_id(),
        "run_id": run_id,
        "agent_name": agent_name,
        "configuration": configuration,
        "period_start": period_start,
        "period_end": period_end,
        "model_version": model_version,
        "parameters": parameters,
        "artifact_path": artifact_path,
        "checksum_sha256": checksum_sha256,
        "status": status,
        "produced_at": _now(),
    }
    _append_jsonl("registry.jsonl", row)
    client = _write_client()
    if client is None:
        return
    try:
        client.table(TABLE_AGENT_RESULT_REGISTRY).insert(row).execute()
    except Exception as exc:
        logger.warning("Insert registry échoué : %s", exc)


def upsert_dataframe(table: str, df: pd.DataFrame, *, run_id: str | None = None) -> dict[str, Any]:
    suffix = f"_{run_id}" if run_id else ""
    local_path = LOCAL_DIR / "tables" / f"{table}{suffix}.parquet"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_path, index=False)
    result = {"ok_local": True, "path": str(local_path), "rows": int(len(df)), "ok_supabase": False}
    client = _write_client()
    if client is None or df.empty:
        return result
    send = df.copy()
    if "parameters" in send.columns:
        send["parameters"] = send["parameters"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    payload = json.loads(send.to_json(orient="records", date_format="iso"))
    written = 0
    try:
        for i in range(0, len(payload), BATCH):
            client.table(table).insert(payload[i : i + BATCH]).execute()
            written += len(payload[i : i + BATCH])
        result["ok_supabase"] = True
        result["supabase_rows"] = written
    except Exception as exc:
        result["supabase_error"] = str(exc)
        logger.warning("Insert %s échoué : %s", table, exc)
    return result


def list_jobs(limit: int = 80) -> pd.DataFrame:
    client = _write_client()
    if client is not None:
        try:
            resp = (
                client.table(TABLE_AGENT_JOBS)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            if resp.data:
                return pd.DataFrame(resp.data)
        except Exception:
            pass
    rows = _read_jsonl("jobs.jsonl")
    updates = _read_jsonl("job_updates.jsonl")
    by_id = {r["id"]: dict(r) for r in rows}
    for u in updates:
        jid = u.get("id")
        if jid in by_id:
            by_id[jid].update({k: v for k, v in u.items() if k != "updated_at"})
    df = pd.DataFrame(list(by_id.values()))
    if df.empty:
        return df
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)
    return df.head(limit)


def list_logs(workflow_id: str | None = None, limit: int = 200) -> pd.DataFrame:
    client = _write_client()
    if client is not None:
        try:
            q = client.table(TABLE_AGENT_LOGS).select("*").order("created_at", desc=True).limit(limit)
            if workflow_id:
                q = q.eq("workflow_id", workflow_id)
            resp = q.execute()
            if resp.data:
                return pd.DataFrame(resp.data)
        except Exception:
            pass
    rows = _read_jsonl("logs.jsonl")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if workflow_id and "workflow_id" in df.columns:
        df = df[df["workflow_id"] == workflow_id]
    return df.sort_values("created_at", ascending=False).head(limit) if "created_at" in df.columns else df


def list_events(workflow_id: str | None = None, limit: int = 100) -> pd.DataFrame:
    rows = _read_jsonl("events.jsonl")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if workflow_id and "workflow_id" in df.columns:
        df = df[df["workflow_id"] == workflow_id]
    return df.sort_values("created_at", ascending=False).head(limit) if "created_at" in df.columns else df


TABLES = {
    "recommendations": TABLE_AGENT_RECOMMENDATIONS,
    "holdings": TABLE_AGENT_PORTFOLIO_HOLDINGS,
    "backtest_monthly": TABLE_AGENT_BACKTEST_MONTHLY,
    "kpis": TABLE_AGENT_EVALUATION_KPIS,
}
