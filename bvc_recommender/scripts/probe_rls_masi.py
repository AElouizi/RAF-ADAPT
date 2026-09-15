"""Probe RLS vs données MASI 2010+ — comparaison REST anon vs patterns id/date."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from bvc_recommender.data.loader import fetch_table, get_supabase_client  # noqa: E402

load_env_file(ROOT / ".env")

TABLE = "market_data_indices_historique"
USER_UUID = "9c49e61a-e4e5-40a4-b58d-80ba95b869ea"
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def count(client, **kwargs) -> int | None:
    q = client.table(TABLE).select("*", count="exact").limit(0)
    for key, value in kwargs.get("filters", {}).items():
        q = q.eq(key, value)
    for key, value in kwargs.get("gte", {}).items():
        q = q.gte(key, value)
    for key, value in kwargs.get("lte", {}).items():
        q = q.lte(key, value)
    for key, value in kwargs.get("lt", {}).items():
        q = q.lt(key, value)
    return q.execute().count


def analyze_ids(df) -> dict:
    uuid_ids: list[str] = []
    int_ids: list[str] = []
    other: list[str] = []
    for raw in df["id"].astype(str):
        if UUID_RE.match(raw):
            uuid_ids.append(raw)
        elif raw.isdigit():
            int_ids.append(raw)
        else:
            other.append(raw)

    by_type_date: dict[str, dict] = {}
    date_col = "date_index"
    for label, mask in [
        ("uuid", df["id"].astype(str).str.match(UUID_RE)),
        ("integer", df["id"].astype(str).str.fullmatch(r"\d+")),
    ]:
        sub = df[mask]
        if sub.empty:
            by_type_date[label] = {"count": 0}
        else:
            by_type_date[label] = {
                "count": int(len(sub)),
                "date_min": str(sub[date_col].min()),
                "date_max": str(sub[date_col].max()),
            }

    return {
        "total": len(df),
        "uuid_count": len(uuid_ids),
        "integer_count": len(int_ids),
        "other_count": len(other),
        "by_type_dates": by_type_date,
        "sample_uuid": uuid_ids[:3],
        "sample_int": int_ids[:3],
    }


def main() -> int:
    client = get_supabase_client()
    out: dict = {"table": TABLE, "hypothesis": "RLS filters old rows for anon"}

    out["counts"] = {
        "total": count(client),
        "masi": count(client, filters={"code_index": "MASI"}),
        "masi_gte_2010": count(
            client, filters={"code_index": "MASI"}, gte={"date_index": "2010-01-01"}
        ),
        "masi_lt_2022": count(
            client, filters={"code_index": "MASI"}, lt={"date_index": "2022-01-01"}
        ),
        "all_lt_2020": count(client, lt={"date_index": "2020-01-01"}),
        "all_lt_2022": count(client, lt={"date_index": "2022-01-01"}),
    }

    out["masi_earliest_5"] = (
        client.table(TABLE)
        .select("id,code_index,date_index,valeur_index,scraped_at")
        .eq("code_index", "MASI")
        .order("date_index")
        .limit(5)
        .execute()
        .data
    )

    out["masi_gte2010_first5"] = (
        client.table(TABLE)
        .select("id,code_index,date_index,valeur_index,scraped_at")
        .eq("code_index", "MASI")
        .gte("date_index", "2010-01-01")
        .order("date_index")
        .limit(5)
        .execute()
        .data
    )

    user_row = client.table(TABLE).select("*").eq("id", USER_UUID).execute()
    out["user_screenshot_uuid"] = {
        "id": USER_UUID,
        "found_via_anon": len(user_row.data or []),
        "row": (user_row.data or [None])[0],
    }

    df_masi = fetch_table(
        client, TABLE, filters={"code_index": "MASI"}, order_by="date_index", page_size=1000
    )
    out["masi_fetch_all"] = {
        "rows": len(df_masi),
        "matches_count_masi": len(df_masi) == out["counts"]["masi"],
        "date_min": str(df_masi["date_index"].min()) if len(df_masi) else None,
        "date_max": str(df_masi["date_index"].max()) if len(df_masi) else None,
        "id_analysis": analyze_ids(df_masi),
    }

    if len(df_masi):
        out["scraped_at_top"] = (
            df_masi["scraped_at"].astype(str).value_counts().head(5).to_dict()
        )

    df_all = fetch_table(client, TABLE, order_by="date_index", page_size=1000)
    out["all_indices_id_analysis"] = analyze_ids(df_all)
    out["all_indices_date_range"] = {
        "rows": len(df_all),
        "date_min": str(df_all["date_index"].min()) if len(df_all) else None,
        "date_max": str(df_all["date_index"].max()) if len(df_all) else None,
    }

    # Optional service_role cross-check (bypasses RLS) — never invent a key
    service_key = ""
    service_key_source: str | None = None
    for env_path, label in [
        (ROOT / ".env", "project .env"),
        (ROOT.parent / ".env.txt", "sibling .env.txt"),
    ]:
        if service_key:
            break
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("SUPABASE_SERVICE_ROLE_KEY="):
                service_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                service_key_source = label
                break
    if not service_key:
        service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if service_key:
            service_key_source = "process env"

    if service_key.startswith("eyJ"):
        from supabase import ClientOptions, create_client

        from bvc_recommender.data.loader import _build_httpx_client

        settings = __import__(
            "bvc_recommender.config", fromlist=["get_settings"]
        ).get_settings()
        sr_client = create_client(
            settings.supabase_url,
            service_key,
            ClientOptions(httpx_client=_build_httpx_client(settings.verify_ssl)),
        )
        out["service_role_crosscheck"] = {
            "configured": True,
            "key_source": service_key_source,
            "total": count(sr_client),
            "masi": count(sr_client, filters={"code_index": "MASI"}),
            "masi_lt_2022": count(
                sr_client, filters={"code_index": "MASI"}, lt={"date_index": "2022-01-01"}
            ),
            "user_uuid_found": len(
                sr_client.table(TABLE)
                .select("id")
                .eq("id", USER_UUID)
                .execute()
                .data
                or []
            ),
        }
    else:
        out["service_role_crosscheck"] = {
            "configured": False,
            "key_source": None,
            "hint": (
                "Ajoutez SUPABASE_SERVICE_ROLE_KEY dans .env : "
                "Dashboard → Project Settings → API → service_role (secret)"
            ),
        }

    # RLS inference
    counts = out["counts"]
    user_invisible_anon = out["user_screenshot_uuid"]["found_via_anon"] == 0
    sr = out.get("service_role_crosscheck", {})
    sr_same_as_anon = sr.get("masi") == counts["masi"] if sr.get("masi") is not None else None
    out["rls_inference"] = {
        "anon_sees_pre2022_masi": counts["masi_lt_2022"] > 0,
        "user_uuid_invisible_to_anon": user_invisible_anon,
        "count_equals_fetch": out["masi_fetch_all"]["matches_count_masi"],
        "service_role_same_counts_as_anon": sr_same_as_anon,
        "rls_likely_cause": user_invisible_anon
        and counts["masi_lt_2022"] == 0
        and sr_same_as_anon is False,
        "data_absent_not_rls": sr_same_as_anon is True and counts["masi_lt_2022"] == 0,
    }

    report_path = ROOT / "bvc_recommender" / "reports" / "masi_rls_probe.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
