from __future__ import annotations
import argparse, json
from pathlib import Path
from .config import load_config, write_default_config, DEFAULT_OPTIMIZER_JSON
from .preprocessing.openweather_resampler import resample_to_5min
from .aggregation.daily_climate_aggregator import aggregate_5m_to_growth_daily
from .services.offline_worker import OfflineWorker
from .services.work_offline_evaluator import WorkOfflineEvaluator
from .services.online_worker import OnlineWorker
from .learning.ccll_sel import build_ccll_from_5m
from .learning.sarg_sel import build_sarg_reference_library
from .learning.setd_kstore import export_setd_kstore
from .learning.knowledge_reset import reset_learned_knowledge
from .evaluation.reports import rebuild_reports_from_artifacts


def _cfg(args):
    return load_config(args.config)


def prepare_dataset(args):
    cfg = _cfg(args)
    dataset = args.dataset or cfg.dataset_csv
    out_5m = args.out_5m or cfg.prepared_5m_csv
    out_daily = args.out_daily or cfg.prepared_daily_csv
    df5 = resample_to_5min(dataset, out_5m)
    daily = aggregate_5m_to_growth_daily(df5, season_start_date=args.growth_start_date, output_csv=out_daily)
    print(json.dumps({"prepared_5m_rows": len(df5), "prepared_daily_rows": len(daily), "prepared_5m_csv": out_5m, "prepared_daily_csv": out_daily}, indent=2))


def prepare_ccll_sel(args):
    cfg = _cfg(args)
    src = args.prepared_5m or cfg.prepared_5m_csv
    if not Path(src).exists():
        resample_to_5min(cfg.dataset_csv, cfg.prepared_5m_csv)
        src = cfg.prepared_5m_csv
    out_dir = args.output_dir or cfg.prepared_dir
    k = args.contexts if args.contexts is not None else cfg.ccll_context_count
    result = build_ccll_from_5m(
        src,
        out_dir,
        k_hint=k,
        max_iter=args.max_iter if args.max_iter is not None else cfg.ccll_max_iter,
        seed=args.seed if args.seed is not None else cfg.ccll_random_seed,
        feature_columns=cfg.ccll_feature_columns,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def prepare_sarg_sel(args):
    cfg = _cfg(args)
    diets_text = args.diets if args.diets else ",".join(str(x) for x in cfg.sarg_diets)
    diets = [int(x) for x in diets_text.split(',') if x.strip()]
    daily_csv = args.daily_csv or cfg.prepared_daily_csv
    if not Path(daily_csv).exists():
        # Build the daily growth input first so SARG references are generated
        # from the same climate/growth horizon used by TRAIN.
        if not Path(cfg.prepared_5m_csv).exists():
            resample_to_5min(cfg.dataset_csv, cfg.prepared_5m_csv)
        df5 = None
        try:
            import pandas as pd
            df5 = pd.read_csv(cfg.prepared_5m_csv)
        except Exception:
            df5 = resample_to_5min(cfg.dataset_csv, cfg.prepared_5m_csv)
        aggregate_5m_to_growth_daily(df5, season_start_date=args.growth_start_date, output_csv=daily_csv)
    result = build_sarg_reference_library(
        args.output_dir or cfg.prepared_dir,
        diets=diets,
        top_k=args.top_k if args.top_k is not None else cfg.sarg_top_k,
        daily_csv=daily_csv,
        simulator_mode=True,
        horizon_days=args.horizon_days if args.horizon_days is not None else getattr(cfg, "sarg_reference_horizon_days", 1000),
        breed=args.breed if args.breed is not None else cfg.default_breed,
        scale=args.scale if args.scale is not None else cfg.default_scale,
        sex_animal=args.sex_animal if args.sex_animal is not None else cfg.default_sex_animal,
        housing=args.housing if args.housing is not None else cfg.default_housing,
        case_id=args.case_id if args.case_id is not None else cfg.default_case_id,
        timeout_seconds=args.timeout_seconds if args.timeout_seconds is not None else getattr(cfg, "sarg_reference_timeout_seconds", 180),
        max_programs=args.max_programs if args.max_programs is not None else getattr(cfg, "sarg_reference_max_programs", None),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def export_kstore(args):
    cfg = _cfg(args)
    result = export_setd_kstore(
        args.sqlite or cfg.sqlite_path,
        args.output_dir or cfg.setd_kstore_dir,
        ccll_path=args.ccll or cfg.ccll_library_json,
        sarg_path=args.sarg or cfg.sarg_reference_library_json,
        tddt_version=Path(cfg.project_root).name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))




def rebuild_reports(args):
    cfg = _cfg(args)
    result = rebuild_reports_from_artifacts(
        args.report_dir or cfg.report_dir,
        ss_kstore_dir=args.ss_kstore_dir or cfg.ss_kstore_dir,
        growth_csv=args.growth_csv,
        config=cfg,
        mode_name="REBUILD-OFFLINE",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

def reset_knowledge(args):
    cfg = _cfg(args)
    result = reset_learned_knowledge(
        cfg,
        dry_run=args.dry_run,
        include_reports=not args.keep_reports,
        include_prepared_knowledge=args.include_prepared_knowledge,
        recreate_database=not args.no_recreate_database,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

def init_config(args):
    write_default_config(args.output)
    print(f"Wrote optimizer config: {args.output}")


def train(args):
    cfg = _cfg(args)
    if not Path(cfg.prepared_ccll_5m_csv).exists() and not Path(cfg.prepared_5m_csv).exists():
        resample_to_5min(cfg.dataset_csv, cfg.prepared_5m_csv)
    result = OfflineWorker(cfg).run(
        args.growth_start_date,
        end_date=args.growth_end_date,
        max_steps=args.max_steps,
        case_id=args.case_id if args.case_id is not None else cfg.default_case_id,
        show_progress=not args.no_progress,
        light_on_hour=args.light_on_hour if args.light_on_hour is not None else cfg.light_on_hour,
        light_on_minute=args.light_on_minute if args.light_on_minute is not None else cfg.light_on_minute,
        light_hours_on=args.light_hours_on if args.light_hours_on is not None else cfg.light_hours_on,
        light_hours_off=args.light_hours_off if args.light_hours_off is not None else cfg.light_hours_off,
        mode_name="TRAIN",
    )
    print(json.dumps(result, indent=2, default=str))


def work_offline(args):
    cfg = _cfg(args)
    if not Path(cfg.prepared_ccll_5m_csv).exists() and not Path(cfg.prepared_5m_csv).exists():
        resample_to_5min(cfg.dataset_csv, cfg.prepared_5m_csv)
    result = WorkOfflineEvaluator(cfg).run(
        args.growth_start_date,
        end_date=args.growth_end_date,
        max_steps=args.max_steps,
        case_id=args.case_id if args.case_id is not None else cfg.default_case_id,
        show_progress=not args.no_progress,
        light_on_hour=args.light_on_hour if args.light_on_hour is not None else cfg.light_on_hour,
        light_on_minute=args.light_on_minute if args.light_on_minute is not None else cfg.light_on_minute,
        light_hours_on=args.light_hours_on if args.light_hours_on is not None else cfg.light_hours_on,
        light_hours_off=args.light_hours_off if args.light_hours_off is not None else cfg.light_hours_off,
    )
    print(json.dumps(result, indent=2, default=str))


def work_online(args):
    cfg = _cfg(args)
    result = OnlineWorker(cfg).run(args.growth_start_date, max_steps=args.max_steps)
    print(json.dumps(result, indent=2, default=str))


def out_service(args):
    print(json.dumps({"mode": "OUT_SERVICE", "status": "NO_ACTION", "actuator_command": {"ventilation_group_pct": 0, "heating_group_pct": 0, "light_on": False}}, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="tddt-optimizer", description="Trans-Domain Digital Twin optimizer for livestock climate/growth coupling")
    p.add_argument("--config", default=str(DEFAULT_OPTIMIZER_JSON), help="Central optimizer JSON config path")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("init-config"); s.add_argument("--output", default=str(DEFAULT_OPTIMIZER_JSON)); s.set_defaults(func=init_config)
    s = sub.add_parser("prepare-dataset"); s.add_argument("--dataset", default=None); s.add_argument("--growth-start-date", default=None); s.add_argument("--out-5m", default=None); s.add_argument("--out-daily", default=None); s.set_defaults(func=prepare_dataset)
    s = sub.add_parser("prepare-ccll-sel", help="Build CCLL-SEL climate context local library using nearest-centroid clustering"); s.add_argument("--prepared-5m", default=None); s.add_argument("--output-dir", default=None); s.add_argument("--contexts", type=int, default=None, help="Number of nearest-centroid climate contexts"); s.add_argument("--max-iter", type=int, default=None); s.add_argument("--seed", type=int, default=None); s.set_defaults(func=prepare_ccll_sel)
    s = sub.add_parser("prepare-sarg-sel", help="Build SARG-SEL growth-simulator-generated stage-aware reference library")
    s.add_argument("--output-dir", default=None)
    s.add_argument("--diets", default=None)
    s.add_argument("--top-k", type=int, default=None)
    s.add_argument("--daily-csv", default=None, help="Daily growth input CSV used for LiGAPS reference generation")
    s.add_argument("--growth-start-date", default="2015-01-01", help="Used if the daily CSV must be regenerated")
    s.add_argument("--horizon-days", type=int, default=None, help="Full-horizon reference length; default from optimizer.json")
    s.add_argument("--breed", type=int, default=None)
    s.add_argument("--scale", type=int, default=None)
    s.add_argument("--sex-animal", type=int, default=None)
    s.add_argument("--housing", type=int, default=None)
    s.add_argument("--case-id", type=int, default=None)
    s.add_argument("--timeout-seconds", type=int, default=None)
    s.add_argument("--max-programs", type=int, default=None, help="Optional cap for generated reference programs; all summaries remain simulator-derived")
    s.set_defaults(func=prepare_sarg_sel)
    s = sub.add_parser("export-setd-kstore", help="Export SETD-KStore knowledge package from the runtime SQLite database"); s.add_argument("--sqlite", default=None); s.add_argument("--output-dir", default=None); s.add_argument("--ccll", default=None); s.add_argument("--sarg", default=None); s.set_defaults(func=export_kstore)
    s = sub.add_parser("rebuild-reports", help="Rebuild all reports from existing SS-KStore/SETD-KStore artifacts without rerunning TRAIN")
    s.add_argument("--report-dir", default=None)
    s.add_argument("--ss-kstore-dir", default=None)
    s.add_argument("--growth-csv", default=None)
    s.set_defaults(func=rebuild_reports)
    s = sub.add_parser("reset-knowledge", help="Reset learned episode knowledge: SQLite runtime, SETD-KStore, RL snapshot, and reports")
    s.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting anything")
    s.add_argument("--keep-reports", action="store_true", help="Keep reports/ while resetting learned model memory")
    s.add_argument("--include-prepared-knowledge", action="store_true", help="Also remove prepared CCLL/SARG reference artifacts so they must be regenerated")
    s.add_argument("--no-recreate-database", action="store_true", help="Do not recreate an empty SQLite database after reset")
    s.set_defaults(func=reset_knowledge)
    def add_period_and_lighting_args(sp):
        sp.add_argument("--growth-start-date", required=True, help="Start datetime/date of the growth/climate period")
        sp.add_argument("--growth-end-date", default=None, help="Optional end datetime/date of the period. Date-only values are inclusive.")
        sp.add_argument("--max-steps", type=int, default=None, help="Optional hard cap on 5-minute rows. If omitted, the selected date range is used.")
        sp.add_argument("--case-id", type=int, default=None)
        sp.add_argument("--no-progress", action="store_true", help="Disable terminal progress output")
        sp.add_argument("--light-on-hour", type=int, default=None, help="Hour when barn lighting turns on, 0-23")
        sp.add_argument("--light-on-minute", type=int, default=None, help="Minute when barn lighting turns on, 0-59")
        sp.add_argument("--light-hours-on", type=float, default=None, help="Lighting ON duration in hours, e.g. 14")
        sp.add_argument("--light-hours-off", type=float, default=None, help="Lighting OFF duration in hours, kept for traceability/reporting")

    s = sub.add_parser("train", help="TRAIN mode: previous WORK-OFFLINE training loop, unchanged internally except for mode name")
    add_period_and_lighting_args(s)
    s.set_defaults(func=train)

    s = sub.add_parser("work-offline", help="New WORK-OFFLINE mode: offline learned-policy evaluation without funnel/training")
    add_period_and_lighting_args(s)
    s.set_defaults(func=work_offline)
    s = sub.add_parser("work-online"); s.add_argument("--growth-start-date", required=True); s.add_argument("--max-steps", type=int, default=1); s.set_defaults(func=work_online)
    s = sub.add_parser("out-service"); s.set_defaults(func=out_service)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
