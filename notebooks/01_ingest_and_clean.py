#!/usr/bin/env python3
"""
01_ingest_and_clean.py

Ingestão e limpeza inicial do MIMIC-IV-ED Assist.
- valida esquema básico com pandera
- checa unicidade/consistência de chaves
- corrige dtypes
- remove pediatria (opcional)
- filtro/recuperação de hadm_id (opcional)
- detecção simples de outliers e imputação 'nearest' dentro de stay
- gera profile HTML (ydata-profiling), parquet de saída e cleaning_report.json

Uso:
python src/01_ingest_and_clean.py \
  --master data/master_dataset.csv \
  --edstays data/edstays.csv \
  --admissions data/admissions.csv \
  --out-parquet data/ed_master_clean.parquet \
  --profile reports/profile_master.html \
  --report reports/cleaning_report.json \
  --drop-pediatrics \
  --require-hadm

Instalação prévia (recomendada):
pip install pandas pyarrow pandera ydata-profiling python-dateutil
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict
import importlib

import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, Check, DataFrameSchema
# ydata_profiling (profiling) - optional import via importlib to avoid static linter errors
try:
    ydata_profiling = importlib.import_module("ydata_profiling")
    ProfileReport = getattr(ydata_profiling, "ProfileReport", None)
    HAS_PROFILE = ProfileReport is not None
except Exception:
    ProfileReport = None
    HAS_PROFILE = False
    HAS_PROFILE = False

# logging
LOG = logging.getLogger("ingest_clean")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# ----------------------
# Schema: adaptável
# ----------------------
BASE_SCHEMA = DataFrameSchema(
    {
        "stay_id": Column(pa.String, nullable=False),
        "subject_id": Column(pa.Int, nullable=False),
        "hadm_id": Column(pa.Int, nullable=True),
        "intime": Column(pa.DateTime, nullable=False),
        "outtime": Column(pa.DateTime, nullable=True),
        "gender": Column(pa.String, nullable=True, checks=Check.isin(["M", "F"])),
        "race": Column(pa.String, nullable=True),
        "arrival_transport": Column(pa.String, nullable=True),
        "disposition": Column(pa.String, nullable=True),
        "anchor_age": Column(pa.Int, nullable=True),
        "anchor_year": Column(pa.Int, nullable=True),
        "dod": Column(pa.DateTime, nullable=True),
        "admittime": Column(pa.DateTime, nullable=True),
        "dischtime": Column(pa.DateTime, nullable=True),
        "deathtime": Column(pa.DateTime, nullable=True),
        # as colunas abaixo são exemplos — o schema aceita colunas extras
        # "age": Column(pa.Float, nullable=True),
        # "temperature": Column(pa.Float, nullable=True),
    },
    coerce=False,
    strict=False
)


# ----------------------
# Utilities
# ----------------------
def safe_read_csv(path: Path, nrows: Optional[int] = None, parse_dates: Optional[list] = None) -> pd.DataFrame:
    LOG.info("Carregando %s", path)
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    # tentamos parsear qualquer coluna datetime padrão
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    else:
        # heurística simples
        for c in ["intime", "outtime", "charttime", "chart_time", "admittime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def ensure_dtypes(master: pd.DataFrame) -> pd.DataFrame:
    if "stay_id" in master.columns:
        master["stay_id"] = master["stay_id"].astype(str)
    if "subject_id" in master.columns:
        master["subject_id"] = pd.to_numeric(master["subject_id"], errors="coerce").astype("Int64")
    if "hadm_id" in master.columns:
        master["hadm_id"] = pd.to_numeric(master["hadm_id"], errors="coerce").astype("Int64")
    # datetimes normalized
    for c in ["intime", "outtime", "charttime", "admittime"]:
        if c in master.columns:
            master[c] = pd.to_datetime(master[c], errors="coerce")
    return master


def validate_basic_schema(df: pd.DataFrame) -> Dict:
    """Valida o schema base; retorna resumo de validação"""
    res = {"valid": True, "errors": []}
    try:
        BASE_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        res["valid"] = False
        # resumir as mensagens
        res["errors"] = str(e.failure_cases.head(50).to_dict())
    return res


def check_uniqueness(edstays: pd.DataFrame) -> Dict:
    res = {}
    if "stay_id" in edstays.columns:
        res["stay_id_unique"] = bool(edstays["stay_id"].is_unique)
        res["n_stays"] = int(edstays.shape[0])
    else:
        res["stay_id_unique"] = False
        res["n_stays"] = 0
    return res


def timestamp_checks(df: pd.DataFrame) -> Dict:
    out = {}
    if ("intime" in df.columns) and ("outtime" in df.columns):
        count_inv = int((df["intime"].notna()) & (df["outtime"].notna()) & (df["intime"] > df["outtime"]).sum())
        out["intime_after_outtime_count"] = count_inv
    # charttime before intime?
    if "charttime" in df.columns and "intime" in df.columns:
        cnt = int(((df["charttime"].notna()) & (df["intime"].notna()) & (df["charttime"] < df["intime"])).sum())
        out["chart_before_intime_count"] = cnt
    return out


def missingness_report(df: pd.DataFrame, top_n: int = 20) -> Dict:
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    return {"n_columns": int(len(miss)), "top_missing": miss.head(top_n).to_dict()}


def remove_pediatrics(df: pd.DataFrame, age_col: str = "age", min_age: int = 18) -> Dict:
    if age_col not in df.columns:
        return {"removed": 0, "reason": "age_column_missing"}
    before = len(df)
    cleaned = df[df[age_col].notna() & (df[age_col] >= min_age)].copy()
    removed = before - len(cleaned)
    LOG.info("Removed %d pediatric rows (age < %d)", removed, min_age)
    return {"removed": int(removed)}


def detect_basic_outliers(df: pd.DataFrame) -> Dict:
    # exemplo: temperatura, hr, sysbp, dias de valores plausíveis
    out = {}
    if "temperature" in df.columns:
        tmp = df["temperature"]
        out["temperature_outliers"] = int(((tmp < 30) | (tmp > 43)).sum())
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        out["hr_outliers"] = int(((hr < 20) | (hr > 250)).sum())
    return out


def impute_within_stay(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Imputação simples: forward fill then back fill within each stay_id group.
    Substitui valores extremos por NA antes do ffill/bfill se aplicável.
    """
    if "stay_id" not in df.columns:
        return df
    df_sorted = df.sort_values(["stay_id", "intime"])
    for c in cols:
        if c in df_sorted.columns:
            # convert to numeric
            df_sorted[c] = pd.to_numeric(df_sorted[c], errors="coerce")
            df_sorted[c] = df_sorted.groupby("stay_id")[c].ffill().bfill()
    return df_sorted


# ----------------------
# Main pipeline
# ----------------------
def run(args):
    p_master = Path(args.master)
    p_ed = Path(args.edstays)
    p_adm = Path(args.admissions) if args.admissions else None
    out_parquet = Path(args.out_parquet)
    out_profile = Path(args.profile_html) if args.profile_html else None
    out_report = Path(args.report_json)

    # 1) leitura segura
    master = safe_read_csv(p_master)
    edstays = safe_read_csv(p_ed)
    if p_adm and p_adm.exists():
        admissions = safe_read_csv(p_adm)
    else:
        admissions = None

    # 2) dtypes e coersões
    master = ensure_dtypes(master)
    edstays = ensure_dtypes(edstays)
    if admissions is not None:
        admissions = ensure_dtypes(admissions)

    # 3) validações básicas
    report = {}
    report["counts"] = {
        "master_rows": int(len(master)),
        "edstays_rows": int(len(edstays)),
        "admissions_rows": int(len(admissions)) if admissions is not None else None,
    }
    report["schema_validation_master"] = validate_basic_schema(master)
    report["schema_validation_edstays"] = validate_basic_schema(edstays)
    report["uniqueness_edstays"] = check_uniqueness(edstays)
    report["timestamps_master"] = timestamp_checks(master)
    report["timestamps_edstays"] = timestamp_checks(edstays)
    report["missingness_master"] = missingness_report(master)
    report["missingness_edstays"] = missingness_report(edstays)

    # 4) consistência de keys: stay_id do master deve existir em edstays (sinal de merge correto)
    if "stay_id" in master.columns and "stay_id" in edstays.columns:
        master_not_in_ed = master[~master["stay_id"].isin(edstays["stay_id"])]
        report["master_missing_in_ed_count"] = int(len(master_not_in_ed))
        if len(master_not_in_ed) > 0:
            LOG.warning("Encontradas %d stay_id no master que nao existem em edstays", len(master_not_in_ed))
    else:
        report["master_missing_in_ed_count"] = None

    # 5) hadm_id handling
    n_hadm_null = int(master["hadm_id"].isna().sum()) if "hadm_id" in master.columns else None
    report["hadm_null_master_count"] = n_hadm_null
    LOG.info("hadm_id nulos em master: %s", n_hadm_null)

    # se require_hadm: descartamos as linhas que nao tem hadm_id (caso o objetivo do job seja ficar so com internados)
    if args.require_hadm:
        before = len(master)
        master = master[master["hadm_id"].notna()].copy()
        report["rows_dropped_for_missing_hadm"] = int(before - len(master))
        LOG.info("Dropping rows without hadm_id: %d -> %d", before, len(master))

    # se drop_pediatrics: remover pediatria baseado em age (se idade existir)
    if args.drop_pediatrics:
        if "age" in master.columns:
            before = len(master)
            master = master[master["age"].notna() & (master["age"] >= args.min_age)].copy()
            removed = before - len(master)
            report["rows_dropped_pediatrics"] = int(removed)
            LOG.info("Removed pediatrics rows: %d", removed)
        else:
            report["rows_dropped_pediatrics"] = 0
            LOG.info("Coluna 'age' nao encontrada; pulando remoção pediatrica")

    # 6) outliers e imputação dentro de stay
    outlier_info = detect_basic_outliers(master)
    report["outlier_detect"] = outlier_info
    # escolha das colunas para impute (as que existirem)
    candidate_cols = [c for c in ["temperature", "heart_rate", "systolic_bp", "diastolic_bp", "respiratory_rate"] if c in master.columns]
    if candidate_cols:
        master = impute_within_stay(master, candidate_cols)

    # 7) checagens finais pós-limpeza
    report["final_counts"] = {
        "rows": int(len(master)),
        "columns": int(len(master.columns))
    }
    report["final_missingness"] = missingness_report(master, top_n=40)
    report["post_timestamps"] = timestamp_checks(master)

    # 8) validate final schema (warn if fails)
    final_schema_ok = validate_basic_schema(master)
    report["final_schema_validation"] = final_schema_ok

    # 9) salvar parquet e report
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Salvando parquet em %s", out_parquet)
    master.to_parquet(out_parquet, index=False)

    out_report.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Salvando cleaning report em %s", out_report)
    with open(out_report, "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2, default=int)

    # 10) gerar profiling (opcional)
    if out_profile and HAS_PROFILE:
        LOG.info("Gerando profile html em %s", out_profile)
        profile = ProfileReport(master, title="Master Dataset Profile", minimal=True)
        out_profile.parent.mkdir(parents=True, exist_ok=True)
        profile.to_file(out_profile)
    elif out_profile and not HAS_PROFILE:
        LOG.warning("ydata-profiling nao instalado — ignorando profile html")

    LOG.info("Ingest & clean concluido. Linhas finais: %d", len(master))
    return 0


def cli():
    p = argparse.ArgumentParser(description="Ingest and clean MIMIC-IV-ED master dataset")
    p.add_argument("--master", required=True, help="Caminho para master_dataset.csv")
    p.add_argument("--edstays", required=True, help="Caminho para edstays.csv")
    p.add_argument("--admissions", required=False, help="(Opcional) admissions.csv para recuperação/checagens")
    p.add_argument("--out-parquet", required=True, help="Caminho de saída parquet (ex: data/ed_master_clean.parquet)")
    p.add_argument("--profile-html", required=False, help="Gerar profile HTML (ydata-profiling)")
    p.add_argument("--report-json", default="reports/cleaning_report.json", help="JSON com summary do cleaning")
    p.add_argument("--drop-pediatrics", action="store_true", help="Remover pacientes < --min-age")
    p.add_argument("--min-age", type=int, default=18, help="Idade mínima (pediatrics filter)")
    p.add_argument("--require-hadm", action="store_true", help="Remover linhas sem hadm_id (manter apenas internados)")
    args = p.parse_args()
    try:
        return run(args)
    except AssertionError as e:
        LOG.exception("ASSERTION ERROR: %s", e)
        return 2
    except Exception as e:
        LOG.exception("ERRO durante ingest/clean: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
