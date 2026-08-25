import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

FOLDER = Path(__file__).resolve().parent
OUT_XLSX = FOLDER / "date.xlsx"
OUT_CSV = FOLDER / "date.csv"


def curs_eur_ron_lunar():
    raw = urllib.request.urlopen(
        "http://infovalutar.ro/bnr/export/csv/eur", timeout=90
    ).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        content = zf.read(name)

    df = pd.read_csv(io.BytesIO(content), encoding="utf-8", skiprows=1)
    df.columns = ["data", "EUR_RON"]
    df["data"] = pd.to_datetime(df["data"])
    df["EUR_RON"] = pd.to_numeric(df["EUR_RON"], errors="coerce")
    df = df.dropna().sort_values("data")
    lunar = (
        df.set_index("data")["EUR_RON"]
        .resample("MS")
        .mean()
        .reset_index()
    )
    return lunar


def inflatie_lunara():
    """Indice CPI Romania (FRED) -> inflatie anuala %."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CP0000ROM086NEST"
    df = pd.read_csv(url)
    df.columns = ["data", "CPI"]
    df["data"] = pd.to_datetime(df["data"])
    df["CPI"] = pd.to_numeric(df["CPI"], errors="coerce")
    df["inflatie"] = df["CPI"].pct_change(12) * 100
    return df[["data", "inflatie"]].dropna()


def robor_3m_lunar():
    """Rata interbancara 3 luni Romania (Eurostat)."""
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/"
        "data/irt_st_m?format=JSON&lang=en&geo=RO&int_rt=IRT_M3&lastTimePeriod=400"
    )
    data = json.loads(urllib.request.urlopen(url, timeout=90).read().decode())
    time_idx = data["dimension"]["time"]["category"]["index"]
    inv = {v: k for k, v in time_idx.items()}

    rows = []
    for i, val in data["value"].items():
        if val is None:
            continue
        tcode = inv[int(i)]
        rows.append(
            {
                "data": pd.Period(tcode, freq="M").to_timestamp(),
                "ROBOR3M": float(val),
            }
        )
    return pd.DataFrame(rows).sort_values("data")


def main():
    print("se descarca...")
    eur = curs_eur_ron_lunar()
    inf = inflatie_lunara()
    rob = robor_3m_lunar()

    df = eur.merge(inf, on="data", how="inner").merge(rob, on="data", how="inner")
    df = df[(df["data"] >= "2005-01-01") & (df["data"] <= "2024-12-01")]
    df = df.sort_values("data").reset_index(drop=True)

    df.to_excel(OUT_XLSX, index=False)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Salvat: {OUT_XLSX} ({len(df)} luni, {df['data'].min().date()} - {df['data'].max().date()})")
    print(df.tail(3))


if __name__ == "__main__":
    main()
