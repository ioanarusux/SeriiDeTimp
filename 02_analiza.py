from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

FOLDER = Path(__file__).resolve().parent
FIG = FOLDER / "figuri"
FIG.mkdir(exist_ok=True)
REZ = FOLDER / "rezultate_analiza.txt"

# setari prognoza
DATA_END_TRAIN = "2022-12-01"
ORizont = 12  # luni de prognoza dupa setul de antrenare


def salveaza(text):
    with open(REZ, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def adf_test(serie, nume):
    r = adfuller(serie.dropna(), autolag="AIC")
    linie = f"ADF {nume}: stat={r[0]:.4f}, p={r[1]:.4f} -> {'stationar' if r[1] < 0.05 else 'nestationar'}"
    salveaza(linie)
    return r[1] < 0.05


def metrici(y_real, y_pred):
    y_real = np.asarray(y_real, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_real - y_pred
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err / y_real)) * 100
    return rmse, mape


def alege_arima(train):
    """Alege un ARIMA/SARIMA simplu dupa AIC (grid mic)."""
    best_aic, best_order, best_seasonal, best_model = np.inf, None, None, None
    candidati = [
        ((1, 1, 1), (0, 0, 0, 0)),
        ((1, 1, 1), (1, 0, 1, 12)),
        ((0, 1, 1), (0, 1, 1, 12)),
        ((2, 1, 1), (1, 0, 1, 12)),
        ((1, 1, 2), (0, 1, 1, 12)),
    ]
    for order, seasonal in candidati:
        try:
            mod = ARIMA(
                train,
                order=order,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = mod.fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_order, best_seasonal, best_model = order, seasonal, fit
        except Exception:
            continue
    salveaza(f"ARIMA ales: order={best_order}, seasonal={best_seasonal}, AIC={best_aic:.2f}")
    return best_model


def prognoza_hw(train, pasi):
    if len(train) < 24:
        seasonal = None
        period = None
    else:
        seasonal = "add"
        period = 12
    mod = ExponentialSmoothing(
        train, trend="add", seasonal=seasonal, seasonal_periods=period
    )
    fit = mod.fit(optimized=True)
    return fit.forecast(pasi)


def main():
    if REZ.exists():
        REZ.unlink()

    df = pd.read_excel(FOLDER / "date.xlsx")
    df["data"] = pd.to_datetime(df["data"])
    df = df.set_index("data").sort_index()

    salveaza("=== DATE ===")
    salveaza(f"Perioada: {df.index.min().date()} - {df.index.max().date()}, N={len(df)}")

    # --- grafic serii ---
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(df.index, df["EUR_RON"], color="steelblue")
    ax[0].set_title("Curs mediu lunar EUR/RON (BNR)")
    ax[1].plot(df.index, df["inflatie"], color="darkorange")
    ax[1].set_title("Inflatie anuala (%) - CPI Romania")
    ax[2].plot(df.index, df["ROBOR3M"], color="seagreen")
    ax[2].set_title("ROBOR 3 luni (%) - Eurostat")
    ax[2].set_xlabel("Data")
    plt.tight_layout()
    plt.savefig(FIG / "01_serii_originale.png", dpi=150)
    plt.close()

    # ========== PARTEA UNIVARIATA: EUR_RON ==========
    salveaza("\n=== ANALIZA UNIVARIATA: EUR_RON ===")
    y = df["EUR_RON"].copy()

    # trend determinist (regresie liniara pe timp)
    t = np.arange(len(y))
    coef = np.polyfit(t, y.values, 1)
    trend_det = np.polyval(coef, t)
    salveaza(f"Trend determinist (panta liniara): {coef[0]:.6f} lei/luna")

    adf_test(y, "EUR_RON nivel")
    dy = y.diff().dropna()
    adf_test(dy, "EUR_RON prima diferenta")

    # descompunere (trend stochastic aproximat)
    dec = seasonal_decompose(y, model="additive", period=12)
    dec.plot()
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig(FIG / "02_descompunere_eur_ron.png", dpi=150)
    plt.close()

    # netezire exponentiala (Holt) pe tot setul pentru vizualizare
    hw_fit_line = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()
    plt.figure(figsize=(10, 4))
    plt.plot(y.index, y.values, label="Observat")
    plt.plot(y.index, hw_fit_line.fittedvalues, label="Holt-Winters (fit)")
    plt.legend()
    plt.title("Netezire exponentiala Holt-Winters")
    plt.tight_layout()
    plt.savefig(FIG / "03_holt_winters_fit.png", dpi=150)
    plt.close()

    # impartire train / test
    train = y.loc[:DATA_END_TRAIN]
    test = y.loc[DATA_END_TRAIN:].iloc[1 : ORizont + 1]
    salveaza(f"Train pana la {train.index.max().date()}, test {len(test)} luni")

    # SARIMA
    model_arima = alege_arima(train)
    fc_arima = model_arima.get_forecast(steps=len(test))
    pred_arima = fc_arima.predicted_mean
    ci_arima = fc_arima.conf_int()

    # Holt-Winters out-of-sample
    pred_hw = prognoza_hw(train, len(test))

    rmse_a, mape_a = metrici(test.values, pred_arima.values)
    rmse_h, mape_h = metrici(test.values, pred_hw.values)
    salveaza(f"Test SARIMA: RMSE={rmse_a:.4f}, MAPE={mape_a:.2f}%")
    salveaza(f"Test Holt-Winters: RMSE={rmse_h:.4f}, MAPE={mape_h:.2f}%")
    if mape_a < mape_h:
        salveaza("Pe setul de test, SARIMA este mai precis decat Holt-Winters (MAPE mai mic).")
    else:
        salveaza("Pe setul de test, Holt-Winters este mai precis decat SARIMA (MAPE mai mic).")

    # grafic prognoza test
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, "o-", label="Test (real)")
    plt.plot(test.index, pred_arima.values, "s--", label="SARIMA")
    plt.plot(test.index, pred_hw, "^--", label="Holt-Winters")
    plt.fill_between(
        test.index, ci_arima.iloc[:, 0], ci_arima.iloc[:, 1], alpha=0.2, label="IC 95% SARIMA"
    )
    plt.legend()
    plt.title("Prognoza out-of-sample: SARIMA vs Holt-Winters")
    plt.tight_layout()
    plt.savefig(FIG / "04_prognoza_test.png", dpi=150)
    plt.close()

    # prognoza pe orizont (dupa tot esantionul pana in 2024)
    full_train = y
    model_f = alege_arima(full_train)
    fc_f = model_f.get_forecast(steps=ORizont)
    pred_f = fc_f.predicted_mean
    ci_f = fc_f.conf_int()
    idx_f = pd.date_range(full_train.index[-1] + pd.offsets.MonthBegin(), periods=ORizont, freq="MS")

    plt.figure(figsize=(10, 5))
    plt.plot(full_train.index[-60:], full_train.values[-60:], label="Ultimii 5 ani")
    plt.plot(idx_f, pred_f, "r--", label=f"Prognoza {ORizont} luni")
    plt.fill_between(idx_f, ci_f.iloc[:, 0], ci_f.iloc[:, 1], color="red", alpha=0.15)
    plt.legend()
    plt.title("Prognoza punctuala si interval de incredere (SARIMA)")
    plt.tight_layout()
    plt.savefig(FIG / "05_prognoza_orizont.png", dpi=150)
    plt.close()

    # ========== PARTEA MULTIVARIATA ==========
    salveaza("\n=== ANALIZA MULTIVARIATA ===")
    multi = df[["EUR_RON", "inflatie", "ROBOR3M"]].dropna()

    for col in multi.columns:
        adf_test(multi[col], col)

    # Johansen cointegration
    joh = coint_johansen(multi, det_order=0, k_ar_diff=1)
    salveaza("Test Johansen (urmarirea rangului):")
    for i, tr in enumerate(joh.lr1):
        salveaza(f"  r<={i}: trace={tr:.3f}, crit 5%={joh.cvt[i, 1]:.3f}")

    rank = 0
    for i in range(len(multi.columns)):
        if joh.lr1[i] > joh.cvt[i, 1]:
            rank = len(multi.columns) - i
    rank = max(0, min(rank, len(multi.columns) - 1))
    salveaza(f"Rang cointegrare estimat: {rank}")

    # VAR pe primele diferente daca e nevoie; aici folosim niveluri daca rank>=1
    if rank >= 1:
        salveaza("Model: VECM (existenta relatie de cointegrare)")
        vecm = VECM(multi, k_ar_diff=2, coint_rank=rank, deterministic="ci")
        vecm_fit = vecm.fit()
        salveaza(str(vecm_fit.summary())[:1200] + "...")
        var_model = vecm_fit
        for_causality = vecm_fit
    else:
        salveaza("Model: VAR in niveluri (fara cointegrare clara)")
        model = VAR(multi)
        lag = model.select_order(12).aic
        salveaza(f"Lag VAR ales (AIC): {lag}")
        var_model = model.fit(max(1, lag))
        salveaza(str(var_model.summary())[:1200] + "...")
        for_causality = var_model

    # Granger (pe VAR simplu pentru claritate)
    var_g = VAR(multi).fit(2)
    salveaza("\nCauzalitate Granger (lag=2), p-values:")
    perechi = [
        ("inflatie", "EUR_RON"),
        ("ROBOR3M", "EUR_RON"),
        ("EUR_RON", "inflatie"),
        ("ROBOR3M", "inflatie"),
    ]
    for cauza, efect in perechi:
        gc = grangercausalitytests(
            multi[[efect, cauza]], maxlag=2, verbose=False
        )
        p = gc[2][0]["ssr_ftest"][1]
        salveaza(f"  {cauza} -> {efect}: p={p:.4f}")

    # IRF si FEVD din VAR
    irf = var_g.irf(10)
    irf.plot(orth=True)
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig(FIG / "06_impulse_response.png", dpi=150)
    plt.close()

    fevd = var_g.fevd(10)
    fevd.plot()
    plt.gcf().set_size_inches(10, 8)
    plt.tight_layout()
    plt.savefig(FIG / "07_descompunere_varianta.png", dpi=150)
    plt.close()

    salveaza("\nGrafice salvate in folderul 'figuri/'.")
    salveaza("Analiza finalizata.")


if __name__ == "__main__":
    main()
