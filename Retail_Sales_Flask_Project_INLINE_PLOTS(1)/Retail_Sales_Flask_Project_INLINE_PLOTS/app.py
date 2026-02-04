import io
import base64
import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # avoid GUI backend errors
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

app = Flask(__name__)
app.secret_key = "super-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def safe_read_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Sales" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Sales' columns.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"])

    df = df.sort_values("Date").set_index("Date")

    # Fill missing days to keep time series continuous
    df = df.asfreq("D")
    df["Sales"] = df["Sales"].interpolate(method="time").bfill().ffill()

    return df


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to base64 string (NOT saving as .png file)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def adf_test(series: pd.Series):
    res = adfuller(series.dropna())
    return {
        "adf_stat": float(res[0]),
        "p_value": float(res[1]),
        "lags": int(res[2]),
        "nobs": int(res[3]),
        "critical": {k: float(v) for k, v in res[4].items()}
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        period_str = request.form.get("period", "30").strip()
        try:
            period = int(period_str)
            if period < 2 or period > 365:
                raise ValueError
        except Exception:
            flash("Period must be an integer between 2 and 365 (ex: 7, 30).", "error")
            return redirect(url_for("index"))

        file = request.files.get("file")
        use_sample = request.form.get("use_sample") == "yes"

        try:
            if use_sample:
                filepath = os.path.join(BASE_DIR, "retail_sales.csv")
                if not os.path.exists(filepath):
                    raise FileNotFoundError("Sample file retail_sales.csv not found in project folder.")
            else:
                if not file or file.filename.strip() == "":
                    raise ValueError("Please upload a CSV file.")
                if not file.filename.lower().endswith(".csv"):
                    raise ValueError("Only .csv files are allowed.")

                unique_name = f"upload_{uuid.uuid4().hex}.csv"
                filepath = os.path.join(UPLOAD_DIR, unique_name)
                file.save(filepath)

            # Pass filename + flag through querystring
            return redirect(url_for("report", f=os.path.basename(filepath), sample=("yes" if use_sample else "no"), period=period))

        except Exception as e:
            flash(str(e), "error")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/report")
def report():
    sample = request.args.get("sample", "no")
    period = int(request.args.get("period", "30"))
    fname = request.args.get("f", "")

    if sample == "yes":
        filepath = os.path.join(BASE_DIR, "retail_sales.csv")
    else:
        filepath = os.path.join(UPLOAD_DIR, fname)

    if not os.path.exists(filepath):
        flash("File not found. Please upload again.", "error")
        return redirect(url_for("index"))

    try:
        df = safe_read_csv(filepath)
        s = df["Sales"]

        summary = {
            "rows": int(df.shape[0]),
            "start": df.index.min().date().isoformat(),
            "end": df.index.max().date().isoformat(),
            "mean": float(s.mean()),
            "min": float(s.min()),
            "max": float(s.max()),
            "std": float(s.std())
        }

        # 1) Trend plot
        fig1 = plt.figure(figsize=(11, 4))
        plt.plot(df.index, s)
        plt.title("Daily Retail Sales")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        trend_b64 = fig_to_base64(fig1)

        # 2) Decomposition
        dec = seasonal_decompose(s, model="additive", period=period)
        fig2 = dec.plot()
        fig2.set_size_inches(11, 7)
        decomp_b64 = fig_to_base64(fig2)

        # 3) ACF
        fig3 = plt.figure(figsize=(11, 4))
        plot_acf(s, lags=min(60, len(s)//2), ax=plt.gca())
        plt.title("ACF (Autocorrelation)")
        acf_b64 = fig_to_base64(fig3)

        # 4) PACF
        fig4 = plt.figure(figsize=(11, 4))
        plot_pacf(s, lags=min(60, max(5, len(s)//2 - 1)), ax=plt.gca(), method="ywm")
        plt.title("PACF (Partial Autocorrelation)")
        pacf_b64 = fig_to_base64(fig4)

        # 5) ADF tests
        adf = adf_test(s)
        stationary = adf["p_value"] < 0.05

        s_diff = s.diff().dropna()
        fig5 = plt.figure(figsize=(11, 4))
        plt.plot(s_diff.index, s_diff)
        plt.title("Differenced Sales (1st Difference)")
        plt.xlabel("Date")
        plt.ylabel("Δ Sales")
        diff_b64 = fig_to_base64(fig5)

        adf_diff = adf_test(s_diff)

        insights = [
            "Trend plot shows long-term direction (up/down).",
            "Decomposition splits Trend, Seasonality, and Residual (noise).",
            "ACF/PACF reveal lag correlations and seasonal repeats.",
            "ADF test checks stationarity (p < 0.05 ⇒ stationary).",
            "Differencing often improves stationarity for forecasting."
        ]

        images = {
            "trend": trend_b64,
            "decomp": decomp_b64,
            "acf": acf_b64,
            "pacf": pacf_b64,
            "diff": diff_b64
        }

        return render_template(
            "report.html",
            summary=summary,
            period=period,
            stationary=stationary,
            adf=adf,
            adf_diff=adf_diff,
            images=images,
            insights=insights
        )

    except Exception as e:
        flash(f"Error while processing file: {e}", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
