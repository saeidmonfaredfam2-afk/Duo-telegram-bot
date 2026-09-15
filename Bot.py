import os
import logging
import requests
import pandas as pd
import numpy as np

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# =========================
# SETTINGS
# =========================

BOT_TOKEN = os.getenv("BOT_TOKEN")

SYMBOL = "SUIUSDT"
BYBIT_URL = "https://api.bybit.com"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# =========================
# BYBIT DATA
# =========================

def get_klines(interval, limit=200):
    url = f"{BYBIT_URL}/v5/market/kline"

    params = {
        "category": "spot",
        "symbol": SYMBOL,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if data.get("retCode") != 0:
        raise Exception(data.get("retMsg", "Bybit error"))

    rows = data["result"]["list"]

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover"
        ]
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    df["time"] = pd.to_datetime(
        pd.to_numeric(df["time"]),
        unit="ms"
    )

    df = df.sort_values("time").reset_index(drop=True)

    # حذف کندل باز فعلی
    if len(df) > 2:
        df = df.iloc[:-1].copy()

    return df


def get_current_price():
    url = f"{BYBIT_URL}/v5/market/tickers"

    params = {
        "category": "spot",
        "symbol": SYMBOL
    }

    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    if data.get("retCode") != 0:
        raise Exception(data.get("retMsg", "Price error"))

    return float(data["result"]["list"][0]["lastPrice"])


# =========================
# INDICATORS
# =========================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)

    macd_line = fast - slow
    signal = ema(macd_line, 9)
    histogram = macd_line - signal

    return macd_line, signal, histogram


def bollinger(series, period=20, std_mult=2):
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()

    upper = middle + std_mult * std
    lower = middle - std_mult * std

    return upper, middle, lower


def atr(df, period=14):
    high_low = df["high"] - df["low"]

    high_close = (
        df["high"] - df["close"].shift()
    ).abs()

    low_close = (
        df["low"] - df["close"].shift()
    ).abs()

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


# =========================
# PARABOLIC SAR
# =========================

def parabolic_sar(df, step=0.02, max_af=0.2):
    high = df["high"].values
    low = df["low"].values

    sar = np.zeros(len(df))

    bullish = True
    af = step
    ep = high[0]

    sar[0] = low[0]

    for i in range(1, len(df)):

        previous_sar = sar[i - 1]

        if bullish:
            sar[i] = previous_sar + af * (ep - previous_sar)

            sar[i] = min(
                sar[i],
                low[i - 1]
            )

            if i >= 2:
                sar[i] = min(
                    sar[i],
                    low[i - 2]
                )

            if low[i] < sar[i]:
                bullish = False
                sar[i] = ep
                ep = low[i]
                af = step

            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_af)

        else:
            sar[i] = previous_sar + af * (ep - previous_sar)

            sar[i] = max(
                sar[i],
                high[i - 1]
            )

            if i >= 2:
                sar[i] = max(
                    sar[i],
                    high[i - 2]
                )

            if high[i] > sar[i]:
                bullish = True
                sar[i] = ep
                ep = high[i]
                af = step

            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_af)

    return pd.Series(sar, index=df.index)


# =========================
# SUPPORT / RESISTANCE
# =========================

def support_resistance(df):
    recent = df.tail(50)

    support = recent["low"].rolling(5, center=True).min()
    resistance = recent["high"].rolling(5, center=True).max()

    supports = support.dropna().tail(10)
    resistances = resistance.dropna().tail(10)

    support_level = supports.min()
    resistance_level = resistances.max()

    return float(support_level), float(resistance_level)


# =========================
# DIVERGENCE
# =========================

def detect_divergence(df):
    if len(df) < 40:
        return "نامشخص"

    price = df["close"]
    r = df["rsi"]

    p1 = price.iloc[-20]
    p2 = price.iloc[-1]

    r1 = r.iloc[-20]
    r2 = r.iloc[-1]

    # Bullish divergence
    if p2 < p1 and r2 > r1:
        return "🟢 واگرایی مثبت RSI"

    # Bearish divergence
    if p2 > p1 and r2 < r1:
        return "🔴 واگرایی منفی RSI"

    return "⚪ واگرایی واضح دیده نشد"


# =========================
# ANALYSIS
# =========================

def analyze(df):
    df = df.copy()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)

    df["rsi"] = rsi(df["close"])

    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"]
    ) = macd(df["close"])

    (
        df["bb_upper"],
        df["bb_middle"],
        df["bb_lower"]
    ) = bollinger(df["close"])

    df["sar"] = parabolic_sar(df)

    df["atr"] = atr(df)

    current = df.iloc[-1]
    previous = df.iloc[-2]

    score = 0
    reasons = []

    # EMA
    if current["ema20"] > current["ema50"]:
        score += 2
        reasons.append("EMA20 بالای EMA50")
    else:
        score -= 2
        reasons.append("EMA20 زیر EMA50")

    # MACD
    if current["macd"] > current["macd_signal"]:
        score += 1
        reasons.append("MACD مثبت")
    else:
        score -= 1
        reasons.append("MACD منفی")

    # RSI
    if 50 <= current["rsi"] <= 70:
        score += 1
        reasons.append("RSI صعودی")
    elif current["rsi"] < 30:
        score += 1
        reasons.append("RSI اشباع فروش")
    elif current["rsi"] > 70:
        score -= 1
        reasons.append("RSI اشباع خرید")
    else:
        reasons.append("RSI خنثی")

    # Bollinger
    if current["close"] > current["bb_middle"]:
        score += 1
        reasons.append("قیمت بالای میانگین Bollinger")
    else:
        score -= 1
        reasons.append("قیمت زیر میانگین Bollinger")

    # SAR
    if current["close"] > current["sar"]:
        score += 1
        reasons.append("Parabolic SAR صعودی")
    else:
        score -= 1
        reasons.append("Parabolic SAR نزولی")

    # Volume
    avg_volume = df["volume"].tail(20).mean()

    if current["volume"] > avg_volume:
        reasons.append("حجم بالاتر از میانگین")
        score += 1
    else:
        reasons.append("حجم معمولی")

    # Trend
    if current["close"] > current["ema50"]:
        trend = "🟢 صعودی"
    else:
        trend = "🔴 نزولی"

    # Signal
    if score >= 5:
        signal = "🟢 خرید / LONG"
    elif score <= -4:
        signal = "🔴 فروش / SHORT"
    else:
        signal = "🟡 خنثی / WAIT"

    support, resistance = support_resistance(df)

    divergence = detect_divergence(df)

    return {
        "price": current["close"],
        "ema20": current["ema20"],
        "ema50": current["ema50"],
        "rsi": current["rsi"],
        "macd": current["macd"],
        "macd_signal": current["macd_signal"],
        "bb_upper": current["bb_upper"],
        "bb_middle": current["bb_middle"],
        "bb_lower": current["bb_lower"],
        "sar": current["sar"],
        "volume": current["volume"],
        "avg_volume": avg_volume,
        "support": support,
        "resistance": resistance,
        "divergence": divergence,
        "score": score,
        "signal": signal,
        "trend": trend,
        "reasons": reasons
    }


# =========================
# ENTRY / SL / TP
# =========================

def trade_levels(result):

    price = result["price"]
    support = result["support"]
    resistance = result["resistance"]

    # Long setup
    if result["score"] >= 5:
        entry = price
        stop = support

        if stop >= entry:
            stop = entry * 0.97

        risk = entry - stop

        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 2.5
        direction = "LONG"

    # Short setup
    elif result["score"] <= -4:
        entry = price
        stop = resistance

        if stop <= entry:
            stop = entry * 1.03

        risk = stop - entry

        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 2.5
        direction = "SHORT"

    else:
        return None

    return {
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2
    }


# =========================
# FORMAT TIMEFRAME
# =========================

def timeframe_name(tf):
    names = {
        "D": "روزانه",
        "W": "هفتگی",
        "M": "ماهانه"
    }

    return names.get(tf, tf)


# =========================
# FULL REPORT
# =========================

def create_report():
    current_price = get_current_price()

    timeframes = {
        "D": 200,
        "W": 200,
        "M": 100
    }

    results = {}

    for tf, limit in timeframes.items():
        df = get_klines(tf, limit)
        results[tf] = analyze(df)

    total_score = sum(
        results[tf]["score"]
        for tf in results
    )

    average_score = total_score / len(results)

    if average_score >= 4:
        overall = "🟢 تمایل کلی: صعودی"
    elif average_score <= -3:
        overall = "🔴 تمایل کلی: نزولی"
    else:
        overall = "🟡 تمایل کلی: خنثی"

    text = []

    text.append("🚀 *SUI/USDT Technical Analysis*")
    text.append("")
    text.append(f"💰 قیمت فعلی: `{current_price:.4f}`")
    text.append(f"{overall}")
    text.append(
        f"📊 امتیاز کلی: `{total_score:+d}`"
    )
    text.append("")

    for tf in ["M", "W", "D"]:

        r = results[tf]

        text.append(
            f"━━━ *{timeframe_name(tf)}* ━━━"
        )

        text.append(
            f"📈 روند: {r['trend']}"
        )

        text.append(
            f"🎯 سیگنال: {r['signal']}"
        )

        text.append(
            f"⭐ امتیاز: `{r['score']:+d}`"
        )

        text.append(
            f"EMA20: `{r['ema20']:.4f}`"
        )

        text.append(
            f"EMA50: `{r['ema50']:.4f}`"
        )

        text.append(
            f"RSI: `{r['rsi']:.1f}`"
        )

        text.append(
            f"MACD: `{r['macd']:.5f}`"
        )

        text.append(
            f"BB بالا: `{r['bb_upper']:.4f}`"
        )

        text.append(
            f"BB پایین: `{r['bb_lower']:.4f}`"
        )

        text.append(
            f"SAR: `{r['sar']:.4f}`"
        )

        text.append(
            f"حمایت: `{r['support']:.4f}`"
        )

        text.append(
            f"مقاومت: `{r['resistance']:.4f}`"
        )

        text.append(
            f"واگرایی: {r['divergence']}"
        )

        text.append(
            f"📦 حجم: `{r['volume']:.0f}`"
        )

        text.append("")

        trade = trade_levels(r)

        if trade:
            text.append("🎯 *سطوح تکنیکال*")

            text.append(
                f"Entry: `{trade['entry']:.4f}`"
            )

            text.append(
                f"SL: `{trade['stop']:.4f}`"
            )

            text.append(
                f"TP1: `{trade['tp1']:.4f}`"
            )

            text.append(
                f"TP2: `{trade['tp2']:.4f}`"
            )

            text.append("")

    text.append("🧠 *جمع‌بندی عوامل:*")

    for tf in ["D", "W", "M"]:
        text.append(
            f"\n{timeframe_name(tf)}:"
        )

        for reason in results[tf]["reasons"][:6]:
            text.append(f"• {reason}")

    text.append("")
    text.append(
        "⚠️ این تحلیل صرفاً تکنیکال و الگوریتمی است و "
        "سیگنال قطعی یا تضمین سود نیست."
    )

    return "\n".join(text)


# =========================
# TELEGRAM
# =========================

def keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🔄 بروزرسانی تحلیل",
                callback_data="refresh"
            )
        ]
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    message = (
        "👋 سلام!\n\n"
        "🤖 من ربات تحلیل تکنیکال SUI/USDT هستم.\n\n"
        "برای دریافت تحلیل کامل:\n"
        "📊 /sui\n\n"
        "تایم‌فریم‌ها:\n"
        "• ماهانه\n"
        "• هفتگی\n"
        "• روزانه\n\n"
        "اندیکاتورها:\n"
        "EMA • MACD • RSI • Bollinger • SAR\n"
        "Volume • Support/Resistance • Divergence"
    )

    await update.message.reply_text(
        message,
        parse_mode="Markdown",
        reply_markup=keyboard()
    )


async def sui(update: Update, context: ContextTypes.DEFAULT_TYPE):

    msg = await update.message.reply_text(
        "⏳ در حال دریافت اطلاعات SUI از Bybit..."
    )

    try:
        report = create_report()

        await msg.edit_text(
            report,
            parse_mode="Markdown",
            reply_markup=keyboard()
        )

    except Exception as e:

        logging.exception("Analysis error")

        await msg.edit_text(
            "❌ خطا در دریافت اطلاعات.\n"
            "لطفاً چند لحظه بعد دوباره امتحان کن."
        )


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):

    query = update.callback_query

    await query.answer()

    if query.data == "refresh":

        await query.edit_message_text(
            "⏳ در حال بروزرسانی تحلیل SUI..."
        )

        try:
            report = create_report()

            await query.edit_message_text(
                report,
                parse_mode="Markdown",
                reply_markup=keyboard()
            )

        except Exception:

            logging.exception("Refresh error")

            await query.edit_message_text(
                "❌ خطا در بروزرسانی.\n"
                "لطفاً دوباره تلاش کن.",
                reply_markup=keyboard()
            )


# =========================
# MAIN
# =========================

def main():

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN environment variable is missing."
        )

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start)
    )

    application.add_handler(
        CommandHandler("sui", sui)
    )

    application.add_handler(
        CallbackQueryHandler(button)
    )

    print("SUI Telegram Bot is running...")

    application.run_polling()


if __name__ == "__main__":
    main()
