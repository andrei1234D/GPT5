import yfinance as yf
import pandas as pd

ticker = "EVOK"

print(f"[TEST] Downloading {ticker} 6mo daily data…")
df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False, progress=False)

if df.empty:
    print("[ERROR] Empty DataFrame — no data returned.")
else:
    print(f"[INFO] Downloaded {len(df)} rows.")
    print("[INFO] Columns:", list(df.columns))

    missing_cols = [c for c in ["Open", "High", "Low", "Close"] if c not in df.columns]
    if missing_cols:
        print(f"[WARN] Missing expected OHLC columns: {missing_cols}")
    else:
        print("[OK] All OHLC columns present!")

    # Show last few rows
    print("\n[DATA SAMPLE]")
    print(df.tail(5))

# Extra: print some quick stats
if not df.empty:
    try:
        latest = df.iloc[-1]
        print("\n[LATEST ROW]")
        print(latest)
        print(f"\nLatest Close: {latest['Close']}, Open: {latest['Open']}, Low: {latest['Low']}, High: {latest['High']}")
    except Exception as e:
        print(f"[ERROR] Could not read columns: {e}")
