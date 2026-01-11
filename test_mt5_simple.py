"""
Simple MT5 test to see what data structure we're actually getting
"""
import MetaTrader5 as mt5
import pandas as pd

# Initialize MT5
if not mt5.initialize():
    print("❌ MT5 initialization failed")
    quit()

print("✅ Connected to MT5\n")

# Try to get just 10 bars of EURUSD Daily data
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_D1
bars = 10

print(f"📥 Fetching {bars} bars of {symbol} {timeframe}...\n")

rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)

if rates is None:
    print("❌ No data returned")
    print(f"Error: {mt5.last_error()}")
else:
    print(f"✅ Got {len(rates)} bars\n")
    
    # Show what type it is
    print(f"Type: {type(rates)}")
    print(f"Shape: {rates.shape if hasattr(rates, 'shape') else 'N/A'}")
    print(f"Dtype: {rates.dtype if hasattr(rates, 'dtype') else 'N/A'}\n")
    
    # Show field names if it's a structured array
    if hasattr(rates, 'dtype') and hasattr(rates.dtype, 'names'):
        print(f"Field names: {rates.dtype.names}\n")
    
    # Show first 3 records
    print("First 3 records:")
    print(rates[:3])
    print()
    
    # Try converting to DataFrame
    print("Attempting DataFrame conversion...")
    try:
        df = pd.DataFrame(rates)
        print(f"✅ DataFrame created")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}\n")
        print("First 3 rows:")
        print(df.head(3))
    except Exception as e:
        print(f"❌ DataFrame conversion failed: {e}")

# Cleanup
mt5.shutdown()
print("\n✅ MT5 connection closed")
