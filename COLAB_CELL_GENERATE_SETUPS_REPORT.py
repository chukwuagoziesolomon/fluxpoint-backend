"""
COLAB CELL - Generate All 97 TCE Setups Report

Add this to your Colab notebook AFTER Cell 4 (validation)
It will create a detailed CSV with all setup data
"""

# ============================================================================
# GENERATE DETAILED SETUPS REPORT
# ============================================================================

import pandas as pd

print("\n" + "="*120)
print("GENERATING DETAILED REPORT OF ALL VALID TCE SETUPS")
print("="*120)

# Convert to DataFrame for easy analysis
df_setups = pd.DataFrame(valid_setups_detailed)

if len(df_setups) > 0:
    # Create CSV file
    csv_filename = 'ALL_TCE_SETUPS_DETAILED.csv'
    df_setups.to_csv(csv_filename, index=False)
    
    # Save to Google Drive
    from google.colab import files
    print(f"\n✅ Generated CSV: {csv_filename}")
    print(f"   Total setups: {len(df_setups)}")
    
    # Display summary by pair
    print(f"\n📊 SETUPS BY PAIR:\n")
    pair_counts = df_setups['symbol'].value_counts().sort_values(ascending=False)
    for pair, count in pair_counts.items():
        print(f"   {pair:10s}: {count:2d} setups")
    
    # Display detailed table
    print(f"\n" + "="*120)
    print("FIRST 25 SETUPS (DETAILED):")
    print("="*120)
    
    for idx, setup in enumerate(df_setups.head(25).itertuples(), 1):
        print(f"\n[{idx}] {setup.symbol} | {setup.date}")
        print(f"    Direction:  {setup.direction.upper()}")
        print(f"    Entry:      {setup.price:.5f}")
        print(f"    Stop Loss:  {setup.stop_loss:.5f}  ({setup.sl_pips:.1f} pips)")
        print(f"    Take Profit: {setup.take_profit:.5f}  ({setup.tp_pips:.1f} pips)")
        print(f"    Risk/Reward: {setup.risk_reward:.2f}:1")
        
        # Entry reasons
        reasons = []
        if setup.trend_ok:
            if setup.direction == 'BUY':
                reasons.append(f"Uptrend: MA6({setup.ma6:.4f}) > MA18({setup.ma18:.4f}) > MA50({setup.ma50:.4f})")
            else:
                reasons.append(f"Downtrend: MA200({setup.ma200:.4f}) > MA50({setup.ma50:.4f}) > MA18({setup.ma18:.4f})")
        if setup.fib_ok:
            reasons.append(f"Fibonacci retest")
        if setup.swing_ok:
            reasons.append(f"Swing structure confirmed")
        if setup.ma_level_ok:
            reasons.append(f"Price at MA level (±5%)")
        if setup.ma_retest_ok:
            reasons.append(f"MA retest in last 20 candles")
        if setup.candlestick_ok:
            reasons.append(f"Candlestick pattern confirmed")
        
        print(f"    Reasons:   {' → '.join(reasons)}")
    
    # Create summary table
    print(f"\n" + "="*120)
    print("SUMMARY TABLE (All Setups):")
    print("="*120 + "\n")
    
    # Select key columns
    summary_cols = ['symbol', 'date', 'price', 'stop_loss', 'take_profit', 'direction', 'risk_reward']
    df_summary = df_setups[summary_cols].copy()
    df_summary.columns = ['Pair', 'Date', 'Entry', 'SL', 'TP', 'Dir', 'RR']
    df_summary['Dir'] = df_summary['Dir'].str.upper()
    df_summary['Entry'] = df_summary['Entry'].apply(lambda x: f"{x:.5f}")
    df_summary['SL'] = df_summary['SL'].apply(lambda x: f"{x:.5f}")
    df_summary['TP'] = df_summary['TP'].apply(lambda x: f"{x:.5f}")
    df_summary['RR'] = df_summary['RR'].apply(lambda x: f"{x:.2f}:1")
    
    print(df_summary.to_string(index=False))
    
    # Validation rule statistics
    print(f"\n" + "="*120)
    print("VALIDATION RULE EFFECTIVENESS:")
    print("="*120 + "\n")
    
    total = len(df_setups)
    print(f"  ✅ Trend OK:        {df_setups['trend_ok'].sum()}/{total} ({100*df_setups['trend_ok'].sum()/total:.1f}%)")
    print(f"  ✅ Fibonacci OK:    {df_setups['fib_ok'].sum()}/{total} ({100*df_setups['fib_ok'].sum()/total:.1f}%)")
    print(f"  ✅ Swing OK:        {df_setups['swing_ok'].sum()}/{total} ({100*df_setups['swing_ok'].sum()/total:.1f}%)")
    print(f"  ✅ MA Level OK:     {df_setups['ma_level_ok'].sum()}/{total} ({100*df_setups['ma_level_ok'].sum()/total:.1f}%)")
    print(f"  ✅ MA Retest OK:    {df_setups['ma_retest_ok'].sum()}/{total} ({100*df_setups['ma_retest_ok'].sum()/total:.1f}%)")
    print(f"  ✅ Candlestick OK:  {df_setups['candlestick_ok'].sum()}/{total} ({100*df_setups['candlestick_ok'].sum()/total:.1f}%)")
    print(f"  ✅ Multi-TF OK:     {df_setups['multi_tf_ok'].sum()}/{total} ({100*df_setups['multi_tf_ok'].sum()/total:.1f}%)")
    print(f"  ✅ Correlation OK:  {df_setups['correlation_ok'].sum()}/{total} ({100*df_setups['correlation_ok'].sum()/total:.1f}%)")
    print(f"  ✅ Risk Mgmt OK:    {df_setups['risk_management_ok'].sum()}/{total} ({100*df_setups['risk_management_ok'].sum()/total:.1f}%)")
    
    # Direction split
    print(f"\n" + "="*120)
    print("BUY vs SELL SPLIT:")
    print("="*120 + "\n")
    
    buy_count = (df_setups['direction'] == 'BUY').sum()
    sell_count = (df_setups['direction'] == 'SELL').sum()
    print(f"  🟢 BUY:  {buy_count} setups ({100*buy_count/total:.1f}%)")
    print(f"  🔴 SELL: {sell_count} setups ({100*sell_count/total:.1f}%)")
    
    # Risk/Reward analysis
    print(f"\n" + "="*120)
    print("RISK/REWARD RATIO ANALYSIS:")
    print("="*120 + "\n")
    
    print(f"  Average R/R:  {df_setups['risk_reward'].mean():.2f}:1")
    print(f"  Min R/R:      {df_setups['risk_reward'].min():.2f}:1")
    print(f"  Max R/R:      {df_setups['risk_reward'].max():.2f}:1")
    print(f"  Median R/R:   {df_setups['risk_reward'].median():.2f}:1")
    
    print(f"\n" + "="*120)
    print(f"✅ REPORT COMPLETE - All data saved to: {csv_filename}")
    print("="*120 + "\n")
    
    # Download CSV
    files.download(csv_filename)

else:
    print("\n❌ No setups found!")
