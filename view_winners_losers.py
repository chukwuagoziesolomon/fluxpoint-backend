import json

# Load training data
with open('usdjpy_h1_training_data.json', 'r') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}\n")

# Separate profitable and unprofitable
profitable = [ex for ex in data if ex['profitable']]
unprofitable = [ex for ex in data if not ex['profitable']]

print("="*100)
print(f"PROFITABLE SETUPS: {len(profitable)} ({len(profitable)/len(data)*100:.1f}%)")
print("="*100)

print("\nFirst 5 WINNING trades (entry at 2nd bounce):\n")
for i, ex in enumerate(profitable[:5], 1):
    print(f"{i}. {ex['time']} | {ex['direction']} @ {ex['ma_level']}")
    print(f"   1st Bounce (TEST): {ex['first_bounce_time']} @ {ex['first_bounce_price']}")
    print(f"   2nd Bounce (ENTRY): {ex['time']} @ {ex['entry_price']}")
    print(f"   Swing Duration: {ex['swing_duration_hours']:.1f} hours between bounces")
    print(f"   OUTCOME: Profit={ex['max_profit_pips']:.1f}p, Loss={ex['max_loss_pips']:.1f}p, Ratio={ex['profit_loss_ratio']:.2f}")
    print(f"   ✅ Winner!\n")

print("\n" + "="*100)
print(f"UNPROFITABLE SETUPS: {len(unprofitable)} ({len(unprofitable)/len(data)*100:.1f}%)")
print("="*100)

print("\nFirst 5 LOSING trades (entry at 2nd bounce):\n")
for i, ex in enumerate(unprofitable[:5], 1):
    print(f"{i}. {ex['time']} | {ex['direction']} @ {ex['ma_level']}")
    print(f"   1st Bounce (TEST): {ex['first_bounce_time']} @ {ex['first_bounce_price']}")
    print(f"   2nd Bounce (ENTRY): {ex['time']} @ {ex['entry_price']}")
    print(f"   Swing Duration: {ex['swing_duration_hours']:.1f} hours between bounces")
    print(f"   OUTCOME: Profit={ex['max_profit_pips']:.1f}p, Loss={ex['max_loss_pips']:.1f}p, Ratio={ex['profit_loss_ratio']:.2f}")
    print(f"   ❌ Loser!\n")

print("\n" + "="*100)
print("SUMMARY FOR ML TRAINING")
print("="*100)
print(f"✅ Winners: {len(profitable)} examples (models learn what works)")
print(f"❌ Losers: {len(unprofitable)} examples (models learn what doesn't work)")
print(f"📊 Total: {len(data)} balanced examples for training")
print("\nKey Point: Entry is ALWAYS at the 2nd bounce (retest)")
print("           1st bounce = test/setup, 2nd bounce = entry signal")
