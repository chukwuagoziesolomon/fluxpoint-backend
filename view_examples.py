import json

# Load training data
with open('usdjpy_h1_training_data.json', 'r') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}\n")

# Distribution by MA level
ma_counts = {}
for ex in data:
    ma = ex['ma_level']
    ma_counts[ma] = ma_counts.get(ma, 0) + 1

print("=== Distribution by MA Level ===")
for ma, count in sorted(ma_counts.items()):
    print(f"{ma}: {count} ({count/len(data)*100:.1f}%)")

# Distribution by direction
dir_counts = {}
for ex in data:
    dir = ex['direction']
    dir_counts[dir] = dir_counts.get(dir, 0) + 1

print("\n=== Distribution by Direction ===")
for dir, count in sorted(dir_counts.items()):
    print(f"{dir}: {count} ({count/len(data)*100:.1f}%)")

# Profitable vs unprofitable
profitable = sum(1 for ex in data if ex['profitable'])
print(f"\n=== Success Rate ===")
print(f"Profitable (1.5:1+): {profitable} ({profitable/len(data)*100:.1f}%)")
print(f"Very Profitable (2:1+): {sum(1 for ex in data if ex['very_profitable'])} ({sum(1 for ex in data if ex['very_profitable'])/len(data)*100:.1f}%)")

print("\n" + "="*100)
print("=== First 10 Examples ===")
print("="*100)
for i, ex in enumerate(data[:10], 1):
    print(f"\n{i}. {ex['time']} | {ex['direction']} @ {ex['ma_level']}")
    print(f"   Entry: {ex['entry_price']} | MA Values: 6={ex['ma6']:.3f}, 18={ex['ma18']:.3f}, 50={ex['ma50']:.3f}")
    print(f"   Swing: {ex['swing_duration_hours']:.1f} hours ({ex['first_bounce_time']} → {ex['time']})")
    print(f"   Outcome: Profit={ex['max_profit_pips']:.1f}p, Loss={ex['max_loss_pips']:.1f}p, Ratio={ex['profit_loss_ratio']:.2f}")
    print(f"   Labels: Profitable={ex['profitable']}, Very Profitable={ex['very_profitable']}")

print("\n" + "="*100)
print("=== Sample MA50 Examples ===")
print("="*100)
ma50_examples = [ex for ex in data if ex['ma_level'] == 'MA50'][:5]
for i, ex in enumerate(ma50_examples, 1):
    print(f"\n{i}. {ex['time']} | {ex['direction']} @ MA50")
    print(f"   Entry: {ex['entry_price']} | MA Values: 6={ex['ma6']:.3f}, 18={ex['ma18']:.3f}, 50={ex['ma50']:.3f}")
    print(f"   Swing: {ex['swing_duration_hours']:.1f} hours")
    print(f"   Outcome: Profit={ex['max_profit_pips']:.1f}p, Loss={ex['max_loss_pips']:.1f}p, Profitable={ex['profitable']}")

print("\n" + "="*100)
print("=== Sample MA18 Examples ===")
print("="*100)
ma18_examples = [ex for ex in data if ex['ma_level'] == 'MA18'][:5]
for i, ex in enumerate(ma18_examples, 1):
    print(f"\n{i}. {ex['time']} | {ex['direction']} @ MA18")
    print(f"   Entry: {ex['entry_price']} | MA Values: 6={ex['ma6']:.3f}, 18={ex['ma18']:.3f}, 50={ex['ma50']:.3f}")
    print(f"   Swing: {ex['swing_duration_hours']:.1f} hours")
    print(f"   Outcome: Profit={ex['max_profit_pips']:.1f}p, Loss={ex['max_loss_pips']:.1f}p, Profitable={ex['profitable']}")
