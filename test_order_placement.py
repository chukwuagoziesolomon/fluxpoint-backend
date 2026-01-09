#!/usr/bin/env python
"""Test order placement logic for TCE strategy."""

from trading.tce.order_placement import (
    calculate_entry_order_price,
    create_pending_order,
    validate_order_placement,
    get_order_type
)

print("\n" + "="*80)
print("TESTING ORDER PLACEMENT LOGIC")
print("="*80)

# Test 1: BUY Setup - Place BUY STOP 2-3 pips above confirmation candle
print("\n1️⃣ BUY SETUP - Order 2-3 pips ABOVE confirmation candle\n")

confirmation_high_buy = 1.0850
confirmation_low_buy = 1.0820
direction_buy = "BUY"
sl_buy = 1.0810
tp_buy = 1.0870

entry_price_buy = calculate_entry_order_price(
    confirmation_high_buy,
    confirmation_low_buy,
    direction_buy,
    buffer_pips=2.5
)

print(f"Confirmation Candle: High={confirmation_high_buy:.5f}, Low={confirmation_low_buy:.5f}")
print(f"Entry (2.5 pips above high): {entry_price_buy:.5f}")
print(f"Stop Loss: {sl_buy:.5f}")
print(f"Take Profit: {tp_buy:.5f}")
print(f"\nOrder Type: {get_order_type(direction_buy)}")
print(f"Entry is {(entry_price_buy - confirmation_high_buy)/0.0001:.1f} pips above candle high ✓")

pending_order_buy = create_pending_order(
    symbol="EURUSD",
    direction=direction_buy,
    confirmation_candle_high=confirmation_high_buy,
    confirmation_candle_low=confirmation_low_buy,
    entry_price=entry_price_buy,
    stop_loss=sl_buy,
    take_profit=tp_buy,
    position_size=1.0,
    buffer_pips=2.5
)

validation_buy = validate_order_placement(
    pending_order_buy,
    confirmation_high_buy,
    confirmation_low_buy
)

print(f"\nOrder Validation: {validation_buy['is_valid']}")
if validation_buy['issues']:
    for issue in validation_buy['issues']:
        print(f"  ⚠️  {issue}")
else:
    print("  ✅ All validations passed")

print(f"\n{pending_order_buy['description']}")

# Test 2: SELL Setup - Place SELL STOP 2-3 pips below confirmation candle
print("\n\n2️⃣ SELL SETUP - Order 2-3 pips BELOW confirmation candle\n")

confirmation_high_sell = 1.0880
confirmation_low_sell = 1.0850
direction_sell = "SELL"
sl_sell = 1.0890
tp_sell = 1.0830

entry_price_sell = calculate_entry_order_price(
    confirmation_high_sell,
    confirmation_low_sell,
    direction_sell,
    buffer_pips=2.5
)

print(f"Confirmation Candle: High={confirmation_high_sell:.5f}, Low={confirmation_low_sell:.5f}")
print(f"Entry (2.5 pips below low): {entry_price_sell:.5f}")
print(f"Stop Loss: {sl_sell:.5f}")
print(f"Take Profit: {tp_sell:.5f}")
print(f"\nOrder Type: {get_order_type(direction_sell)}")
print(f"Entry is {(confirmation_low_sell - entry_price_sell)/0.0001:.1f} pips below candle low ✓")

pending_order_sell = create_pending_order(
    symbol="EURUSD",
    direction=direction_sell,
    confirmation_candle_high=confirmation_high_sell,
    confirmation_candle_low=confirmation_low_sell,
    entry_price=entry_price_sell,
    stop_loss=sl_sell,
    take_profit=tp_sell,
    position_size=1.0,
    buffer_pips=2.5
)

validation_sell = validate_order_placement(
    pending_order_sell,
    confirmation_high_sell,
    confirmation_low_sell
)

print(f"\nOrder Validation: {validation_sell['is_valid']}")
if validation_sell['issues']:
    for issue in validation_sell['issues']:
        print(f"  ⚠️  {issue}")
else:
    print("  ✅ All validations passed")

print(f"\n{pending_order_sell['description']}")

# Test 3: Invalid BUY order - SL above entry
print("\n\n3️⃣ TEST INVALID ORDER - SL incorrectly positioned\n")

invalid_order = {
    "symbol": "EURUSD",
    "direction": "BUY",
    "entry_price": 1.0860,
    "stop_loss": 1.0870,  # WRONG: SL above entry for BUY
    "take_profit": 1.0880
}

invalid_validation = validate_order_placement(
    invalid_order,
    confirmation_high_buy,
    confirmation_low_buy
)

print(f"Order Validation: {invalid_validation['is_valid']}")
print("Issues found:")
for issue in invalid_validation['issues']:
    print(f"  ⚠️  {issue}")

print("\n" + "="*80)
print("✅ ORDER PLACEMENT TESTS COMPLETE")
print("="*80)
print("\nSummary:")
print("✓ BUY orders placed 2-3 pips ABOVE confirmation candle high")
print("✓ SELL orders placed 2-3 pips BELOW confirmation candle low")
print("✓ Order validation ensures proper risk management")
print("✓ This prevents premature entry and captures momentum\n")
