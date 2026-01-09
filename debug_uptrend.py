import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fluxpoint.settings')
django.setup()

from trading.tce.types import Indicators
from trading.tce.utils import is_uptrend, is_downtrend

# Test with sample data from our test
ma6, ma18, ma50, ma200 = 80.349, 79.269, 77.725, 74.813
slope6, slope18, slope50 = +0.104, +0.149, +0.098

ind = Indicators(
    ma6=ma6, ma18=ma18, ma50=ma50, ma200=ma200,
    slope6=slope6, slope18=slope18, slope50=slope50, slope200=0,
    atr=0.001
)

print(f"MAs: {ma6:.2f} > {ma18:.2f} > {ma50:.2f} > {ma200:.2f}")
print(f"MA6 > MA18: {ma6 > ma18}")
print(f"MA18 > MA50: {ma18 > ma50}")
print(f"MA50 > MA200: {ma50 > ma200}")
print(f"All MAs aligned: {ma6 > ma18 > ma50 > ma200}")

slopes_positive = sum([slope6 > 0, slope18 > 0, slope50 > 0])
print(f"\nSlopes positive: {slopes_positive}/3")
print(f"At least 2: {slopes_positive >= 2}")

print(f"\nis_uptrend(): {is_uptrend(ind)}")
print(f"is_downtrend(): {is_downtrend(ind)}")
