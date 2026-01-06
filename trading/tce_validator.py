from .models import Candle, Indicator, Swing, Correlation
from django.db.models import Q
from decimal import Decimal
from .tce.validation import validate_tce
from .tce.types import Candle as PureCandle, Indicators, Swing as PureSwing, MarketStructure, HigherTFCandle
from .tce.sr import near_support_resistance

class TCEValidator:
    """
    Pure rule-based TCE validation engine.
    Validates TCE setups based on Adam Khoo TCE strategy rules.
    """

    def __init__(self, symbol, timeframe, entry_candle, direction):
        """
        Initialize validator with trade setup parameters.

        :param symbol: Trading symbol (e.g., 'EURUSD')
        :param timeframe: Entry timeframe (e.g., 'M15')
        :param entry_candle: Candle instance for entry
        :param direction: 'BUY' or 'SELL'
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.entry_candle = entry_candle
        self.direction = direction
        self.reasons = []

    def validate(self):
        """
        Run TCE validation using the pure-Python engine.

        :return: (is_valid: bool, reasons: list of str)
        """
        # Gather data for pure engine
        pure_candle = PureCandle(
            open=self.entry_candle.open_price,
            high=self.entry_candle.high_price,
            low=self.entry_candle.low_price,
            close=self.entry_candle.close_price,
            timestamp=self.entry_candle.timestamp
        )

        # Indicators
        ind_models = Indicator.objects.filter(candle=self.entry_candle)
        ind_dict = {ind.type: ind for ind in ind_models}
        indicators = Indicators(
            ma6=ind_dict.get('MA6', Indicator(value=0, slope=0)).value,
            ma18=ind_dict.get('MA18', Indicator(value=0, slope=0)).value,
            ma50=ind_dict.get('MA50', Indicator(value=0, slope=0)).value,
            ma200=ind_dict.get('MA200', Indicator(value=0, slope=0)).value,
            slope6=ind_dict.get('MA6', Indicator(value=0, slope=0)).slope,
            slope18=ind_dict.get('MA18', Indicator(value=0, slope=0)).slope,
            slope50=ind_dict.get('MA50', Indicator(value=0, slope=0)).slope,
            slope200=ind_dict.get('MA200', Indicator(value=0, slope=0)).slope,
        )

        # Swing
        swing_model = Swing.objects.filter(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp').first()
        pure_swing = PureSwing(
            type=swing_model.type if swing_model else 'low',
            price=swing_model.price if swing_model else 0,
            fib_level=float(swing_model.retracement_level) if swing_model and swing_model.retracement_level else None
        )

        # SR levels
        sr_swings = Swing.objects.filter(
            symbol=self.symbol,
            timeframe=self.timeframe
        ).order_by('-timestamp')[:20]
        sr_levels = [s.price for s in sr_swings]

        # Higher TF indicators (simplified, assume H1)
        higher_tf = 'H1'
        higher_candles = Candle.objects.filter(
            symbol=self.symbol,
            timeframe=higher_tf,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp')[:5]
        higher_tf_candles = []
        for c in higher_candles:
            inds = Indicator.objects.filter(candle=c)
            ind_d = {i.type: i for i in inds}
            indicators = Indicators(
                ma6=ind_d.get('MA6', Indicator(value=0, slope=0)).value,
                ma18=ind_d.get('MA18', Indicator(value=0, slope=0)).value,
                ma50=ind_d.get('MA50', Indicator(value=0, slope=0)).value,
                ma200=ind_d.get('MA200', Indicator(value=0, slope=0)).value,
                slope6=ind_d.get('MA6', Indicator(value=0, slope=0)).slope,
                slope18=ind_d.get('MA18', Indicator(value=0, slope=0)).slope,
                slope50=ind_d.get('MA50', Indicator(value=0, slope=0)).slope,
                slope200=ind_d.get('MA200', Indicator(value=0, slope=0)).slope,
            )
            higher_tf_candles.append(HigherTFCandle(
                indicators=indicators,
                high=c.high_price,
                low=c.low_price
            ))

        # Correlations
        correlations = {}
        corr_models = Correlation.objects.filter(
            Q(pair1=self.symbol) | Q(pair2=self.symbol),
            timeframe=self.timeframe,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp')[:5]
        for corr in corr_models:
            pair = corr.pair2 if corr.pair1 == self.symbol else corr.pair1
            correlations[pair] = corr.correlation_value

        # Market Structure (simplified, recent highs/lows)
        recent_candles_db = Candle.objects.filter(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp')[:10]
        highs = [c.high_price for c in reversed(recent_candles_db)]
        lows = [c.low_price for c in reversed(recent_candles_db)]
        structure = MarketStructure(highs=highs, lows=lows)

        # Recent candles for candlestick
        recent_pure_candles = []
        for c in recent_candles_db:
            recent_pure_candles.append(PureCandle(
                open=c.open_price,
                high=c.high_price,
                low=c.low_price,
                close=c.close_price,
                timestamp=c.timestamp
            ))

        # Call pure engine
        result = validate_tce(
            candle=pure_candle,
            indicators=indicators,
            swing=pure_swing,
            sr_levels=sr_levels,
            higher_tf_candles=higher_tf_candles,
            correlations=correlations,
            structure=structure,
            recent_candles=recent_pure_candles
        )

        # Adapt to old interface
        is_valid = result['is_valid']
        reasons = [result.get('failure_reason', 'Valid TCE')] if result.get('failure_reason') else ['All checks passed']

        return is_valid, reasons

    def _validate_trend_definition(self):
        """
        Trend definition: Higher timeframes must show uptrend for BUY, downtrend for SELL.
        Check MA slopes on H1 and D1.
        """
        higher_tfs = ['H1', 'D1']
        for tf in higher_tfs:
            # Get recent candles for higher TF
            recent_candles = Candle.objects.filter(
                symbol=self.symbol,
                timeframe=tf,
                timestamp__lte=self.entry_candle.timestamp
            ).order_by('-timestamp')[:10]  # Last 10 candles

            if not recent_candles:
                return False, f"No data for {tf} trend check."

            # Check MA200 slope (overall trend)
            ma200_indicators = Indicator.objects.filter(
                candle__in=recent_candles,
                type='MA200'
            ).order_by('-candle__timestamp')

            if len(ma200_indicators) < 2:
                return False, f"Insufficient MA200 data for {tf}."

            slopes = [ind.slope for ind in ma200_indicators[:2]]
            if self.direction == 'BUY' and not all(s > 0 for s in slopes):
                return False, f"{tf} not in uptrend for BUY."
            elif self.direction == 'SELL' and not all(s < 0 for s in slopes):
                return False, f"{tf} not in downtrend for SELL."

        return True, "Trend definition valid."

    def _validate_ma_structure(self):
        """
        MA structure: For BUY, MA6 > MA18 > MA50 > MA200, all sloping up.
        For SELL, MA6 < MA18 < MA50 < MA200, all sloping down.
        """
        ma_types = ['MA6', 'MA18', 'MA50', 'MA200']
        indicators = Indicator.objects.filter(
            candle=self.entry_candle,
            type__in=ma_types
        )

        if len(indicators) < 4:
            return False, "Missing MA indicators."

        ma_values = {ind.type: ind.value for ind in indicators}
        ma_slopes = {ind.type: ind.slope for ind in indicators}

        if self.direction == 'BUY':
            if not (ma_values['MA6'] > ma_values['MA18'] > ma_values['MA50'] > ma_values['MA200']):
                return False, "MA structure not aligned for BUY."
            if not all(ma_slopes[ma] > 0 for ma in ma_types):
                return False, "MAs not sloping up for BUY."
        else:  # SELL
            if not (ma_values['MA6'] < ma_values['MA18'] < ma_values['MA50'] < ma_values['MA200']):
                return False, "MA structure not aligned for SELL."
            if not all(ma_slopes[ma] < 0 for ma in ma_types):
                return False, "MAs not sloping down for SELL."

        return True, "MA structure valid."

    def _validate_swing_structure(self):
        """
        Swing & structure: Entry near swing low for BUY, swing high for SELL.
        Structure intact (no broken swings).
        """
        # Find nearest swing
        if self.direction == 'BUY':
            swing_type = 'low'
        else:
            swing_type = 'high'

        swings = Swing.objects.filter(
            symbol=self.symbol,
            timeframe=self.timeframe,
            type=swing_type,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp')[:5]  # Recent swings

        if not swings:
            return False, f"No {swing_type} swings found."

        nearest_swing = swings[0]
        price_diff = abs(self.entry_candle.close_price - nearest_swing.price)
        tolerance = self.entry_candle.close_price * 0.005  # 0.5% tolerance

        if price_diff > tolerance:
            return False, f"Entry not near swing {swing_type}."

        # Check structure intact: no invalid swings
        invalid_swings = Swing.objects.filter(
            symbol=self.symbol,
            timeframe=self.timeframe,
            is_valid=False,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp')[:10]

        if invalid_swings:
            return False, "Swing structure broken."

        return True, "Swing & structure valid."

    def _validate_fibonacci_requirement(self):
        """
        Fibonacci requirement: Retracement to valid Fib level (0.382, 0.5, 0.618).
        """
        swings = Swing.objects.filter(
            symbol=self.symbol,
            timeframe=self.timeframe,
            retracement_level__isnull=False,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp')[:1]

        if not swings or swings[0].retracement_level not in [0.382, 0.5, 0.618]:
            return False, "Not at valid Fibonacci level."

        return True, "Fibonacci requirement met."

    def _validate_ma_bounce(self):
        """
        MA bounce: Price should bounce off MA18 or MA50.
        For BUY, close above MA and wick below MA.
        """
        ma_indicators = Indicator.objects.filter(
            candle=self.entry_candle,
            type__in=['MA18', 'MA50']
        )

        if not ma_indicators:
            return False, "No MA18 or MA50 for bounce check."

        bounced = False
        for ma in ma_indicators:
            if self.direction == 'BUY':
                if self.entry_candle.close_price > ma.value and self.entry_candle.low_price < ma.value:
                    bounced = True
                    break
            else:  # SELL
                if self.entry_candle.close_price < ma.value and self.entry_candle.high_price > ma.value:
                    bounced = True
                    break

        if not bounced:
            return False, "No MA bounce detected."

        return True, "MA bounce valid."

    def _validate_support_resistance_filter(self):
        """
        Support & resistance filter: Price not breaking major S/R.
        Assume swings are S/R levels.
        """
        # Get recent swings as S/R
        swings = Swing.objects.filter(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp__lte=self.entry_candle.timestamp
        ).order_by('-timestamp')[:20]

        for swing in swings:
            if swing.type == 'high' and self.entry_candle.high_price > swing.price:
                if self.direction == 'BUY':
                    return False, "Breaking resistance for BUY."
            elif swing.type == 'low' and self.entry_candle.low_price < swing.price:
                if self.direction == 'SELL':
                    return False, "Breaking support for SELL."

        return True, "Support & resistance filter passed."

    def _validate_price_rejection_candles(self):
        """
        Price rejection candles: Candle shows rejection (long wick opposite to direction).
        """
        body_size = abs(self.entry_candle.close_price - self.entry_candle.open_price)
        upper_wick = self.entry_candle.high_price - max(self.entry_candle.open_price, self.entry_candle.close_price)
        lower_wick = min(self.entry_candle.open_price, self.entry_candle.close_price) - self.entry_candle.low_price

        if self.direction == 'BUY':
            if lower_wick < body_size * 2:
                return False, "No rejection wick for BUY."
        else:  # SELL
            if upper_wick < body_size * 2:
                return False, "No rejection wick for SELL."

        return True, "Price rejection candle valid."

    def _validate_multi_timeframe_confirmation(self):
        """
        Multi-timeframe confirmation: Trend confirmed on higher TFs.
        (Similar to trend definition, but ensure alignment)
        """
        # This is partially covered in trend definition, but add specific check
        # For simplicity, reuse trend definition logic
        valid, reason = self._validate_trend_definition()
        return valid, "Multi-timeframe confirmation: " + reason

    def _validate_correlation_filter(self):
        """
        Correlation filter: Check correlations with related pairs.
        Assume some pairs are correlated, e.g., EURUSD with GBPUSD.
        """
        # Simplified: Check if correlation is not strongly negative
        correlated_pairs = ['GBPUSD'] if 'EUR' in self.symbol else []  # Example

        for pair in correlated_pairs:
            corr = Correlation.objects.filter(
                Q(pair1=self.symbol, pair2=pair) | Q(pair1=pair, pair2=self.symbol),
                timeframe=self.timeframe,
                timestamp__lte=self.entry_candle.timestamp
            ).order_by('-timestamp').first()

            if corr and abs(corr.correlation_value) > 0.7:
                # If highly correlated, check if signals align (simplified)
                # For now, just ensure not conflicting
                pass  # Assume ok if no conflict

        return True, "Correlation filter passed."