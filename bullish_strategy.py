# region imports
from AlgorithmImports import *
# endregion

class CurrencyVolumeImbalanceStrategy(QCAlgorithm):
    """
    Currency Futures - Volume Imbalance Zone Strategy

    Detect candles with abnormally high volume and a strong directional body,
    mark their body range as a zone, wait for price to retrace into the zone,
    confirm with a directional candle or micro structure break, enter long/short.
    Stop beyond the imbalance candle's wick, take profit at its extreme.
    """

    MIN_POSITION_SIZE = 1
    MAX_POSITION_SIZE = 5
    ZONE_EXPIRY_BARS = 300       # ~5 hours at 1-min
    VOLUME_LOOKBACK = 50         # bars for rolling volume average
    VOLUME_MULTIPLIER = 2.0      # volume must exceed avg by this factor
    BODY_RATIO_MIN = 0.6         # minimum body / range ratio
    TREND_EMA_PERIOD = 50        # EMA period on higher timeframe
    TREND_BAR_MINUTES = 15       # higher timeframe bar size

    def initialize(self):
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2023, 12, 31)
        self.set_cash(100000)

        self._future = self.add_future(
            Futures.Currencies.EUR,
            extended_market_hours=True,
            data_mapping_mode=DataMappingMode.LAST_TRADING_DAY,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0
        )
        self._future.set_filter(0, 92)

        self._volume_window = RollingWindow[float](self.VOLUME_LOOKBACK)
        self._current_contract = None
        self._prev_bar = None
        self._zones = []
        self._bar_count = 0

        # Higher-timeframe trend filter
        self._trend_ema = ExponentialMovingAverage(self.TREND_EMA_PERIOD)
        self._trend_consolidator = TradeBarConsolidator(timedelta(minutes=self.TREND_BAR_MINUTES))
        self._trend_consolidator.data_consolidated += self._on_trend_bar

    # ── Data Handling ───────────────────────────────────────────────

    def on_data(self, data: Slice):
        mapped = self._future.mapped
        if mapped is None:
            return

        if self._current_contract != mapped:
            self._current_contract = mapped
            self._volume_window.reset()
            self._prev_bar = None
            # Recreate consolidator for new contract symbol (EMA persists across rolls)
            self._trend_consolidator = TradeBarConsolidator(timedelta(minutes=self.TREND_BAR_MINUTES))
            self._trend_consolidator.data_consolidated += self._on_trend_bar

        if not data.bars.contains_key(mapped):
            return

        bar = data.bars[mapped]
        self._trend_consolidator.update(bar)
        self._detect_volume_imbalance(bar)

        if not self.portfolio.invested:
            self._check_entry(bar)

        self._prev_bar = bar

    # ── Trend Filter ────────────────────────────────────────────────

    def _on_trend_bar(self, sender, bar):
        self._trend_ema.update(bar.end_time, bar.close)

    # ── Volume Imbalance Detection ─────────────────────────────────

    def _detect_volume_imbalance(self, bar):
        self._bar_count += 1

        # Expire old zones
        self._zones = [
            z for z in self._zones
            if z["active"] and (self._bar_count - z["birth_bar"]) < self.ZONE_EXPIRY_BARS
        ]

        vol = float(bar.volume)
        self._volume_window.add(vol)

        if not self._volume_window.is_ready:
            return

        # Volume must be well above average
        avg_vol = sum(self._volume_window) / self._volume_window.count
        if avg_vol == 0 or vol < avg_vol * self.VOLUME_MULTIPLIER:
            return

        # Candle must have a strong directional body
        candle_range = bar.high - bar.low
        if candle_range == 0:
            return
        body = abs(bar.close - bar.open)
        if body / candle_range < self.BODY_RATIO_MIN:
            return

        # Confidence score: 0.0 – 1.0
        # Volume component (0-0.5): how far above the threshold the spike is
        #   At 2x avg → 0.0, at 6x+ avg → 0.5
        vol_ratio = vol / avg_vol
        vol_score = min((vol_ratio - self.VOLUME_MULTIPLIER) / 4.0, 0.5)

        # Body component (0-0.3): how clean the directional candle is
        #   At 0.6 ratio → 0.0, at 1.0 → 0.3
        body_score = (body / candle_range - self.BODY_RATIO_MIN) / (1.0 - self.BODY_RATIO_MIN) * 0.3

        # Confirmation component (0-0.2): both confirmations present on this bar
        is_bull = bar.close > bar.open
        has_micro = (
            self._prev_bar is not None
            and (bar.high > self._prev_bar.high if is_bull else bar.low < self._prev_bar.low)
        )
        confirm_score = 0.2 if has_micro else 0.0

        confidence = vol_score + body_score + confirm_score

        zone_base = {
            "stop": bar.low if is_bull else bar.high,
            "target": bar.high if is_bull else bar.low,
            "direction": "bull" if is_bull else "bear",
            "birth_bar": self._bar_count,
            "active": True,
            "confidence": confidence,
        }

        if is_bull:
            zone_base["top"] = bar.close
            zone_base["bottom"] = bar.open
        else:
            zone_base["top"] = bar.open
            zone_base["bottom"] = bar.close

        self._zones.append(zone_base)

        # Keep only the 10 most recent active zones
        active = [z for z in self._zones if z["active"]]
        if len(active) > 10:
            self._zones = active[-10:]

    # ── Entry Logic ─────────────────────────────────────────────────

    def _check_entry(self, bar):
        # Only enter when price is above the higher-timeframe EMA
        if not self._trend_ema.is_ready or bar.close < self._trend_ema.current.value:
            return

        for zone in self._zones:
            if not zone["active"]:
                continue

            # Only trade bullish zones
            if zone["direction"] != "bull":
                continue

            # Price must touch the zone (the imbalance candle's body)
            if bar.low > zone["top"] or bar.high < zone["bottom"]:
                # Invalidate if price closes beyond the candle's wick
                if zone["direction"] == "bull" and bar.close < zone["stop"]:
                    zone["active"] = False
                if zone["direction"] == "bear" and bar.close > zone["stop"]:
                    zone["active"] = False
                continue

            is_bull = zone["direction"] == "bull"

            # Confirm: directional candle OR micro structure break
            directional_candle = (bar.close > bar.open) if is_bull else (bar.close < bar.open)
            micro_break = (
                self._prev_bar is not None
                and (bar.high > self._prev_bar.high if is_bull else bar.low < self._prev_bar.low)
            )

            if not (directional_candle or micro_break):
                continue

            stop_price = zone["stop"]
            tp_price = zone["target"]

            if is_bull:
                risk = bar.close - stop_price
                reward = tp_price - bar.close
            else:
                risk = stop_price - bar.close
                reward = bar.close - tp_price

            if risk <= 0 or reward <= 0:
                continue

            # Final confidence: zone score + entry-time adjustments
            confidence = zone["confidence"]

            # Freshness bonus: retrace within 30 bars → +0.15, decays to 0
            age = self._bar_count - zone["birth_bar"]
            confidence += max(0, 0.15 * (1 - age / 30))

            # Entry confirmation bonus: both signals present → +0.1
            if directional_candle and micro_break:
                confidence += 0.1

            # Scale position: lerp between min and max based on confidence
            confidence = min(confidence, 1.0)
            size = self.MIN_POSITION_SIZE + confidence * (self.MAX_POSITION_SIZE - self.MIN_POSITION_SIZE)
            size = round(size)
            size = max(self.MIN_POSITION_SIZE, min(self.MAX_POSITION_SIZE, size))

            qty = size if is_bull else -size

            self.market_order(self._current_contract, qty)
            self.stop_market_order(self._current_contract, -qty, stop_price)
            self.limit_order(self._current_contract, -qty, tp_price)

            zone["active"] = False
            side = "Long" if is_bull else "Short"
            self.debug(
                f"[ENTRY] {side} x{size} conf={confidence:.2f} @ {bar.close:.5f} | "
                f"Stop: {stop_price:.5f} | TP: {tp_price:.5f} | "
                f"R:R {reward/risk:.1f}"
            )
            return

    # ── Order Management ────────────────────────────────────────────

    def on_order_event(self, order_event):
        if order_event.status != OrderStatus.FILLED:
            return
        if self._current_contract is None:
            return

        # When a stop or TP fills and flattens the position, cancel the counterpart
        if not self.portfolio.invested:
            self.transactions.cancel_open_orders(self._current_contract)

    # ── Contract Rollover ───────────────────────────────────────────

    def on_symbol_changed_events(self, symbol_changed_events):
        for symbol, changed_event in symbol_changed_events.items():
            old_symbol = self.symbol(changed_event.old_symbol)
            new_symbol = self.symbol(changed_event.new_symbol)
            quantity = self.portfolio[old_symbol].quantity

            tag = (
                f"Rollover at {self.time}: "
                f"{old_symbol.value} -> {new_symbol.value}"
            )

            self.transactions.cancel_open_orders(old_symbol)
            self.liquidate(symbol=old_symbol, tag=tag)
            if quantity != 0:
                self.market_order(new_symbol, quantity, tag=tag)
