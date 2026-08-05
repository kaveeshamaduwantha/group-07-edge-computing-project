"""
=============================================================================
ac_simulation.py — Smart Classroom Edge AI | AC & Thermal Physics Engine
=============================================================================
This module simulates thermal dynamics and rule-based AC control:
  1. Hysteresis Delay (5s): Occupancy changes must remain stable for 5 seconds
     before the AC updates power/target temperature.
  2. Janitor Safety Override (15s): If a janitor/cleaner is detected, the AC is
     immediately forced OFF for 15 seconds. Periodic checks (+15s) extend the delay.
  3. Newton's Law of Cooling:
     - AC ON: room_temp cools towards target temp (-0.5°C / sec)
     - AC OFF: room_temp warms towards ambient 30.0°C (+0.2°C / sec)
=============================================================================
"""

import time
from datetime import datetime


class ACSimulation:
    """
    Simulates classroom Air Conditioning (AC) behavior, room thermal physics,
    occupancy stability verification (hysteresis), and janitor safety overrides.
    """

    # Default rule set mapping occupancy band -> AC power and target temperature
    RULES = {
        "low":    {"ac_on": False, "target_temp": None,  "label": "OFF"},
        "medium": {"ac_on": True,  "target_temp": 24,    "label": "ON 24C"},
        "high":   {"ac_on": True,  "target_temp": 20,    "label": "ON 20C"},
    }

    # 5-second hysteresis confirmation delay to prevent AC short-cycling
    AC_ACTIVATE_DELAY = 5.0   # seconds

    def __init__(self, room_temp: float = 30.0):
        """Initialize AC state, timers, and thermal simulation parameters."""
        self.occupancy          = "low"        # Current active occupancy band
        self.room_temp          = room_temp    # Simulated room temperature in °C
        self.ac_on              = False        # Power status of AC compressor
        self.target_temp        = None         # Target temperature setpoint in °C
        self.ac_on_since        = None         # Timestamp when AC was turned ON
        self.total_ac_secs      = 0.0          # Accumulated AC operational runtime
        self._last_tick         = time.time()  # Last physics calculation timestamp
        self.history            = []           # Log of state changes

        # Hysteresis delay tracking
        self._pending_occ       = "low"        # Target occupancy awaiting 5s confirmation
        self._pending_since     = time.time()  # Timestamp when target occupancy changed

        # Janitor override state tracking
        self.janitor_detected   = False        # True if janitor detected in current frame
        self.janitor_until      = 0.0          # Unix timestamp until AC must remain OFF
        self.last_janitor_check = 0.0          # Timestamp of last background check

    def reset(self):
        """Reset all state and timers to factory defaults (e.g. on video stop)."""
        self.occupancy          = "low"
        self.room_temp          = 30.0
        self.ac_on              = False
        self.target_temp        = None
        self.ac_on_since        = None
        self.total_ac_secs      = 0.0
        self._last_tick         = time.time()
        self.history            = []
        self._pending_occ       = "low"
        self._pending_since     = time.time()
        self.janitor_detected   = False
        self.janitor_until      = 0.0
        self.last_janitor_check = 0.0

    def update_janitor_status(self, is_janitor: bool):
        """
        PROCESS: Janitor Safety Override Logic
        ---------------------------------------
        - If a janitor/cleaner is detected: Force AC OFF for 15 seconds.
        - Every 5 seconds during active cleaning: Extend the timer by +15 seconds.
        """
        now = time.time()
        self.janitor_detected = is_janitor

        if is_janitor:
            if now > self.janitor_until:
                # First detection: start 15s timer
                self.janitor_until = now + 15.0
                self.last_janitor_check = now
                print("[ac_sim] Janitor detected! AC turned OFF for 15 seconds.")
            elif now - self.last_janitor_check >= 5.0:
                # Extension check: janitor still present, add 15s more
                self.janitor_until = now + 15.0
                self.last_janitor_check = now
                print("[ac_sim] Janitor still cleaning! Extended AC OFF for +15s.")

    def set_target_temp(self, temp: int):
        """Manually adjust target setpoint temperature (clamped 16°C – 28°C)."""
        if self.target_temp is not None:
            self.target_temp = max(16, min(28, temp))
            rule_label = f"ON {self.target_temp}C"
            self.RULES[self.occupancy]["target_temp"] = self.target_temp
            self.RULES[self.occupancy]["label"] = rule_label

    def set_thresholds(self, medium_min=None, high_min=None, medium_temp=None, high_temp=None):
        """Update target temperatures and occupancy boundaries dynamically."""
        if medium_temp is not None:
            self.RULES["medium"]["target_temp"] = int(medium_temp)
            self.RULES["medium"]["label"] = f"ON {int(medium_temp)}C"
        if high_temp is not None:
            self.RULES["high"]["target_temp"] = int(high_temp)
            self.RULES["high"]["label"] = f"ON {int(high_temp)}C"
        if medium_min is not None:
            self._medium_min = int(medium_min)
        if high_min is not None:
            self._high_min = int(high_min)

    def update_occupancy(self, new_occ: str):
        """
        PROCESS: Occupancy Evaluation & Hysteresis Delay
        ------------------------------------------------
        Updates occupancy state after verifying that the candidate state
        has remained constant for at least AC_ACTIVATE_DELAY (5.0) seconds.
        Also enforces janitor safety override if active.
        """
        new_occ = new_occ.lower()
        if new_occ not in self.RULES:
            return

        now = time.time()

        # Step 1: Detect candidate state change
        if new_occ != self._pending_occ:
            self._pending_occ   = new_occ
            self._pending_since = now

        # Step 2: Check 5-second hysteresis stability condition
        stable_secs = now - self._pending_since
        confirmed   = stable_secs >= self.AC_ACTIVATE_DELAY
        display_occ = new_occ

        # Step 3: Enforce Janitor Safety Override (forces AC OFF)
        if now < self.janitor_until:
            self.ac_on = False
            self.occupancy = display_occ
            return

        # Step 4: If not yet confirmed, log PENDING status
        if not confirmed:
            if display_occ != self.occupancy:
                self.occupancy = display_occ
                self.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "occupancy": display_occ.upper(),
                    "ac_status": f"PENDING ({self.AC_ACTIVATE_DELAY:.0f}s delay)…",
                })
                self.history = self.history[-50:]
            return

        # Step 5: Confirmed change — update AC power state and target temp
        changed        = display_occ != self.occupancy
        self.occupancy = display_occ
        rule           = self.RULES[display_occ]
        prev_ac        = self.ac_on

        self.ac_on       = rule["ac_on"]
        self.target_temp = rule["target_temp"]

        # Track total AC runtime accumulation
        if self.ac_on and not prev_ac:
            self.ac_on_since = now
        elif not self.ac_on and prev_ac:
            if self.ac_on_since:
                self.total_ac_secs += now - self.ac_on_since
            self.ac_on_since = None

        # Log state change to event history
        if changed:
            self.history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "occupancy": display_occ.upper(),
                "ac_status": rule["label"],
            })
            self.history = self.history[-50:]

    def tick(self):
        """
        PROCESS: Thermal Physics Simulation Step
        -----------------------------------------
        Uses Newtonian cooling rate equations to simulate room temperature change:
          - When AC is ON:  room_temp cools towards target temp (-0.5°C/sec)
          - When AC is OFF: room_temp warms towards ambient 30.0°C (+0.2°C/sec)
        """
        now = time.time()
        dt  = now - self._last_tick
        self._last_tick = now

        # Force AC OFF during janitor override
        if now < self.janitor_until:
            self.ac_on = False

        if self.ac_on and self.target_temp is not None:
            diff = self.room_temp - self.target_temp
            if diff > 0:
                self.room_temp -= min(diff, 0.5 * dt)
        else:
            diff = 30.0 - self.room_temp
            if diff > 0:
                self.room_temp += min(diff, 0.2 * dt)
        self.room_temp = round(self.room_temp, 2)

    @property
    def current_ac_runtime_secs(self) -> float:
        """Calculate total cumulative AC operational runtime in seconds."""
        total = self.total_ac_secs
        if self.ac_on and self.ac_on_since:
            total += time.time() - self.ac_on_since
        return total

    @property
    def pending_secs_remaining(self) -> float:
        """Calculate remaining seconds in the 5-second hysteresis countdown."""
        elapsed = time.time() - self._pending_since
        return max(0.0, self.AC_ACTIVATE_DELAY - elapsed)

    def state_dict(self) -> dict:
        """
        PROCESS: Format Telemetry Dictionary for /api/status Endpoint
        Returns structured dictionary containing all current AC readings,
        timers, temperatures, janitor status, and event logs.
        """
        now = time.time()
        janitor_remaining = max(0.0, self.janitor_until - now)
        is_janitor_active = janitor_remaining > 0

        rule    = self.RULES[self.occupancy]
        runtime = self.current_ac_runtime_secs
        h = int(runtime // 3600)
        m = int((runtime % 3600) // 60)
        s = int(runtime % 60)
        remaining = self.pending_secs_remaining

        ac_on_final = self.ac_on and not is_janitor_active
        ac_label    = "OFF (JANITOR)" if is_janitor_active else (rule["label"] if ac_on_final else "OFF")

        display_occ = "OFF" if is_janitor_active else self.occupancy.upper()

        return {
            "occupancy":         display_occ,
            "ac_on":             ac_on_final,
            "ac_label":          ac_label,
            "target_temp":       self.target_temp if ac_on_final else None,
            "room_temp":         self.room_temp,
            "ac_runtime_secs":   round(runtime, 1),
            "ac_runtime_str":    f"{h:02d}:{m:02d}:{s:02d}",
            "history":           self.history[-20:],
            "pending_occ":       self._pending_occ.upper(),
            "pending_secs":      round(remaining, 1),
            "ac_confirmed":      remaining == 0.0,
            "janitor_active":    is_janitor_active,
            "janitor_remaining": round(janitor_remaining, 1),
        }
