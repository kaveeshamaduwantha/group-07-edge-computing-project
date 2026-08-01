import time
from datetime import datetime


class ACSimulation:
    RULES = {
        "low":    {"ac_on": False, "target_temp": None,  "label": "OFF"},
        "medium": {"ac_on": True,  "target_temp": 24,    "label": "ON 24C"},
        "high":   {"ac_on": True,  "target_temp": 20,    "label": "ON 20C"},
    }

    AC_ACTIVATE_DELAY = 5.0   # seconds

    def __init__(self, room_temp=30.0):
        self.occupancy          = "low"
        self.room_temp          = room_temp
        self.ac_on              = False
        self.target_temp        = None
        self.ac_on_since        = None
        self.total_ac_secs      = 0.0
        self._last_tick         = time.time()
        self.history            = []

        self._pending_occ       = "low"
        self._pending_since     = time.time()

        # Janitor 15-second timer & 5-second check state
        self.janitor_detected   = False
        self.janitor_until      = 0.0
        self.last_janitor_check = 0.0

    def reset(self):
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
        Janitor detection logic:
        - If janitor detected, turn off AC for 15 seconds.
        - Every 5 seconds background check: if janitor still cleaning, add 15s more.
        """
        now = time.time()
        self.janitor_detected = is_janitor

        if is_janitor:
            if now > self.janitor_until:
                self.janitor_until = now + 15.0
                self.last_janitor_check = now
                print(f"[ac_sim] Janitor detected! AC turned OFF for 15 seconds.")
            elif now - self.last_janitor_check >= 5.0:
                self.janitor_until = now + 15.0
                self.last_janitor_check = now
                print(f"[ac_sim] Janitor still cleaning! Extended AC OFF for +15s.")

    def set_target_temp(self, temp: int):
        """Manually adjust target temp up or down."""
        if self.target_temp is not None:
            self.target_temp = max(16, min(28, temp))
            rule_label = f"ON {self.target_temp}C"
            self.RULES[self.occupancy]["target_temp"] = self.target_temp
            self.RULES[self.occupancy]["label"] = rule_label

    def set_thresholds(self, medium_min=None, high_min=None, medium_temp=None, high_temp=None):
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
        new_occ = new_occ.lower()
        if new_occ not in self.RULES:
            return

        now = time.time()

        if new_occ != self._pending_occ:
            self._pending_occ   = new_occ
            self._pending_since = now

        stable_secs = now - self._pending_since
        confirmed   = stable_secs >= self.AC_ACTIVATE_DELAY
        display_occ = new_occ

        # If janitor timer active, force AC OFF
        if now < self.janitor_until:
            self.ac_on = False
            self.occupancy = display_occ
            return

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

        changed       = display_occ != self.occupancy
        self.occupancy = display_occ
        rule          = self.RULES[display_occ]
        prev_ac       = self.ac_on

        self.ac_on       = rule["ac_on"]
        self.target_temp = rule["target_temp"]

        if self.ac_on and not prev_ac:
            self.ac_on_since = now
        elif not self.ac_on and prev_ac:
            if self.ac_on_since:
                self.total_ac_secs += now - self.ac_on_since
            self.ac_on_since = None

        if changed:
            self.history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "occupancy": display_occ.upper(),
                "ac_status": rule["label"],
            })
            self.history = self.history[-50:]

    def tick(self):
        now = time.time()
        dt  = now - self._last_tick
        self._last_tick = now

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
    def current_ac_runtime_secs(self):
        total = self.total_ac_secs
        if self.ac_on and self.ac_on_since:
            total += time.time() - self.ac_on_since
        return total

    @property
    def pending_secs_remaining(self) -> float:
        elapsed = time.time() - self._pending_since
        return max(0.0, self.AC_ACTIVATE_DELAY - elapsed)

    def state_dict(self) -> dict:
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

        # When janitor is active, occupancy level is explicitly reported as OFF
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
