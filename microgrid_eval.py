"""Reproducible 24-hour microgrid dispatch benchmark.

The report's headline values are recorded as reference results, but this module
computes its displayed results from dispatch policies and simulated profiles.
The included heuristic policies are not a trained MADRL implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MicrogridConfig:
    solar_capacity_kw: float = 60.0
    battery_capacity_kwh: float = 100.0
    battery_power_limit_kw: float = 25.0
    dt_hours: float = 0.25
    timesteps: int = 96
    soc_min: float = 0.20
    soc_max: float = 0.90
    initial_soc: float = 0.50
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    random_seed: int = 42


@dataclass(frozen=True)
class DispatchAction:
    """Battery power is positive while charging; shift_kw reduces net load."""

    battery_kw: float = 0.0
    shift_kw: float = 0.0


class Controller(Protocol):
    name: str

    def act(
        self, index: int, time_hours: float, solar_kw: float, load_kw: float,
        tariff_per_kwh: float, soc: float, config: MicrogridConfig,
    ) -> DispatchAction: ...


class MicrogridEnv:
    """Simple grid-connected microgrid with a 15-minute dispatch interval."""

    def __init__(self, config: MicrogridConfig | None = None):
        self.config = config or MicrogridConfig()

    def generate_daily_profiles(self) -> tuple[np.ndarray, ...]:
        """Return deterministic time, solar, load and tariff arrays."""
        cfg = self.config
        time = np.arange(cfg.timesteps, dtype=float) * cfg.dt_hours
        tariff = np.where((time >= 18.0) & (time < 22.0), 0.35, 0.15)
        solar = cfg.solar_capacity_kw * np.exp(-0.5 * ((time - 12.0) / 2.5) ** 2)
        rng = np.random.default_rng(cfg.random_seed)
        load = 20.0 + 25.0 * np.exp(-0.5 * ((time - 19.0) / 2.0) ** 2)
        load += 10.0 * rng.random(cfg.timesteps)
        return time, solar, load, tariff

    def run(self, controller: Controller) -> dict[str, float | str | int]:
        cfg = self.config
        time, solar, load, tariff = self.generate_daily_profiles()
        soc = cfg.initial_soc
        total_cost = 0.0
        peak_grid_kw = 0.0
        violations = 0
        shifted_energy_kwh = 0.0

        for i in range(cfg.timesteps):
            action = controller.act(i, time[i], solar[i], load[i], tariff[i], soc, cfg)
            requested_battery = float(action.battery_kw)
            requested_shift = float(action.shift_kw)

            # Count requested battery operation that breaches power or SoC limits.
            if abs(requested_battery) > cfg.battery_power_limit_kw + 1e-9:
                violations += 1
            battery_kw = float(np.clip(
                requested_battery, -cfg.battery_power_limit_kw, cfg.battery_power_limit_kw
            ))
            energy_delta = self._soc_energy_delta(battery_kw, cfg)
            proposed_soc = soc + energy_delta / cfg.battery_capacity_kwh
            if proposed_soc < cfg.soc_min - 1e-9 or proposed_soc > cfg.soc_max + 1e-9:
                violations += 1
                # Enforce physical limits while counting the policy's violation.
                bounded_soc = float(np.clip(proposed_soc, cfg.soc_min, cfg.soc_max))
                energy_delta = (bounded_soc - soc) * cfg.battery_capacity_kwh
                battery_kw = self._power_for_soc_delta(energy_delta, cfg)
                proposed_soc = bounded_soc

            net_load_kw = max(0.0, load[i] - requested_shift)
            grid_kw = max(0.0, net_load_kw + battery_kw - solar[i])
            total_cost += tariff[i] * grid_kw * cfg.dt_hours
            peak_grid_kw = max(peak_grid_kw, grid_kw)
            shifted_energy_kwh += requested_shift * cfg.dt_hours
            soc = proposed_soc

        return {
            "Control method": controller.name,
            "Daily operating cost ($)": total_cost,
            "Battery limit violations": violations,
            "Peak grid power (kW)": peak_grid_kw,
            "Final state of charge": soc,
            "Net load shifted (kWh)": shifted_energy_kwh,
        }

    @staticmethod
    def _soc_energy_delta(power_kw: float, cfg: MicrogridConfig) -> float:
        if power_kw >= 0:
            return power_kw * cfg.dt_hours * cfg.charge_efficiency
        return power_kw * cfg.dt_hours / cfg.discharge_efficiency

    @staticmethod
    def _power_for_soc_delta(energy_delta_kwh: float, cfg: MicrogridConfig) -> float:
        if energy_delta_kwh >= 0:
            return energy_delta_kwh / (cfg.dt_hours * cfg.charge_efficiency)
        return energy_delta_kwh * cfg.discharge_efficiency / cfg.dt_hours


class RuleBasedController:
    name = "Standard Rule-Based Control"

    def act(self, index, time_hours, solar_kw, load_kw, tariff_per_kwh, soc, config):
        surplus = solar_kw - load_kw
        if surplus > 0 and soc < config.soc_max:
            battery_kw = min(surplus, config.battery_power_limit_kw)
        elif surplus < 0 and soc > config.soc_min:
            battery_kw = -min(-surplus, config.battery_power_limit_kw)
        else:
            battery_kw = 0.0
        return DispatchAction(battery_kw=battery_kw)


class UncoordinatedController:
    """Independent price-following battery and evening-load heuristics."""

    name = "Uncoordinated RL (No Coach Critic)"

    def act(self, index, time_hours, solar_kw, load_kw, tariff_per_kwh, soc, config):
        # The independent battery heuristic responds to tariff but does not
        # forecast the evening demand or coordinate with load shifting.
        if tariff_per_kwh >= 0.35 and soc > config.soc_min:
            battery_kw = -22.0
        elif tariff_per_kwh <= 0.15 and soc < config.soc_max and solar_kw < load_kw:
            battery_kw = 12.0
        else:
            battery_kw = 0.0
        shift_kw = 3.0 if 18.0 <= time_hours < 22.0 else 0.0
        return DispatchAction(battery_kw=battery_kw, shift_kw=shift_kw)


class CoordinatedController:
    """Coordinated heuristic for solar charging, peak discharge and load shifting."""

    name = "Coordinated Multi-Agent Strategy (MADRL-style heuristic)"

    def act(self, index, time_hours, solar_kw, load_kw, tariff_per_kwh, soc, config):
        solar_surplus = solar_kw - load_kw
        if solar_surplus > 2.0 and soc < config.soc_max:
            battery_kw = min(solar_surplus, config.battery_power_limit_kw)
        elif tariff_per_kwh >= 0.35 and soc > config.soc_min + 0.03:
            battery_kw = -config.battery_power_limit_kw
        elif tariff_per_kwh <= 0.15 and soc < 0.45 and solar_kw < load_kw:
            battery_kw = 8.0
        else:
            battery_kw = 0.0

        # Move a matching amount of demand from the evening peak into the
        # solar-rich midday window; positive shift reduces load at that time.
        if 11.0 <= time_hours < 15.0:
            shift_kw = -3.0
        elif 18.0 <= time_hours < 22.0:
            shift_kw = 3.0
        else:
            shift_kw = 0.0
        return DispatchAction(battery_kw=battery_kw, shift_kw=shift_kw)


REPORT_REFERENCE_RESULTS = pd.DataFrame([
    ["Standard Rule-Based Control", 48.60, 0, 54.2],
    ["Uncoordinated RL (No Coach Critic)", 43.10, 4, 44.8],
    ["Proposed Multi-Agent RL (MADRL)", 39.65, 0, 36.5],
], columns=["Control method", "Daily operating cost ($)",
            "Battery limit violations", "Peak grid power (kW)"])


def evaluate_strategies() -> pd.DataFrame:
    """Run the three executable baseline policies and return computed metrics."""
    env = MicrogridEnv()
    controllers = [RuleBasedController(), UncoordinatedController(), CoordinatedController()]
    return pd.DataFrame(env.run(controller) for controller in controllers)


def main() -> None:
    print("\nMADRL MICROGRID ENERGY MANAGEMENT — 24-HOUR SIMULATION")
    print("System: 60 kW solar | 100 kWh battery | 15-minute intervals")
    print("Computed results (not hard-coded report outcomes):")
    print(evaluate_strategies().to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nReport reference results for comparison:")
    print(REPORT_REFERENCE_RESULTS.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nNote: MADRL-style policy here is a transparent heuristic, not a trained agent.")


if __name__ == "__main__":
    main()
