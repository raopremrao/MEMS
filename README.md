# Microgrid Energy Management System (MEMS)

A small, reproducible 24-hour microgrid benchmark based on the parameters and comparison table in `AIML_IA.pdf`.

## Report reference results

The existing report's Table 1 values are retained as reference data in the code:

| Control method | Daily operating cost | Battery limit violations | Peak grid power |
| --- | ---: | ---: | ---: |
| Standard Rule-Based Control | $48.60 | 0 | 54.2 kW |
| Uncoordinated RL (No Coach Critic) | $43.10 | 4 | 44.8 kW |
| Proposed Multi-Agent RL (MADRL) | $39.65 | 0 | 36.5 kW |

These are **reported reference outcomes**, not results produced by training the included policies. Running the code computes outcomes from the explicit dispatch logic below; it does not substitute the report values for simulation results.

## Run the simulation

Requires Python 3.10+.

```bash
python -m pip install -r requirements.txt
python microgrid_eval.py
```

The simulation uses a deterministic synthetic 24-hour solar/load/tariff profile (seed 42), a 60 kW solar array, a 100 kWh battery, 15-minute steps, and 20–90% battery state-of-charge limits. Output includes computed energy cost, peak grid draw, operating-limit violations, final battery state of charge, and shifted energy. Change `MicrogridConfig` in `microgrid_eval.py` to explore assumptions.

## Included controllers and limitations

- **Standard rule-based:** charges from solar surplus and discharges when load exceeds solar.
- **Uncoordinated:** simple price-following battery and evening-load heuristics.
- **Coordinated MADRL-style heuristic:** coordinates solar-aware charging, peak-time discharge, and a balanced midday/evening load-shift schedule.

The third controller is a heuristic baseline, **not a learned multi-agent reinforcement-learning system**. The project does not claim to reproduce or independently validate the paper's exact results. The simplified model also excludes generator dispatch, export revenue, degradation, network constraints, and stochastic uncertainty beyond the seeded synthetic load profile.

## Tests

```bash
python -m unittest discover -s tests -v
```
