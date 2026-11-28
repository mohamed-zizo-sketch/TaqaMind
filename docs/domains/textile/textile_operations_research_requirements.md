# Textile Operations Research & Production Optimization Requirements (Task 14)

- **Module:** TaqaMind Module 4 — Operations Research and Production Optimization
- **Domain Scope:** `textile` (`weaving`, `knitting`)
- **Status:** Requirements research only (no universal solver or DB schema implemented)

---

## 1. Decision Variables vs Fixed & Uncertain Inputs

Before formulating any solver, variables must be strictly separated:
- **Controllable Decision Variables:** Order-to-machine assignment ($x_{o,m,t} \in \{0,1\}$), shift allocation ($s_{o,m,k}$), target machine speed within OEM limits ($v_{o,m} \in [v_{\min}, v_{\max}]$), changeover/maintenance window timing ($u_{m,t} \in \{0,1\}$).
- **Fixed Machine/Fabric Characteristics (Not Decision Variables):** Machine rated power (`kW`), knitting cylinder diameter/gauge, loom reed width/type (`air_jet` vs `rapier`), customer-specified yarn count (`Ne`/`tex`) and fabric density (`EPI`/`PPI`, `target_gsm`).
- **Uncertain External Conditions:** Ambient temperature/humidity, unplanned yarn breaks, grid voltage/power outages.

---

## 2. Prioritized Textile Optimization Problems

### Problem 1: Waste- & Cost-Aware Order-to-Machine Assignment (Weaving & Knitting)
- **Industrial Problem:** Factories have heterogeneous looms (air-jet, rapier, water-jet) and circular knitting machines (varying gauge $E$, diameter, feeders) with different waste and energy profiles per fabric spec.
- **Objective Function:** Minimize total expected variable manufacturing cost + changeover penalty + lateness penalty:
  $$\min \sum_{o,m} \left( c^{\text{waste}}_{o,m} \cdot \hat{W}_{o,m} + c^{\text{elec}}_{o,m} \cdot \hat{E}_{o,m} \right) x_{o,m} + \sum_{m,o_1,o_2} c^{\text{setup}}_{m,o_1,o_2} y_{m,o_1,o_2} + \sum_o p^{\text{late}}_o L_o$$
- **Hard Constraints:**
  - Machine compatibility (reed width $\ge$ fabric width; gauge/diameter compatible with yarn Tex & target GSM).
  - Order quantity fulfillment ($\sum_m q_{o,m} = Q_o$).
  - Shift & machine capacity ($\sum_o d_{o,m} x_{o,m} + \text{setup} \le H^{\text{avail}}_m$).
  - Raw yarn inventory availability per yarn lot.
  - Approved OEM speed limits ($v_{\min,m} \le v_{o,m} \le v_{\max,m}$) and maintenance windows.
- **Soft Constraints:** Delivery deadline preferences ($L_o \ge 0$), balanced machine utilization.
- **Granularity & Units:** `production_order` -> `machine` / `shift` (`h`, `kg` or `m`/`yd`, `EGP`).
- **Recommended Method:** **Mixed Integer Linear Programming (MILP)** or **Constraint Programming (CP-SAT)** because assignment ($x_{o,m} \in \{0,1\}$) and sequence-dependent beam/creel setups ($y_{m,o_1,o_2}$) are combinatorial with linear capacity/material balances.

### Problem 2: Energy-Aware Shift & Time-of-Use (TOU) Tariff Scheduling
- **Industrial Problem:** Scheduling energy-intensive air-jet weaving or high-feeder knitting runs across off-peak vs peak electricity tariff windows and plant peak-demand (`kVA`/`kW`) caps.
- **Objective Function:** Minimize electricity cost ($\sum_{m,t} \tau_t \cdot \hat{E}_{m,t} \cdot z_{m,t}$) + peak demand charge ($\pi^{\text{peak}} P_{\max}$).
- **Hard Constraints:** Plant substation/transformer power cap ($\sum_m P^{\text{rated}}_m z_{m,t} \le P^{\text{site}}_{\max}$), shift labor availability, non-preemptive beam/batch run continuity where required, delivery deadlines.
- **Soft Constraints:** Minimize overtime shift premiums.
- **Recommended Method:** **MILP** (time-indexed binary/integer scheduling with linear TOU tariffs and peak-demand epigraph constraint).

### Problem 3: Speed-Set-Point Optimization (Subject to ML Surrogate Bounds)
- **Industrial Problem:** Selecting planned machine speed (`planned_loom_speed_rpm` or `planned_machine_speed_rpm`) to balance throughput against predicted waste ($\hat{W}(v)$) and energy ($\hat{E}(v)$).
- **Recommended Method:** **Discrete Speed-Grid MILP** (discretizing allowed RPM into verified operating bands $[v_1, \dots, v_K]$) or **Piecewise-Linear Approximation** rather than unconstrained black-box nonlinear optimization, preventing extrapolation outside training data.

---

## 3. ML-in-the-Loop Safety & Feasibility Guardrails

Whenever Waste V1 ($\hat{W}$) or Energy V2 ($\hat{E}$) predictions appear in an objective or constraint:
1. **Valid Operating Envelope (No Extrapolation):** Decision variables (e.g., `planned_loom_speed_rpm`, `planned_utilization_pct`) must be clamped strictly to the intersection of OEM physical limits and the factory's observed historical training support $[q_{0.01}, q_{0.99}]$.
2. **Uncertainty-Aware Margin:** Use conservative upper prediction intervals ($\hat{W} + k \cdot \sigma_W$) when checking hard raw-material or energy-cap constraints.
3. **Tractability of Repeated Inference:** Prefer pre-computed lookup tables over discrete `(order, machine, speed_band)` tuples before invoking MILP/CP-SAT, avoiding millions of runtime `sklearn` calls inside solver callbacks.
4. **Physical Feasibility Gate:** Solver outputs must pass a post-solve contract/feasibility check (`mass_balance`, `speed <= OEM max`, `deadline`) before presentation to plant planners.
