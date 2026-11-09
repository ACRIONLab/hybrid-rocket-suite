# Verification of the Hybrid Rocket Simulator (`core/engine.py`)

## Scope: what this is and is not
This is **code verification**: does the simulator solve its own equations correctly?
It is **not validation**: it does not show that the model matches real hybrid motor
tests. Validation against published test data is listed as future work (section 5).

The tests are in `tests/test_engine_verification.py`. Run `python tests/test_engine_verification.py`
from the repository root; all six tests pass.

## 1. Model being verified
A one-dimensional, quasi-steady model of a single-port cylindrical grain:
- Marxman-type regression law: r_dot = a * G_ox^n, with G_ox = mdot_ox / (pi r^2)
- Fuel mass flow from the burning port surface
- Choked nozzle: Pc = mdot * c* / (Cd * At), with constant c* and Cd
- Isentropic exit Mach number from the area-Mach relation and ideal thrust coefficient
- Port radius advanced with a forward Euler step

## 2. Port regression against the exact solution
For this regression law the port radius has a closed-form solution:

    r(t)^(2n+1) = r0^(2n+1) + (2n+1) * a * (mdot_ox / pi)^n * t

Default inputs (oxidizer flow 1.5 kg/s, initial port diameter 40 mm, 5 s burn):

| Time step dt [s] | Max radius error [%] |
|---|---|
| 0.1 | 0.263 |
| 0.05 | 0.131 |
| **0.02 (default)** | **0.052** |
| 0.01 | 0.026 |
| 0.005 | 0.013 |
| 0.0025 | 0.0065 |

The error falls in proportion to dt (fitted slope 1.00 on a log-log plot), as expected
for forward Euler. See `docs/convergence.png`. At the default step the final port
diameter is 73.07 mm against an exact 73.04 mm.

## 3. Nozzle relations
- Exit Mach number for gamma = 1.4 and A/A* = 1.6875 is 2.000, matching the standard
  isentropic table value.
- The thrust coefficient from the code agrees (relative difference below 1e-9) with an
  independent momentum-balance derivation, F = mdot*ue + (pe - pa)*Ae.

Note: the second check re-derives the same physics by different algebra, so it confirms
the implementation, not the physical assumptions.

## 4. Conservation and consistency
- **Mass:** integrated fuel flow equals the grain mass lost to within 0.08 % at
  dt = 0.01 s (0.38 % at dt = 0.05 s; the gap shrinks with dt).
- **Choked flow:** Pc * Cd * At / c* reproduces the total mass flow at every time step
  (maximum mismatch about 1e-14 %).

## 5. Limitations and next steps
- The characteristic velocity c* and discharge coefficient Cd are fixed inputs. The code
  does not compute combustion chemistry, so c* would come from a tool such as NASA CEA.
- The model is quasi-steady. It has no ignition transient, chamber filling, or
  oxidizer feed-system dynamics.
- Regression follows one power law and ignores erosive and radiation effects.
- **Validation (future work):** compare chamber pressure and thrust curves with a
  published hybrid motor static-fire test, using the paper's inputs and stating the
  source and the error.
