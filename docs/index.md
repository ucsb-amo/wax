# wax

The lab-independent framework under the Weld Lab ARTIQ experiments. The
repository holds two packages:

| Package | Role |
|---|---|
| `waxx` | ARTIQ control and scan framework: the experiment base classes and scan loop, liveOD camera acquisition, the run queue and monitor server, dashboards, calibration write-back |
| `waxa` | Host-side analysis: loading runs, the data vault, fitting, plotting |

`waxx` imports `waxa`, never the other way round. A machine-specific package sits
on top of both; for the potassium machine that is
[kexp](https://ucsb-amo.github.io/k-exp/).

## Guides

The how-to and reference pages for both packages currently live in the
[k-exp wiki](https://github.com/ucsb-amo/k-exp/wiki).

## Contents

```{toctree}
:maxdepth: 2

waxx API reference <api/waxx/index>
waxa API reference <api/waxa/index>
Module index <modindex>
General index <genindex>
building-the-docs
```
