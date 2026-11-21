# wax

The lab-independent framework under the Weld Lab ARTIQ experiments. This
repository holds two packages:

- `waxx` (`waxx-src/`): ARTIQ control and scan framework, liveOD camera
  acquisition, the run queue and monitor server, dashboards.
- `waxa` (`waxa-src/`): host-side analysis: loading runs, fitting, plotting.

`waxx` imports `waxa`, never the other way round.

## Documentation

- **Documentation and API reference:** https://ucsb-amo.github.io/wax/
- **Guides:** currently in the [k-exp wiki](https://github.com/ucsb-amo/k-exp/wiki)
