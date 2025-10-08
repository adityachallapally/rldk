!!! info "Methods and Config"
    - **Commit**: {commit}
    - **Seeds**: Fixture replay uses seeds `2` and `42`; the determinism demo locks to `1234` for scalar/vector noise.
    - **Guardrail**: `value > 0.35` for 5 consecutive samples with 5 step grace
    - **Hardware**: Dockerized x86 CPU run (8 vCPUs, 32 GB RAM); CuDNN disabled so GPU variance stays out of the demo.
    - **Determinism card**: `docs/assets/blog_catch_failures/determinism_det/determinism_card.json`
