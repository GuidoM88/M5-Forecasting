# Docker API

Generate artifacts before starting the service:

```bash
python -m pip install -e ".[api]"
python -m scripts.make_demo_data
python -m scripts.train_hierarchical_lgbm --config config/demo.yaml
```

Select the demo artifact directory, then run Compose:

```bash
# Linux/macOS
export M5_ARTIFACT_DIR=./outputs/demo
# PowerShell: $env:M5_ARTIFACT_DIR="./outputs/demo"
docker compose up --build
```

Without `M5_ARTIFACT_DIR`, Compose mounts `./outputs/forecasts`, produced by the full-data config. Artifacts are mounted read-only and are not baked into the image. The image can build before artifacts exist; `/health` returns 503 until valid artifacts are present and the service is restarted.

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- IDs: http://localhost:8000/items
- Logs: `docker compose logs -f api`
- Stop: `docker compose down`

The image installs the core/API dependencies and LightGBM's OpenMP runtime only. Its health check uses Python's standard library and fails on HTTP errors. No `curl` binary or model-training environment is required. Docker runtime validation was not available in the repair environment.
