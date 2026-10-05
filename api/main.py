"""REST access to precomputed fixed-origin holdout forecasts."""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from api.predictor import M5Predictor
from api.schemas import PredictionRequest


def create_app(output_dir=None):
    directory = Path(output_dir or os.environ.get('M5_OUTPUT_DIR', 'outputs/forecasts'))
    predictor = M5Predictor(directory / 'forecasts.csv', directory / 'summary.json')

    @asynccontextmanager
    async def lifespan(app):
        try:
            predictor.load_model()
        except (OSError, ValueError, KeyError) as exc:
            logging.getLogger(__name__).warning('Artifacts unavailable: %s', exc)
        yield

    app = FastAPI(title='M5 Forecast Artifact API', version='2.0.0', lifespan=lifespan)
    app.state.predictor = predictor

    def ready():
        if not predictor.model_loaded:
            raise HTTPException(503, 'Run the backtest to create forecast artifacts, then restart the API')

    @app.get('/')
    def root():
        return {'service': 'Precomputed M5 holdout forecasts', 'docs': '/docs'}

    @app.get('/health')
    def health():
        return JSONResponse(status_code=200 if predictor.model_loaded else 503,
                            content={'status': 'healthy' if predictor.model_loaded else 'not_ready',
                                     'loaded': predictor.model_loaded, 'version': '2.0.0'})

    @app.get('/model/info')
    def info():
        ready()
        return predictor.summary

    @app.get('/items')
    def items(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        ready()
        all_items = predictor.get_available_items()
        return {'total': len(all_items), 'limit': limit, 'offset': offset,
                'items': all_items[offset:offset + limit]}

    @app.post('/predict')
    def predict(request: PredictionRequest):
        ready()
        try:
            predictions = predictor.predict(request.item_ids)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {'status': 'success', 'mode': predictor.summary['mode'],
                'cutoff': predictor.summary['cutoff'],
                'forecast_dates': predictor.summary['forecast_dates'],
                'data': [{'item_id': key, 'forecasts': value} for key, value in predictions.items()]}

    return app


app = create_app()
