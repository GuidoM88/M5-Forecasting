# Docker Deployment

## Quick Start

Build and run with Docker Compose:

docker-compose up --build

API will be available at: http://localhost:8000

## Manual Build

Build image:

docker build -t m5-forecasting-api .

Run container:

docker run -d --name m5-api -p 8000:8000 -v $(pwd)/outputs:/app/outputs:ro m5-forecasting-api

## Test Endpoints

Health check:

curl http://localhost:8000/health

Model info:

curl http://localhost:8000/model/info

Predict:

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"item_ids": ["HOBBIES_1_001_CA_1_evaluation"]}'

## Stop and Remove

With docker-compose:

docker-compose down

Manually:

docker stop m5-api
docker rm m5-api

## Troubleshooting

View logs:

docker-compose logs -f api

Enter container:

docker exec -it m5-forecasting-api bash
