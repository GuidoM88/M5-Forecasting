# Multivariate Retail Forecasting

Progetto di forecasting per migliaia di serie temporali di vendite retail, con focus su serie intermittenti e conformal predictions.

Questo progetto usa il dataset M5 di Walmart (42.840 serie di vendite), e gli obiettivi sono i seguenti:

- Forecasting scalabile di migliaia di serie in parallelo
- Gestione di serie intermittenti (Croston, SBA)
- Conformal predictions per intervalli di confidenza
- Pipeline MLOps completa fino al deployment

## Setup

pip install -r requirements.txt

python src/data/download.py

jupyter notebook notebooks/

## Struttura

- `data/` - dataset raw e processati
- `notebooks/` - analisi esplorative
- `src/` - codice modulare
- `pipelines/` - training e inference automatizzati
- `api/` - REST API per serving
- `tests/` - unit tests

Il progetto parte dall'esplorazione dei dati, passa per l'implementazione di vari modelli, integra le conformal predictions e conclude con una pipeline MLOps containerizzata.