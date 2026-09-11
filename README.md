# US Macro & Markets Morning Tear-Sheet

> **Status: in development.** The full recruiter-facing README — architecture
> diagram, screenshots, methodology and design notes — is written in Phase 7.

An automated pre-open briefing on US markets and the US economy. Every trading
day a GitHub Actions workflow pulls market and economic data, computes
analytics, and publishes an interactive HTML tear-sheet to GitHub Pages, a PDF,
and an email summary.

## Quick start

```bash
conda create -y -n tearsheet python=3.12
conda activate tearsheet
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
python -m tearsheet run --mock
```

## Data sources

| Domain | Source |
| --- | --- |
| Daily prices | yfinance |
| Intraday (previous session) | Polygon / Massive |
| Economic data | FRED |
| Headlines | Polygon news, yfinance, RSS |

## Licence

MIT — see [LICENSE](LICENSE). This project is for education and information
only and is not investment advice.
