# Contributing

## Tests (no Azure keys required)

```bash
pip install pytest numpy rank-bm25
pytest tests/ -v
```

## Full stack

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app/ui_streamlit.py
```

## Guidelines

- Evaluation changes should update `eval/` metrics and `README.md` benchmarks.
- Keep prompts in `prompts/` versioned separately from agent logic.
