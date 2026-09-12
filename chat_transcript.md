# Chat Transcript — AI Financial Decision Agent

## User goal
Build a complete AI financial decision agent for a HackerRank hackathon using the provided CSV-style financial datasets.

## Key design decisions
- Use Python + pandas for data processing.
- Use Pydantic for structured, validated decisions.
- Use Gemini through the modern `google-genai` SDK.
- Calculate safe liquidity deterministically:
  `current_available_balance - minimum_balance_to_keep`
- Never allow the AI output to exceed safe liquidity.
- Use user messages, payment options and flexible-expense preferences as decision context.
- Export predictions to CSV.
- Keep logs for failed rows.

## Initial issue
The earlier implementation used `from google import genai` and the environment raised an import error because the Gemini SDK was not installed/configured correctly.

## Resolution
Install the modern SDK:

```bash
python3 -m pip install --upgrade google-genai
```

Verify:

```bash
python3 -c "from google import genai; print('Gemini SDK OK')"
```

## Final pipeline
1. Load the four datasets.
2. Validate required columns.
3. Merge requests with financial profiles.
4. Gather user alerts and payment options.
5. Calculate safe liquidity.
6. Ask Gemini for a structured financial decision.
7. Validate the response with Pydantic.
8. Apply deterministic safety clamps and overrides.
9. Write `outputs/predictions.csv`.

## Important note
The actual HackerRank CSV files were not attached to this chat when this project package was created. Therefore, the package contains the production code and output template but does not fabricate competition predictions. Add the real datasets under `data/` and run the agent to generate the real `predictions.csv`.