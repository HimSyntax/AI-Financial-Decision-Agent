# AI Financial Decision Agent

A hackathon-ready financial decision-support agent that combines deterministic safety rules with Gemini structured output.

## Architecture

CSV datasets → Data aggregation → Safe liquidity calculation → Gemini decision agent → Pydantic validation → hard safety guardrail → `predictions.csv`

## Project structure

```text
ai_financial_decision_agent/
├── data/
│   ├── requests.csv
│   ├── financial_profiles.csv
│   ├── request_payment_options.csv
│   └── messages.csv
├── outputs/
│   ├── predictions.csv
│   └── agent.log
├── src/
│   └── financial_agent.py
├── docs/
│   └── chat_transcript.md
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. Put the four HackerRank CSV datasets into `data/`.
2. Create a Gemini API key.
3. Set it in your shell:

```bash
export GEMINI_API_KEY="YOUR_KEY"
```

Or load it through your preferred environment-secret mechanism.

4. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

5. Run:

```bash
python3 src/financial_agent.py
```

The generated predictions will be written to `outputs/predictions.csv`.

## Expected CSV columns

### requests.csv
Required: `request_id`, `user_id`, `requested_amount`

The code also reads: `request_text`, `desired_completion_date`, `allows_partial_payment`.

### financial_profiles.csv
Required: `user_id`, `current_available_balance`, `minimum_balance_to_keep`

The code also reads: `home_currency`, `expense_categories_user_is_willing_to_reduce`, `payment_methods_user_will_consider`.

### request_payment_options.csv
Required: `request_id`. Any additional columns are passed to the model as available options.

### messages.csv
Required: `user_id`, `message_text`.

## Safety design

The LLM does not control the hard financial ceiling. Safe liquidity is calculated in Python as:

`max(0, current_available_balance - minimum_balance_to_keep)`

The model output is then clamped to this value. Full-payment decisions are also overridden deterministically when the requested amount is safely affordable.

This is a demo decision-support system, not a licensed financial-advice service.

## Hackathon pitch

**Problem:** Users may need to decide whether they can make a payment without compromising a required cash buffer.

**Solution:** The agent combines structured financial data, user preferences, alerts and payment options, then produces an explainable payment recommendation.

**Differentiator:** The LLM proposes the strategy, but deterministic guardrails enforce the non-negotiable financial safety limit.