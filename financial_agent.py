import os
import json
from pathlib import Path
from typing import Optional, Literal

import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

REQUESTS_FILE = DATA_DIR / "requests.csv"
PROFILES_FILE = DATA_DIR / "financial_profiles.csv"
PAYMENTS_FILE = DATA_DIR / "request_payment_options.csv"
MESSAGES_FILE = DATA_DIR / "messages.csv"

OUTPUT_FILE = OUTPUT_DIR / "predictions.csv"
LOG_FILE = OUTPUT_DIR / "agent.log"


class FinancialDecision(BaseModel):
    amount_safe_to_pay: float = Field(description="Maximum amount safely payable today.")
    affordability_status: Literal[
        "affordable_now", "affordable_with_plan",
        "affordable_later", "not_affordable"
    ]
    recommended_payment_method: Literal[
        "full_payment", "partial_payment", "installments", "wait", "none"
    ]
    payment_plan: str
    earliest_date_for_full_payment: Optional[str] = None
    spending_changes_needed: str
    decision_explanation: str


def safe_float(value, default=0.0):
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_text(value, default="Not provided"):
    if pd.isna(value):
        return default
    return str(value)


def load_data():
    required = [REQUESTS_FILE, PROFILES_FILE, PAYMENTS_FILE, MESSAGES_FILE]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s):\n" + "\n".join(missing)
        )

    requests = pd.read_csv(REQUESTS_FILE)
    profiles = pd.read_csv(PROFILES_FILE)
    payments = pd.read_csv(PAYMENTS_FILE)
    messages = pd.read_csv(MESSAGES_FILE)

    required_columns = {
        "requests": {"request_id", "user_id", "requested_amount"},
        "profiles": {"user_id", "current_available_balance", "minimum_balance_to_keep"},
        "payments": {"request_id"},
        "messages": {"user_id", "message_text"},
    }
    frames = {
        "requests": requests,
        "profiles": profiles,
        "payments": payments,
        "messages": messages,
    }
    for name, cols in required_columns.items():
        missing_cols = cols - set(frames[name].columns)
        if missing_cols:
            raise ValueError(f"{name}.csv missing columns: {sorted(missing_cols)}")

    return requests.merge(profiles, on="user_id", how="left"), payments, messages


def get_user_messages(user_id, messages_df):
    messages = messages_df[messages_df["user_id"] == user_id]["message_text"].dropna().tolist()
    return "\n".join(f"- {m}" for m in messages) if messages else "- No recent alerts."


def get_payment_options(request_id, payments_df):
    options = payments_df[payments_df["request_id"] == request_id]
    if options.empty:
        return "No payment options available."
    return json.dumps(
        options.drop(columns=["request_id"], errors="ignore").to_dict(orient="records"),
        default=str
    )


def calculate_safe_liquidity(row):
    balance = safe_float(row.get("current_available_balance"))
    minimum = safe_float(row.get("minimum_balance_to_keep"))
    return max(0.0, balance - minimum)


def evaluate_request(row, payments_df, messages_df, client):
    current_balance = safe_float(row.get("current_available_balance"))
    minimum_balance = safe_float(row.get("minimum_balance_to_keep"))
    requested_amount = safe_float(row.get("requested_amount"))
    liquidity = calculate_safe_liquidity(row)

    currency = safe_text(row.get("home_currency"), "")
    allows_partial = safe_bool(row.get("allows_partial_payment"))
    user_id = row["user_id"]
    request_id = row["request_id"]

    prompt = f"""
You are a strict financial decision-support AI agent for a hackathon demo.
Do not present yourself as a licensed financial advisor.
Never invent financial data.

HARD SAFETY LIMIT:
The user's minimum required balance must never be violated.
amount_safe_to_pay MUST be between 0 and {liquidity:.2f}.

FINANCIAL DATA
Current balance: {current_balance:.2f} {currency}
Minimum required balance: {minimum_balance:.2f} {currency}
Safe liquidity today: {liquidity:.2f} {currency}
Requested amount: {requested_amount:.2f} {currency}

USER PREFERENCES
Flexible expenses: {safe_text(row.get("expense_categories_user_is_willing_to_reduce"))}
Payment methods considered: {safe_text(row.get("payment_methods_user_will_consider"))}
Allows partial payment: {allows_partial}

RECENT ALERTS
{get_user_messages(user_id, messages_df)}

REQUEST
{safe_text(row.get("request_text"))}
Desired completion date: {safe_text(row.get("desired_completion_date"))}

AVAILABLE PAYMENT OPTIONS
{get_payment_options(request_id, payments_df)}

RULES
1. Never go below the minimum required balance.
2. Never recommend more than safe liquidity today.
3. If the full amount is safe today, choose affordable_now + full_payment.
4. Otherwise consider only supported partial payments/installments.
5. Only suggest reductions from listed flexible expenses.
6. Do not invent future income, dates, fees, or payment options.
7. If no safe path exists, choose not_affordable + none.
8. Keep the explanation concise.
Return only the requested structured object.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=FinancialDecision,
        ),
    )

    decision = FinancialDecision.model_validate_json(response.text)

    # Non-negotiable post-LLM safety clamp.
    decision.amount_safe_to_pay = max(
        0.0, min(float(decision.amount_safe_to_pay), liquidity)
    )

    # Deterministic override for the most important safety cases.
    if requested_amount <= liquidity:
        decision.amount_safe_to_pay = requested_amount
        decision.affordability_status = "affordable_now"
        decision.recommended_payment_method = "full_payment"
        decision.payment_plan = f"Pay {requested_amount:.2f} {currency} in full today."
        decision.earliest_date_for_full_payment = None
    elif liquidity <= 0:
        decision.amount_safe_to_pay = 0.0
        decision.affordability_status = "not_affordable"
        decision.recommended_payment_method = "none"
        decision.payment_plan = "Do not make the payment today; preserve the required balance."
    elif not allows_partial and decision.recommended_payment_method == "partial_payment":
        decision.recommended_payment_method = "installments"

    return decision


def process_request(row, payments, messages, client):
    try:
        d = evaluate_request(row, payments, messages, client)
        return {
            "request_id": row["request_id"],
            "user_id": row["user_id"],
            "requested_amount": safe_float(row.get("requested_amount")),
            "amount_safe_to_pay": round(d.amount_safe_to_pay, 2),
            "affordability_status": d.affordability_status,
            "recommended_payment_method": d.recommended_payment_method,
            "payment_plan": d.payment_plan,
            "earliest_date_for_full_payment": d.earliest_date_for_full_payment,
            "spending_changes_needed": d.spending_changes_needed,
            "decision_explanation": d.decision_explanation,
        }
    except Exception as exc:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[ERROR] {row.get('request_id')} -> {exc}\n")
        return None


def run():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. See .env.example and README.md.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    client = genai.Client(api_key=api_key)
    df, payments, messages = load_data()

    results = []
    for _, row in df.iterrows():
        result = process_request(row, payments, messages, client)
        if result:
            results.append(result)

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Processed {len(results)}/{len(df)} requests")
    print(f"Predictions: {OUTPUT_FILE}")
    if results:
        print(pd.DataFrame(results)["affordability_status"].value_counts())


if __name__ == "__main__":
    run()