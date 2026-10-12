import os
import glob
import json
import pandas as pd
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

API_KEY = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-3.1-flash-lite"

# Output folder ensure karein
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Schema definitions
class ChargeItem(BaseModel):
    type: Literal["energy", "fixed", "fpa", "quarterly_adjustment", "surcharge", "meter_rent", "subsidy", "other"]
    amount: float

class TaxItem(BaseModel):
    type: Literal["gst", "electricity_duty", "income_tax", "municipal_tax", "other_tax"]
    amount: float

class BillJSON(BaseModel):
    provider: Literal["KE", "LESCO", "IESCO"]
    tariff: Optional[str] = None
    sanctioned_load_kw: Optional[float] = None
    bill_month: Optional[str] = Field(None, description="Format YYYY-MM")
    reading_date: Optional[str] = Field(None, description="Format YYYY-MM-DD")
    issue_date: Optional[str] = Field(None, description="Format YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="Format YYYY-MM-DD")
    previous_reading: Optional[float] = None
    current_reading: Optional[float] = None
    units_consumed: Optional[float] = None
    charges: List[ChargeItem] = []
    total_charges: Optional[float] = None
    taxes: List[TaxItem] = []
    total_taxes: Optional[float] = None
    current_bill: Optional[float] = None
    arrears: Optional[float] = None
    payable_within_due_date: Optional[float] = None
    payable_after_due_date: Optional[float] = None

def get_image_info(bill_path: str):
    mime_type = "image/png" if bill_path.lower().endswith(".png") else "image/jpeg"
    with open(bill_path, "rb") as f:
        return f.read(), mime_type

def process_level1(image_path: str) -> str:
    img_bytes, mime_type = get_image_info(image_path)
    prompt = """
    Extract utility bill details strictly following these rules:
    - Provider must be KE, LESCO, or IESCO. If filename is KESC, provider is KE.
    - Dates: YYYY-MM-DD (or YYYY-MM for bill_month).
    - Numeric values: float/int, no commas, no currency signs.
    - Subsidies and credits must be negative numbers.
    - If a field is missing, obscured, or blank, return null (do not guess).
    - Do NOT extract personal data (names, CNIC, account numbers).
    - Map charge types to: energy, fixed, fpa, quarterly_adjustment, surcharge, meter_rent, subsidy, other.
    - Map tax types to: gst, electricity_duty, income_tax, municipal_tax, other_tax.
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BillJSON,
            temperature=0.0
        )
    )
    parsed = BillJSON.model_validate_json(response.text)
    return parsed.model_dump_json(exclude_none=False)

def process_level2(image_path: str, questions: list[dict]) -> dict:
    img_bytes, mime_type = get_image_info(image_path)
    q_lines = [f"{q['id']}: {q['text']}" for q in questions]
    prompt = f"""
    Answer each customer question in plain English, strictly grounded in the numbers and line items on this bill.
    Do not use external data. If information is not on the bill, explicitly state so.
    Return a JSON object mapping each question ID (e.g. "Q1", "Q2") to its answer string.

    Questions:
    {chr(10).join(q_lines)}
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )
    return json.loads(response.text)

def main():
    # Folder check: Pehle test dhoondega, na mile toh train dhoondega
    if os.path.exists("test/bills"):
        bills_dir = "test/bills"
        template_l1 = "test/csv/level1.csv"
        template_l2 = "test/csv/level2.csv"
    elif os.path.exists("train_bills"):
        bills_dir = "train_bills"
        template_l1 = None
        template_l2 = None
    else:
        print("Koi bills folder nahi mila (na test/bills na train_bills)!")
        return

    # Image files get karein
    image_paths = sorted(glob.glob(os.path.join(bills_dir, "*.jpg")) + 
                         glob.glob(os.path.join(bills_dir, "*.jpeg")) + 
                         glob.glob(os.path.join(bills_dir, "*.png")))

    if not image_paths:
        print(f"Folder '{bills_dir}' mein koi images nahi hain!")
        return

    print(f"Working on {len(image_paths)} bills from '{bills_dir}'...")

    # --- LEVEL 1 ---
    print("\n--- Running Level 1 Extraction ---")
    if template_l1 and os.path.exists(template_l1):
        df_l1 = pd.read_csv(template_l1)
        json_results = []
        for bill_id in df_l1["bill_id"]:
            match = [p for p in image_paths if os.path.splitext(os.path.basename(p))[0] == bill_id]
            if match:
                print(f"Extracting {bill_id}...")
                json_results.append(process_level1(match[0]))
            else:
                json_results.append("{}")
        df_l1["json"] = json_results
    else:
        records = []
        for p in image_paths:
            b_id = os.path.splitext(os.path.basename(p))[0]
            print(f"Extracting {b_id}...")
            j_str = process_level1(p)
            records.append({"bill_id": b_id, "json": j_str})
        df_l1 = pd.DataFrame(records)

    df_l1.to_csv("output/level1.csv", index=False)
    print("Saved output/level1.csv successfully!")

    # --- LEVEL 2 ---
    print("\n--- Running Level 2 Explanations ---")
    if template_l2 and os.path.exists(template_l2):
        df_l2 = pd.read_csv(template_l2)
        answers_map = {}
        for bill_id in df_l2["bill_id"].unique():
            match = [p for p in image_paths if os.path.splitext(os.path.basename(p))[0] == bill_id]
            if match:
                print(f"Answering for {bill_id}...")
                q_list = [{"id": r["question_id"], "text": r["question"]} for _, r in df_l2[df_l2["bill_id"] == bill_id].iterrows()]
                ans_dict = process_level2(match[0], q_list)
                for q_id, a_text in ans_dict.items():
                    answers_map[(bill_id, q_id)] = a_text
        df_l2["answer"] = [answers_map.get((r["bill_id"], r["question_id"]), "") for _, r in df_l2.iterrows()]
    else:
        # Default sample questions for training verification
        default_qs = [
            {"id": f"Q{i+1}", "text": q} for i, q in enumerate([
                "How much of my bill is taxes?",
                "What is the total units consumed?",
                "What is the cost of electricity before taxes?",
                "What is the Fuel Price Adjustment (FPA)?",
                "What is the due date for payment?",
                "What is the amount payable after due date?"
            ])
        ]
        l2_rows = []
        for p in image_paths:
            b_id = os.path.splitext(os.path.basename(p))[0]
            print(f"Answering for {b_id}...")
            ans_dict = process_level2(p, default_qs)
            for q in default_qs:
                l2_rows.append({
                    "bill_id": b_id,
                    "question_id": q["id"],
                    "question": q["text"],
                    "answer": ans_dict.get(q["id"], "")
                })
        df_l2 = pd.DataFrame(l2_rows)

    df_l2.to_csv("output/level2.csv", index=False)
    print("Saved output/level2.csv successfully!")

if __name__ == "__main__":
    main()