"""
llm_extractor.py
* This script processes the raw parsed email threads (from S1) and uses an LLM to extract structured features
  and canonicalize entities across messages. It maintains a reference dictionary to track name variants,
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Dict, Tuple, Any, List
from openai import OpenAI
from dotenv import load_dotenv
from app import ROOT

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Try to import the official OpenAI client (classic or new); adapt as available
try:
    # Classic openai package
    import openai
    OPENAI_CLIENT_LIB = "openai_classic"
except Exception:
    try:
        openai = OpenAI
        OPENAI_CLIENT_LIB = "openai_new"
    except Exception:
        raise ImportError("No OpenAI client library found. Install 'openai' or 'openai>=1.x'.")

    # === SETUP ===
MODEL_NAME = "gpt-4o"

client = OpenAI(api_key=OPENAI_API_KEY)

# === FILE PATHS ===
ROOT_DIR = Path.cwd().parent.parent
INPUT_DIR = ROOT_DIR / "data" / "processed" / "parsed_emails"
REFERENCE_DICT_PATH = ROOT_DIR / "data" / "processed" / "entity_hints" / "canonical_reference_dict.json"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "canonicalized_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Utilities: load/save reference dict ===
def load_reference_dict() -> Dict[str, Dict]:
    if REFERENCE_DICT_PATH.exists():
        try:
            with open(REFERENCE_DICT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    # Ensure top-level keys exist and are in canonical format
    default = {"name_variants": {}, "company_variants": {}, "email_to_name": {}, "phone_to_owner": {}}
    for k, v in default.items():
        if k not in data:
            data[k] = {}
    return data

def save_reference_dict(data: Dict[str, Dict]):
    with open(REFERENCE_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# === PROMPT ===
PROMPT_TEMPLATE = """
You are a legal-AI entity disambiguation assistant.

Given:
1. A new raw email block.
2. The current canonical reference dictionary with previously seen name variants, company variants, email-to-name, and phone-to-owner mappings.

Your job is to:
- Extract sender, recipients, cc, bcc, subject, date, body, links, phones, company, position, location from the email block.
- Canonicalize all person and company names using the existing canonical reference dictionary if available.
- If a new variation is detected that clearly refers to an existing entity, associate it.
- If it appears to be a new person/company, create a new canonical entry.
- If phone numbers or emails are found, associate them with the correct canonical person.
- If you find any email with just an email address and no name, try to link it to an existing canonical name if possible using the local part of the email address.

Output JSON with:
{{
  "transformed_message": {{
    "sender": {{"name": "", "email": ""}},
    "recipients": [{{"name": "", "email": ""}}],
    "cc": [{{"name": "", "email": ""}}],
    "bcc": [{{"name": "", "email": ""}}],
    "subject": "",
    "date": "",
    "body": "",
    "external_links": [""],
    "contact_numbers": [""],
    "company": "",
    "position": "",
    "location": ""
  }},
  "updated_reference_dict": {{
    "name_variants": {{"Asad Shah": ["Asad Shah", "Asad U. Shah", "asadushah", "Asad", ...]}},
    "company_variants": {{"Savvy Commercial Capital": ["Savvy Commercial Capital", "savvycapital", "Savvy Commercial Capital, LLC", ...]}},
    "email_to_name": {{"Asad Shah": ["asad.shah@gmail.com", "aus@parkimon.com", ...]}},
    "phone_to_owner": {{"Paul Camera": ["800-605-8050", 925-324-6360, ...]}}
  }}
}}

Ensure:
- The response is a single, valid JSON object
- Do NOT include markdown, ```json, explanations, or comments
---
Email block:
{email_block}

Current reference dictionary:
{current_dict}
"""

# === LLM call with robust parsing ===
def call_openai_chat(prompt: str, model: str = MODEL_NAME, max_retries: int = 3, backoff: float = 2.0) -> str:
    """
    Sends prompt to the LLM and returns the assistant text (raw).
    Uses either classic openai.ChatCompletion or new API wrapper if present.
    """
    for attempt in range(1, max_retries + 1):
        try:
            if OPENAI_CLIENT_LIB == "openai_classic":
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                # classic returns choices[0].message.content
                content = resp.choices[0].message.content
                
                # Clean any weird whitespace
                content = content.strip()
            else:
                # new style client (OpenAI from openai import OpenAI): client.chat.completions.create
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1600
                )
                # resp is a ChatCompletion-like object; convert to string safely
                # Some new clients return resp.choices[0].message.content
                try:
                    content = resp.choices[0].message.content
                except Exception:
                    # Fallback to string representation
                    content = str(resp)
            return content
        except Exception as e:
            print(f"[LLM] attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
            else:
                raise
    raise RuntimeError("OpenAI call failed after retries")


def extract_json_from_text(text: str) -> Tuple[bool, Any]:
    """
    Try to parse a JSON object from `text`.
    1) Try json.loads directly.
    2) Attempt to find a balanced JSON substring using brace counting.
    Returns (success, parsed_object_or_raw_text)
    """
    text = text.strip()
    # Attempt 1: direct parse
    try:
        return True, json.loads(text)
    except Exception:
        pass

    # Attempt 2: find the largest balanced JSON object substring
    first_brace = text.find("{")
    if first_brace == -1:
        return False, text

    stack = []
    
    for i in range(first_brace, len(text)):
        ch = text[i]
        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    # balanced from first_brace to i
                    candidate = text[first_brace: i + 1]
                    try:
                        parsed = json.loads(candidate)
                        return True, parsed
                    except Exception:
                        # try to continue - maybe there's a better-balanced internal object
                        pass

    # last resort: try regex to extract any JSON-like block (less robust)
    m = re.search(r"(\{(?:[^{}]|(?R))*\})", text, flags=re.DOTALL)
    if m:
        try:
            return True, json.loads(m.group(1))
        except Exception:
            return False, text

    return False, text


# === Merging helpers ===
def merge_variant_dict(base: Dict[str, List[str]], updates: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge updates where both base and updates are mapping canonical -> [variants].
    Return updated base (mutates).
    """
    for canonical, variants in updates.items():
        if not isinstance(variants, list):
            variants = [variants]
        base.setdefault(canonical, [])
        base[canonical] = list({*base[canonical], *variants})
    return base


def merge_email_to_name(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge the email_to_name field. Support either:
      - updates maps canonical_name -> [emails]
      - or updates maps email -> canonical_name
    We'll detect shape and merge accordingly into canonical->list(emails) and email->name maps.
    """
    for k, v in updates.items():
        if isinstance(v, list):
            # assume canonical_name -> [emails]
            canonical = k
            base.setdefault(canonical, [])
            base[canonical] = list({*base[canonical], *v})
        elif isinstance(v, str) and "@" in k:
            # assume k is an email -> v is canonical name
            email = k
            canonical = v
            # store both shapes by adding to base[canonical]
            base.setdefault(canonical, [])
            base[canonical] = list({*base[canonical], email})
        else:
            # fallback: treat v as list or string
            base.setdefault(k, [])
            if isinstance(v, list):
                base[k] = list({*base[k], *v})
            else:
                base[k] = list({*base[k], v})
    return base


def merge_phone_to_owner(base: Dict[str, List[str]], updates: Dict[str, List[str]]) -> Dict[str, List[str]]:
    # Similar to name variants: canonical owner -> list of phones
    for owner, phones in updates.items():
        if not isinstance(phones, list):
            phones = [phones]
        base.setdefault(owner, [])
        base[owner] = list({*base[owner], *phones})
    return base


# === Canonicalization lookup helpers ===
def normalize_text_key(s: str) -> str:
    return re.sub(r"\s+|[^0-9a-zA-Z@\.]", "", (s or "").strip().lower())

def find_canonical_for_name(raw_name: str, ref_dict: Dict[str, Dict]) -> str:
    """
    Try to find a canonical name for raw_name by checking name_variants (canonical -> variants)
    and email_to_name (canonical -> [emails]).
    Returns canonical name if found else original raw_name.
    """
    if not raw_name:
        return raw_name

    norm = normalize_text_key(raw_name)
    # Check name_variants
    for canonical, variants in ref_dict.get("name_variants", {}).items():
        # check canonical itself
        if normalize_text_key(canonical) == norm:
            return canonical
        for v in variants:
            if normalize_text_key(v) == norm:
                return canonical

    # Check email_to_name mapping: keys may be canonical->emails or email->canonical
    # If any canonical->emails includes raw_name (unlikely), or raw_name looks like email
    if "@" in raw_name:
        email = raw_name.strip().lower()
        # scan mapping: canonical->list(emails)
        for canonical, emails in ref_dict.get("email_to_name", {}).items():
            # if value is a list, or single string
            if isinstance(emails, list):
                if email in [e.lower() for e in emails]:
                    return canonical
            elif isinstance(emails, str):
                if email == emails.lower():
                    return canonical
        # also check reversed mapping (some dicts might be email->canonical)
        for k, v in ref_dict.get("email_to_name", {}).items():
            if "@" in k and k.lower() == email:
                # v might be canonical
                return v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else raw_name)
    # fallback
    return raw_name


def canonicalize_message_fields(msg: Dict[str, Any], ref_dict: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Replace detected names in sender/recipients/cc/bcc/company fields using the ref_dict.
    Also normalize email cases and trim whitespace.
    """
    out = dict(msg)  # shallow copy

    # canonicalize sender
    sender = msg.get("sender") or {}
    s_email = (sender.get("email") or "").strip()
    s_name = (sender.get("name") or "").strip()
    # Prefer email-based canonicalization
    canonical_sender = s_name
    if s_email:
        # try to find via email->name
        cand = find_canonical_for_name(s_email, ref_dict)
        if cand != s_email:
            canonical_sender = cand
    # if not found via email, try name variants
    if canonical_sender == s_name:
        canonical_sender = find_canonical_for_name(s_name, ref_dict)

    out_sender = {"name": canonical_sender, "email": s_email}
    out["sender"] = out_sender

    # canonicalize recipients, cc, bcc lists
    for list_key in ("recipients", "cc", "bcc"):
        new_list = []
        for ent in msg.get(list_key, []) or []:
            if isinstance(ent, dict):
                r_name = (ent.get("name") or "").strip()
                r_email = (ent.get("email") or "").strip()
                if r_email:
                    cand = find_canonical_for_name(r_email, ref_dict)
                    if cand != r_email:
                        canonical_name = cand
                    else:
                        canonical_name = find_canonical_for_name(r_name, ref_dict)
                else:
                    canonical_name = find_canonical_for_name(r_name, ref_dict)
                new_list.append({"name": canonical_name, "email": r_email})
            else:
                # ent might be a raw string "Name <email>"
                # Attempt to parse
                m = re.match(r'(.*)<(.+@.+)>', str(ent))
                if m:
                    parsed_name = m.group(1).strip().strip('"')
                    parsed_email = m.group(2).strip()
                    canonical_name = find_canonical_for_name(parsed_email, ref_dict)
                    if canonical_name == parsed_email:
                        canonical_name = find_canonical_for_name(parsed_name, ref_dict)
                    new_list.append({"name": canonical_name, "email": parsed_email})
                else:
                    # no email, treat ent as name
                    canonical_name = find_canonical_for_name(str(ent), ref_dict)
                    new_list.append({"name": canonical_name, "email": ""})
        out[list_key] = new_list

    # canonicalize company field (if present)
    company = (msg.get("company") or "").strip()
    if company:
        norm = normalize_text_key(company)
        found = None
        for canonical, variants in ref_dict.get("company_variants", {}).items():
            if normalize_text_key(canonical) == norm:
                found = canonical
                break
            for v in variants:
                if normalize_text_key(v) == norm:
                    found = canonical
                    break
            if found:
                break
        out["company"] = found or company
    else:
        out["company"] = company

    # normalize contact numbers list (trim)
    phones = msg.get("contact_numbers") or []
    new_phones = []
    for ph in phones:
        phs = str(ph).strip()
        new_phones.append(phs)
    out["contact_numbers"] = new_phones

    # Ensure other fields exist
    for k in ("subject", "date", "body", "external_links", "position", "location"):
        if k not in out:
            out[k] = msg.get(k, "" if k != "external_links" else [])

    return out


# === Top-level processing for a single thread file ===
def process_thread_file(path: Path, ref_dict: Dict[str, Dict]) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
    """
    Load parsed thread JSON, call LLM for each message block with current reference dict,
    merge LLM canonical hints, canonicalize message fields locally, produce an output structure.
    Returns (canonicalized_thread_dict, updated_ref_dict)
    """
    print(f"\n[THREAD] Processing {path.name} ...")
    with open(path, "r", encoding="utf-8") as f:
        thread = json.load(f)

    thread_id = thread.get("thread_id", path.stem)
    file_name = thread.get("file_name", path.name)
    messages = thread.get("messages", [])

    canonicalized_messages = []

    for idx, msg in enumerate(messages, start=1):
        raw_block = msg.get("raw_block", "") or msg.get("text", "") or ""
        if not raw_block.strip():
            # skip empty blocks but keep id metadata
            canonicalized_messages.append({
                "id": msg.get("id"),
                "parent_id": msg.get("parent_id"),
                "quoted_reply_level": msg.get("quoted_reply_level", 0),
                "note": "empty_raw_block"
            })
            continue

        # Build prompt
        prompt = PROMPT_TEMPLATE.format(email_block=raw_block, current_dict=json.dumps(ref_dict, indent=2))
        try:
            assistant_text = call_openai_chat(prompt)
        except Exception as e:
            print(f"[ERROR] LLM call failed for message {msg.get('id')}: {e}")
            # fallback: try to parse minimal fields from raw_block using regex heuristics
            fallback_msg = {
                "sender": {},
                "recipients": [],
                "cc": [],
                "bcc": [],
                "subject": msg.get("subject", ""),
                "date": msg.get("date", ""),
                "body": raw_block,
                "external_links": [],
                "contact_numbers": [],
                "company": "",
                "position": "",
                "location": ""
            }
            parsed_json = {"transformed_message": fallback_msg, "updated_reference_dict": {}}
        else:
            # parse LLM response as JSON robustly
            ok, parsed = extract_json_from_text(assistant_text)
            if not ok:
                print("[WARN] Could not parse JSON from LLM output; saving raw response for inspection.")
                # Save raw LLM output to a debug file for inspection
                debug_file = OUTPUT_DIR / f"{path.stem}_msg_{idx}_raw_llm.txt"
                debug_file.write_text(assistant_text, encoding="utf-8")
                parsed_json = {"transformed_message": {}, "updated_reference_dict": {}}
            else:
                parsed_json = parsed

        # normalized structure retrieval
        transformed = parsed_json.get("transformed_message") or parsed_json.get("message") or {}
        updated_ref = parsed_json.get("updated_reference_dict") or parsed_json.get("canonical_hints") or {}

        # Merge updated reference dict into ref_dict
        # name_variants
        if "name_variants" in updated_ref:
            merge_variant_dict(ref_dict["name_variants"], updated_ref.get("name_variants", {}))
        # company_variants
        if "company_variants" in updated_ref:
            merge_variant_dict(ref_dict["company_variants"], updated_ref.get("company_variants", {}))
        # email_to_name
        if "email_to_name" in updated_ref:
            # This function merges various shapes of email->name hints
            # We'll keep ref_dict["email_to_name"] in canonical->list(email) shape
            merge_email_to_name(ref_dict["email_to_name"], updated_ref.get("email_to_name", {}))
        # phone_to_owner
        if "phone_to_owner" in updated_ref:
            merge_phone_to_owner(ref_dict["phone_to_owner"], updated_ref.get("phone_to_owner", {}))

        # Ensure there is a transformed message even if empty
        if not transformed:
            transformed = {
                "sender": {},
                "recipients": [],
                "cc": [],
                "bcc": [],
                "subject": msg.get("subject", ""),
                "date": msg.get("date", ""),
                "body": raw_block,
                "external_links": [],
                "contact_numbers": [],
                "company": "",
                "position": "",
                "location": ""
            }

        # Add ids from original
        transformed["id"] = msg.get("id")
        transformed["parent_id"] = msg.get("parent_id")
        transformed["quoted_reply_level"] = msg.get("quoted_reply_level", 0)
        transformed["file_name"] = file_name
        transformed["thread_id"] = thread_id

        # Locally canonicalize fields using the updated ref_dict
        canonical_msg = canonicalize_message_fields(transformed, ref_dict)

        canonicalized_messages.append(canonical_msg)

        # Save incremental reference dict after each message (optional: comment out to save after each thread)
        save_reference_dict(ref_dict)

        print(f"[MSG] processed {msg.get('id')} -> canonical sender: {canonical_msg.get('sender',{}).get('name')}")

    out = {
        "thread_id": thread_id,
        "file_name": file_name,
        "messages": canonicalized_messages
    }
    return out, ref_dict


# === Runner ===
def run_pipeline():
    ref_dict = load_reference_dict()
    print(f"[START] Loaded reference dict with {sum(len(v) for v in ref_dict['name_variants'].values())} name variants (keys: {len(ref_dict['name_variants'])}).")

    input_files = sorted([p for p in INPUT_DIR.glob("*.json")])
    if not input_files:
        print(f"[WARN] No parsed thread files found in {INPUT_DIR}")
        return

    for fpath in input_files:
        try:
            canonical_thread, ref_dict = process_thread_file(fpath, ref_dict)
        except Exception as e:
            print(f"[ERROR] Failed processing {fpath.name}: {e}")
            continue

        out_path = OUTPUT_DIR / f"{fpath.stem}_canonicalized.json"
        with open(out_path, "w", encoding="utf-8") as of:
            json.dump(canonical_thread, of, indent=2, ensure_ascii=False)

        print(f"[SAVED] canonicalized thread -> {out_path.name}")

    # final save
    save_reference_dict(ref_dict)
    print(f"[DONE] Pipeline complete. Updated reference dict saved to {REFERENCE_DICT_PATH}")


if __name__ == "__main__":
    run_pipeline()