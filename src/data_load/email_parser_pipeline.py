# Level 1. Email Thread Parsing Pipeline
# This is the First stage of the pipeline. It focuses on parsing the raw PDF files to extract email threads and their hierarchical structure. The output is a structured JSON format that captures the thread ID, file name, and a list of messages with their content, quoted reply levels, and parent-child relationships.
import os
import pymupdf as parser
import re
from datetime import datetime
from typing import List, Dict
import json
from pathlib import Path

# --- CONFIG ---

ROOT_DIR = Path(__file__).parent.parent
PDF_FOLDER = ROOT_DIR / "data" / "case_documents"  # Adjust this path as needed
PARSED_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "parsed_emails"

# --- UTILITIES ---
def extract_text_from_pdf(pdf_path: str) -> str:
    # PDF data extraction using PyMuPDF parser
    doc = parser.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    return text

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "").replace("\t", " ")
    text = re.sub(r"\n{2,}", "\n\n", text.strip())
    return text

def extract_top_level_blocks(text: str) -> List[str]:
    """Extracts blocks starting with 'From:' iteratively."""
    pattern = re.compile(r"^From: .+", re.MULTILINE)    # Look for lines starting with "From:" as block separators
    matches = list(pattern.finditer(text))    # Creates a list of blocks based on the positions of "From:" lines.
    if not matches:
        return [text.strip()]
    
    blocks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)  # End of block is start of next "From:" or end of text
        blocks.append(text[start:end].strip())  # Finally strips the data on matched blocks and adds to the list of blocks.
    return blocks

def extract_nested_replies(block: str) -> List[str]:
    """
    * Extracts in-thread replies (On <date>, wrote:) from inside a block.
    * For each block extracted by extract_top_level_blocks, this function looks for lines that indicate quoted replies and splits the block accordingly.
    """
    pattern = re.compile(r"(?=^On .+?wrote:)|(?=^\S.+? wrote:)", re.MULTILINE)    # Look for lines starting with "On <date>, wrote:" or "<name> wrote:"
    matches = list(pattern.finditer(block))
    if not matches:
        return [block.strip()]
    
    replies = []
    for i, match in enumerate(matches):
        # Indexing the start of each matched reply block, and then slicing the original block to extract the content of each reply. The end of each reply block is determined by the start of the next match or the end of the block.
        start = match.start()   
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        replies.append(block[start:end].strip())
    return replies

def structure_messages_from_blocks(top_blocks: List[str]) -> List[Dict]:
    """
    Converts extracted top-level blocks into structured messages with IDs and parent-child relationships.
    Splits each block on the first reply marker and creates message hierarchy.
    """
    messages = []
    msg_counter = 1

    for block in top_blocks:
        # Look for the first reply marker inside the block
        reply_match = re.search(r"(?=^On .+?wrote:)|(?=^\S.+? wrote:)", block, re.MULTILINE)
        
        if reply_match:
            reply_start = reply_match.start()

            # Original part BEFORE the reply
            original_part = block[:reply_start].strip()
            if original_part:
                original_id = f"{msg_counter:04d}"
                messages.append({
                    "id": original_id,
                    "raw_block": original_part,
                    "quoted_reply_level": 0,
                    "parent_id": None
                })
                msg_counter += 1
            else:
                original_id = None

            # Quoted reply part
            reply_part = block[reply_start:].strip()
            if reply_part:
                messages.append({
                    "id": f"{msg_counter:04d}",
                    "raw_block": reply_part,
                    "quoted_reply_level": estimate_reply_level(reply_part),
                    "parent_id": original_id
                })
                msg_counter += 1
        else:
            # No reply marker: treat whole thing as one original block
            messages.append({
                "id": f"{msg_counter:04d}",
                "raw_block": block,
                "quoted_reply_level": 0,
                "parent_id": None
            })
            msg_counter += 1

    return messages

def estimate_reply_level(text: str) -> int:
    return len(re.findall(r"On .*? wrote:", text, re.IGNORECASE))

def extract_timestamp_from_filename(filename: str) -> str:
    try:
        match = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})[ _](\d{1,2})-(\d{2})-(\d{2})", filename)
        if match:
            month, day, year, hour, minute, second = match.groups()
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
            return dt.strftime("%m-%d-%Y_%H%M")
    except Exception:
        pass
    return Path(filename).stem.replace(" ", "_").replace("/", "_")

# --- MAIN PIPELINE ---
def process_pdf_as_thread(pdf_path: str) -> Dict:
    raw_text = extract_text_from_pdf(pdf_path)
    clean_text = normalize_whitespace(raw_text)
    top_blocks = extract_top_level_blocks(clean_text)

    file_name = os.path.basename(pdf_path)
    thread_id = extract_timestamp_from_filename(file_name)   # Using the timestamp extracted from the filename as the thread ID for better organization and uniqueness.
    out_dir = Path(PARSED_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = structure_messages_from_blocks(top_blocks)
    thread_data = {
        "thread_id": thread_id,
        "file_name": file_name,
        "messages": messages
    }

    out_path = out_dir / f"{thread_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thread_data, f, indent=2)

    return thread_data

def process_all_pdfs(pdf_folder: str) -> List[Dict]:
    all_threads = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            print(f"📄 Processing: {filename}")
            pdf_path = os.path.join(pdf_folder, filename)
            thread = process_pdf_as_thread(pdf_path)
            all_threads.append(thread)
    return all_threads

# --- MAIN RUN ---
if __name__ == "__main__":
    structured_threads = process_all_pdfs(PDF_FOLDER)
    print(f"\n✅ Total threads processed: {len(structured_threads)}")
