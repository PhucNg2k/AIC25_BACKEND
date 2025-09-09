import os
import sys
from dotenv import load_dotenv
import json

from enum import Enum
from pydantic import BaseModel
from google import genai
from google.genai import types  

# Ensure project root is importable when running: python search_api.py
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.dirname(TOOLS_DIR)
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
from config import LLM_MODEL


TRANSLATED_FILE = os.path.join(TOOLS_DIR, "translated_query.json")

def _load_translations() -> dict:
    if os.path.exists(TRANSLATED_FILE):
        with open(TRANSLATED_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def check_translated(file_path):
    file_name = os.path.basename(file_path)
    fname = os.path.splitext(file_name)[0]
    data = _load_translations()
    return data.get(fname, [])

def save_translated(file_path, content):
    existing = check_translated(file_path)
    if existing:
        return
    file_name = os.path.basename(file_path)
    fname = os.path.splitext(file_name)[0]
    data = _load_translations()
    data[fname] = content
    with open(TRANSLATED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class QueryType(str, Enum):
    kis = "kis"
    qa = "qa"
    trake = "trake"

class Query(BaseModel):
    text: str
    type: QueryType
    

class TranslationOutput(BaseModel):
    sentences: list[str]
    
def get_file_content(file_path):
    file_name = os.path.basename(file_path)
    text_type = os.path.splitext(file_name)[0].split('-')[-1]
    valid_values = {t.value for t in QueryType}
    
    if text_type not in valid_values:
        # default to kis if suffix isn't recognized
        text_type = QueryType.kis.value
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
        
    return text_content, QueryType(text_type)


def build_config(query_type: QueryType) -> types.GenerateContentConfig:
    base_rules = (
        "You are a strict translator. Always translate the entire input to English.\n"
        "- Detect the source language automatically.\n"
        "- Translate ALL content to English, including when input mixes languages.\n"
        "- Optimize for CLIP-style retrieval: concise, descriptive, declarative captions.\n"
        "  Prefer concrete nouns, attributes, and actions; avoid pronouns and filler words.\n"
        "  Use present tense; no questions, commands, or speculation.\n"
        "- Do not add explanations, notes, or any extra fields.\n"
        "- Preserve meaning and entities; keep numbers and names faithful.\n"
    )

    if query_type in (QueryType.kis,):
        system_instruction = (
            base_rules +
            "- Split the translation into sentences.\n"
            "- Return a JSON object with key 'sentences' whose value is an array of strings.\n"
            "- Each string MUST be numbered: '1. <sentence>', '2. <sentence>', ...\n"
        )
    else:  # QueryType.qa or QueryType.trake
        system_instruction = (
            base_rules +
            "- The input may include lines like 'E1:', 'E2:', ... indicating specific events.\n"
            "- For each such event line, output ONE English entry following this exact format:\n"
            "  '(E<N>: <concise description> -> <concise moment to choose>)'\n"
            "- Keep the event number N identical to the input (e.g., E1, E2).\n"
            "- If there is also an initial general description without 'E<N>:', translate it as a numbered sentence '1. ...' before the event entries.\n"
            "- Return a JSON object with key 'sentences' whose value is an array of strings containing the numbered description (if present) followed by the '(E<N>: ... -> ...)' entries in order.\n"
        )

    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=TranslationOutput,
    )

def translate_file(file_path: str, llm_client) -> list[str]:
    content = check_translated(file_path)
    if content:
        print("RETRIEVED FROM TRANSLATED")
        return content
    print("TRANSLATING NEW")
    text_content, text_type = get_file_content(file_path)
    q = Query(text=text_content, type=text_type)
    config = build_config(q.type)
    response = llm_client.models.generate_content(
        model=LLM_MODEL,
        config=config,
        contents=q.text.strip()
    )
    parsed: TranslationOutput = response.parsed
    content = parsed.sentences
    save_translated(file_path, content)
    return content 


def main():
    client = genai.Client()
    # Example usage; replace with your own source of Query
    base_path = "../../../query-p2-groupA/"
    sample_file = "query-p2-1-kis.txt"
    fpath = os.path.join(base_path, sample_file)

    sentences = translate_file(fpath, client)

    final_ouptut = '\n'.join(sentences)
    print(final_ouptut)
    
    
    return


if __name__ == "__main__":
    main()
