import re
import os
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# --- загружаем словари ---
with open(BASE_DIR / "toxic_replacements.json", encoding="utf-8") as f:
    TOXIC_REPLACEMENTS_BASE = json.load(f)

with open(BASE_DIR / "toxic_substrings.json", encoding="utf-8") as f:
    TOXIC_SUBSTRINGS = json.load(f)


def _is_toxic_by_substring(word: str) -> bool:
    w = word.lower()
    return any(sub in w for sub in TOXIC_SUBSTRINGS)


def build_toxic_dict(
    tat_twl_path: str = "tat_Cyrl_twl.txt",
    lexicon_path: str = "tt_ru_lexicon.csv",
    use_lexicon: bool = True,
) -> dict:
    toxic = {}

    # 1) ручные “умные” замены
    for w, repl in TOXIC_REPLACEMENTS_BASE.items():
        toxic[w.lower()] = repl

    # 2) tat_Cyrl_twl.txt → удалить
    if os.path.exists(tat_twl_path):
        with open(tat_twl_path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    toxic.setdefault(w.lower(), "")

    # 3) tt_ru_lexicon.csv (но только слова с токс-корнем)
    if use_lexicon and os.path.exists(lexicon_path):
        lex = pd.read_csv(lexicon_path)
        if "text" in lex.columns:
            for w in lex["text"].astype(str):
                w = w.strip()
                if not w:
                    continue
                if _is_toxic_by_substring(w):
                    toxic.setdefault(w.lower(), "")

    return toxic


def build_pattern(toxic_dict: dict) -> re.Pattern | None:
    if not toxic_dict:
        return None
    keys_sorted = sorted(toxic_dict.keys(), key=len, reverse=True)
    return re.compile(
        r"\b(" + "|".join(re.escape(w) for w in keys_sorted) + r")\b",
        flags=re.IGNORECASE | re.UNICODE,
    )


def _apply_case(original: str, replacement: str) -> str:
    if not replacement:
        return ""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _detox_with_dict(text: str, toxic_dict: dict, pattern: re.Pattern) -> str:
    def repl(match):
        orig = match.group(0)
        replacement = toxic_dict.get(orig.lower(), "")
        return _apply_case(orig, replacement)

    result = pattern.sub(repl, text)
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    return re.sub(r"\s{2,}", " ", result).strip()


def _hard_filter_residual(text: str) -> str:
    tokens = text.split()
    cleaned = [t for t in tokens if not _is_toxic_by_substring(t)]
    result = " ".join(cleaned)
    result = re.sub(r"\s+([.,!?;:])", r"\1", result)
    return re.sub(r"\s{2,}", " ", result).strip()


# --------- ГЛОБАЛЬНЫЕ СЛОВАРИ И ПАТТЕРНЫ (строим ОДИН РАЗ) ---------
TOXIC_DICT_FULL = build_toxic_dict(use_lexicon=True)
TOXIC_PATTERN_FULL = build_pattern(TOXIC_DICT_FULL)

TOXIC_DICT_BASE = build_toxic_dict(use_lexicon=False)
TOXIC_PATTERN_BASE = build_pattern(TOXIC_DICT_BASE)


def detox(text: str) -> str:
    if not isinstance(text, str):
        return text

    orig_len = len(text.split())

    # шаг 1 — словарь с лексиконом (готовый глобальный)
    if TOXIC_PATTERN_FULL is not None:
        mid = _detox_with_dict(text, TOXIC_DICT_FULL, TOXIC_PATTERN_FULL)
    else:
        mid = text

    # шаг 2 — подчистить остатки по подстрокам
    mid = _hard_filter_residual(mid)

    # fallback: если стало слишком коротко — без лексикона (готовый глобальный)
    if orig_len and len(mid.split()) < max(3, int(orig_len * 0.6)):
        if TOXIC_PATTERN_BASE is not None:
            mid2 = _detox_with_dict(text, TOXIC_DICT_BASE, TOXIC_PATTERN_BASE)
        else:
            mid2 = text
        mid2 = _hard_filter_residual(mid2)
        return mid2

    return mid


def process_tsv(input_path: str, output_path: str):
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    if "tat_toxic" not in df.columns:
        raise ValueError("Нет колонки tat_toxic")

    df["tat_detox1"] = df["tat_toxic"].apply(detox)
    df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Входной TSV")
    parser.add_argument("-o", "--output", help="Выходной TSV")
    args = parser.parse_args()

    process_tsv(args.input, args.output)
