from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import pandas as pd


GENERIC_PROBLEM_TERMS = {
    "normal",
    "no indexing",
    "lung",
    "spine",
    "thoracic vertebrae",
    "aorta",
    "diaphragm",
}


def extract_problem_terms(value: object) -> list[str]:

    if value is None:
        return ["normal"]
    if isinstance(value, float) and math.isnan(value):
        return ["normal"]

    text = str(value).strip()
    if not text:
        return ["normal"]

    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in text.split(";"):
        term = " ".join(raw_term.strip().split())
        if not term:
            continue
        key = term.lower()
        if key not in seen:
            terms.append(term)
            seen.add(key)

    return terms or ["normal"]


def is_abnormal_terms(terms: Iterable[str]) -> int:
    # abnormal означает наличие хотя бы одной записи о патологии в поле Problems
    normal_terms = {"normal", "no indexing"}
    return int(any(term.lower() not in normal_terms for term in terms))


def sanitize_label_name(term: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", term).strip("_")
    return cleaned or "label"


def top_problem_terms(
    reports: pd.DataFrame,
    top_k: int = 5,
    exclude_terms: set[str] | None = None,
) -> list[str]:
    # общие анатомические слова не берутся как отдельные диагнозы
    exclude = GENERIC_PROBLEM_TERMS if exclude_terms is None else exclude_terms
    counts: Counter[str] = Counter()

    for value in reports["Problems"].fillna(""):
        for term in extract_problem_terms(value):
            if term.lower() not in exclude:
                counts[term] += 1

    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
