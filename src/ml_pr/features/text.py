from __future__ import annotations

import re

import pandas as pd


_ANONYMIZED_TOKEN_RE = re.compile(r"\bX{2,}\b", flags=re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^0-9A-Za-zА-Яа-я]+")


def clean_indication(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    # метки XXXX обозначают скрытые персональные данные и полезного смысла для модели не несут
    text = _ANONYMIZED_TOKEN_RE.sub(" ", str(value))
    # приводит текст к одному виду, чтобы одинаковые слова не считались разными
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.lower().split())
