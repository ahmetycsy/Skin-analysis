"""Supabase (PostgreSQL) bağlantısı: anonim vaka kayıtları ve denetim kaydı.

Ayarlar Streamlit Secrets'tan ortam değişkeni olarak gelir: SUPABASE_URL, SUPABASE_KEY (gizli "secret"/service_role anahtarı).
Tanımlı değilse veritabanı kapalıdır ve uygulama yalnızca bellekte çalışır.
Bağlantı Supabase'in REST arayüzüyle (PostgREST) `requests` üzerinden yapılır; ek kütüphane gerekmez.
Görüntü ve kişisel bilgi (ad, kimlik no) GÖNDERİLMEZ.

Tablolar (SQL kurulumu README'de): cases (vaka başına bir satır), audit (yalnızca eklenen denetim kaydı).
"""
from __future__ import annotations

import os

import requests

TIMEOUT = 8  # saniye; veritabanı yanıt vermezse uygulama beklemeden devam eder


class SupabaseDB:
    def __init__(self, url: str, key: str):
        self.base = url.rstrip("/") + "/rest/v1"
        self.headers = {"apikey": key, "Content-Type": "application/json"}
        if not key.startswith("sb_"):  # eski tip (JWT) service_role anahtarı Authorization başlığında da gönderilir
            self.headers["Authorization"] = f"Bearer {key}"
        self.last_error = None

    # ---------------------------------------------------------------- yardımcı
    def _req(self, method: str, table: str, params=None, json=None, prefer: str | None = None):
        headers = dict(self.headers)
        if prefer:
            headers["Prefer"] = prefer
        try:
            r = requests.request(method, f"{self.base}/{table}", params=params, json=json, headers=headers, timeout=TIMEOUT)
            if r.status_code >= 400:
                self.last_error = f"{r.status_code}: {r.text[:200]}"
                return None
            self.last_error = None
            return r.json() if r.text else []
        except Exception as exc:  # ağ hatası, zaman aşımı, veritabanı uykuda vb.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def ping(self) -> bool:
        return self._req("GET", "cases", params={"select": "case_no", "limit": "1"}) is not None

    # ---------------------------------------------------------------- vakalar
    def upsert_case(self, row: dict) -> bool:
        return self._req("POST", "cases", params={"on_conflict": "case_no"}, json=row,
                         prefer="resolution=merge-duplicates,return=minimal") is not None

    def get_case(self, case_no: str):
        rows = self._req("GET", "cases", params={"select": "*", "case_no": f"eq.{case_no}"})
        return rows[0] if rows else None

    def list_cases(self, limit: int = 300) -> list:
        return self._req("GET", "cases", params={"select": "*", "order": "created_at.desc", "limit": str(limit)}) or []

    # ---------------------------------------------------------------- denetim kaydı
    def add_audit(self, row: dict) -> bool:
        return self._req("POST", "audit", json=row, prefer="return=minimal") is not None

    def list_audit(self, case_no: str | None = None, limit: int = 500) -> list:
        params = {"select": "*", "order": "ts.desc", "limit": str(limit)}
        if case_no:
            params["case_no"] = f"eq.{case_no}"
        return self._req("GET", "audit", params=params) or []


def get_db():
    """Ayarlar tanımlıysa veritabanı nesnesi, değilse None."""
    url, key = os.environ.get("SUPABASE_URL", "").strip(), os.environ.get("SUPABASE_KEY", "").strip()
    if url and key:
        return SupabaseDB(url, key)
    return None
