-- Supabase → SQL Editor → New query → bu dosyanın tamamını yapıştır → Run.
-- Önceki kurulumu (cases, audit) BOZMAZ; yalnızca ekler. Birden fazla çalıştırmak zararsızdır.

-- 1) Lezyon zaman çizelgesi: vakayı aynı lezyonun önceki vakasına bağlayan sütun
alter table public.cases add column if not exists prev_case text;

-- 2) Kullanılabilirlik anketi (SUS): anonim, kişisel bilgi yok
create table if not exists public.sus (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  role text,
  lang text,
  app_version text,
  answers jsonb not null,
  score numeric not null check (score >= 0 and score <= 100)
);
alter table public.sus enable row level security;   -- ziyaretçiler doğrudan erişemez
grant select, insert on public.sus to service_role; -- yalnızca sitenin gizli anahtarı okur/yazar (silme/güncelleme yok)
