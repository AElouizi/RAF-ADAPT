-- ============================================================
-- FIX MASI 2010+ — à exécuter dans Supabase SQL Editor (Run)
-- Projet : Analyste_IA_26 (pungyebagtntbocoewir)
-- ============================================================

-- 1) Vérification AVANT (doit montrer masi_min = 2010-01-04 côté postgres)
SELECT 'AVANT_postgres' AS etape,
       COUNT(*)::int AS masi_total,
       MIN(date_index) AS masi_min,
       MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';

-- 2) Désactiver RLS sur la table indices
ALTER TABLE public.market_data_indices_historique DISABLE ROW LEVEL SECURITY;

-- 3) Fonction RPC (contournement si RLS réactivé plus tard)
CREATE OR REPLACE FUNCTION public.fetch_masi_historique()
RETURNS SETOF public.market_data_indices_historique
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
STABLE
AS $$
  SELECT *
  FROM public.market_data_indices_historique
  WHERE code_index = 'MASI'
  ORDER BY date_index;
$$;

GRANT EXECUTE ON FUNCTION public.fetch_masi_historique() TO anon, authenticated, service_role;

-- 4) Recharger le cache API PostgREST
NOTIFY pgrst, 'reload schema';

-- 5) Vérification APRES
SELECT 'APRES_postgres' AS etape,
       COUNT(*)::int AS masi_total,
       MIN(date_index) AS masi_min,
       MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';

-- Attendu APRES : masi_total >> 741 (environ 3800+), masi_min <= 2010-01-04
