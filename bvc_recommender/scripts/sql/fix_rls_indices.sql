-- fix_rls_indices.sql
-- Projet : tlrborylhfwxrrryyism
-- Problème : SQL Editor (rôle postgres) voit MASI depuis 2010-01-04,
--            API REST (clé anon) ne voit que 741 lignes MASI depuis 2022-11-21.
-- Cause : Row Level Security (RLS) filtre les lignes pour anon/authenticated.
--
-- À exécuter dans Dashboard → SQL Editor :
-- https://supabase.com/dashboard/project/tlrborylhfwxrrryyism/sql/new

-- =============================================================================
-- 1) DIAGNOSTIC — policies actuelles (rôle postgres, bypass RLS)
-- =============================================================================

SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
FROM pg_policies
WHERE tablename = 'market_data_indices_historique';

SELECT relrowsecurity AS rls_enabled, relforcerowsecurity AS rls_forced
FROM pg_class
WHERE relname = 'market_data_indices_historique';

-- Comptages côté postgres (référence « vérité »)
SELECT COUNT(*) AS total_lignes
FROM public.market_data_indices_historique;

SELECT code_index, COUNT(*) AS n, MIN(date_index) AS dmin, MAX(date_index) AS dmax
FROM public.market_data_indices_historique
GROUP BY code_index
ORDER BY n DESC;

SELECT COUNT(*) AS masi_lignes, MIN(date_index) AS masi_min, MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';

SELECT COUNT(*) AS masi_avant_2022
FROM public.market_data_indices_historique
WHERE code_index = 'MASI' AND date_index < '2022-01-01';

-- Ligne exemple 2010 (UUID visible dans le dashboard utilisateur)
SELECT id, code_index, date_index, valeur_index, scraped_at
FROM public.market_data_indices_historique
WHERE id = '9c49e61a-e4e5-40a4-b58d-80ba95b869ea';

-- =============================================================================
-- 2) CORRECTION — autoriser SELECT complet pour anon + authenticated
-- =============================================================================
-- Si des policies SELECT restrictives existent déjà, les supprimer d'abord
-- (adapter les noms listés par la requête pg_policies ci-dessus).

ALTER TABLE public.market_data_indices_historique ENABLE ROW LEVEL SECURITY;

-- Supprimer d'éventuelles policies SELECT restrictives (noms courants Supabase)
DROP POLICY IF EXISTS "Enable read access for all users" ON public.market_data_indices_historique;
DROP POLICY IF EXISTS "Public read access" ON public.market_data_indices_historique;
DROP POLICY IF EXISTS "anon_select_indices" ON public.market_data_indices_historique;
DROP POLICY IF EXISTS "anon_select_indices_historique" ON public.market_data_indices_historique;
DROP POLICY IF EXISTS "authenticated_select_indices_historique" ON public.market_data_indices_historique;
DROP POLICY IF EXISTS "Allow public read access on market_data_indices_historique" ON public.market_data_indices_historique;

-- Policies permissives : lecture de TOUTES les lignes
CREATE POLICY "anon_select_indices_historique"
  ON public.market_data_indices_historique
  FOR SELECT
  TO anon
  USING (true);

CREATE POLICY "authenticated_select_indices_historique"
  ON public.market_data_indices_historique
  FOR SELECT
  TO authenticated
  USING (true);

-- (Optionnel) si vous importez via API anon — décommenter :
-- CREATE POLICY "anon_insert_indices_historique"
--   ON public.market_data_indices_historique
--   FOR INSERT
--   TO anon
--   WITH CHECK (true);
-- CREATE POLICY "anon_update_indices_historique"
--   ON public.market_data_indices_historique
--   FOR UPDATE
--   TO anon
--   USING (true)
--   WITH CHECK (true);

-- =============================================================================
-- 3) VÉRIFICATION post-fix (toujours côté postgres)
-- =============================================================================

SELECT COUNT(*) AS masi_total,
       MIN(date_index) AS masi_min,
       MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';
-- Attendu : masi_min <= '2010-01-04', masi_total >> 741

-- =============================================================================
-- 4) Vérification côté client (après fix)
-- =============================================================================
-- Relancer localement :
--   python -m bvc_recommender.scripts.probe_rls_masi
--   python -m bvc_recommender.scripts.masi_diagnostic
--   python -m bvc_recommender.scripts.run_step1
--   python -m bvc_recommender.scripts.run_step2 --tickers ATW IAM BCP
--
-- Critères de succès probe_rls_masi :
--   - user_screenshot_uuid.found_via_anon = 1
--   - counts.masi_lt_2022 > 0
--   - masi_fetch_all.date_min <= '2010-01-04'
