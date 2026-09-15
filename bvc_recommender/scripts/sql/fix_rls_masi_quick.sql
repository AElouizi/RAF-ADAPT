-- Fix RLS — market_data_indices_historique (MASI 2010+)
-- Coller TOUT ce fichier dans Supabase SQL Editor et exécuter (Run).

-- A) État actuel (postgres)
SELECT 'AVANT' AS etape,
       COUNT(*) AS masi_total,
       MIN(date_index) AS masi_min,
       MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';

SELECT policyname, roles, cmd, qual
FROM pg_policies
WHERE tablename = 'market_data_indices_historique';

-- B) Supprimer TOUTES les policies existantes sur cette table
DO $$
DECLARE pol RECORD;
BEGIN
  FOR pol IN
    SELECT policyname
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'market_data_indices_historique'
  LOOP
    EXECUTE format(
      'DROP POLICY IF EXISTS %I ON public.market_data_indices_historique',
      pol.policyname
    );
  END LOOP;
END $$;

-- C) Option la plus simple pour une table de marché en lecture publique :
ALTER TABLE public.market_data_indices_historique DISABLE ROW LEVEL SECURITY;

-- (Si vous préférez garder RLS activé, commentez la ligne ci-dessus
--  et décommentez le bloc ci-dessous à la place.)
/*
ALTER TABLE public.market_data_indices_historique ENABLE ROW LEVEL SECURITY;
CREATE POLICY "anon_select_indices_historique"
  ON public.market_data_indices_historique FOR SELECT TO anon USING (true);
CREATE POLICY "authenticated_select_indices_historique"
  ON public.market_data_indices_historique FOR SELECT TO authenticated USING (true);
*/

-- D) Vérification après fix (postgres)
SELECT 'APRES' AS etape,
       COUNT(*) AS masi_total,
       MIN(date_index) AS masi_min,
       MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';

-- Attendu : masi_min <= 2010-01-04, masi_total >> 741
