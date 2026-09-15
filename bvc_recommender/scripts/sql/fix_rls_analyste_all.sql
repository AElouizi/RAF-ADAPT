-- Analyste_IA_26 (pungyebagtntbocoewir)
-- Si SQL Editor montre des données mais l'API anon renvoie 0 ligne : exécuter ce script.

DO $$
DECLARE
  tbl text;
  pol record;
  tables text[] := ARRAY[
    'fondamentaux_rs',
    'donnees_financieres',
    'market_data_cours_historique',
    'market_data_indices_historique',
    'entreprises',
    'periodes',
    'secteurs',
    'indicateurs',
    'sources',
    'types_etats'
  ];
BEGIN
  FOREACH tbl IN ARRAY tables LOOP
    IF EXISTS (
      SELECT 1 FROM information_schema.tables
      WHERE table_schema = 'public' AND table_name = tbl
    ) THEN
      EXECUTE format('ALTER TABLE public.%I DISABLE ROW LEVEL SECURITY', tbl);
      FOR pol IN
        SELECT policyname FROM pg_policies
        WHERE schemaname = 'public' AND tablename = tbl
      LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I', pol.policyname, tbl);
      END LOOP;
    END IF;
  END LOOP;
END $$;

NOTIFY pgrst, 'reload schema';

-- Contrôles (postgres)
SELECT 'fondamentaux_rs' AS tbl, COUNT(*)::bigint AS n FROM public.fondamentaux_rs
UNION ALL SELECT 'donnees_financieres', COUNT(*)::bigint FROM public.donnees_financieres
UNION ALL SELECT 'market_data_cours_historique', COUNT(*)::bigint FROM public.market_data_cours_historique
UNION ALL SELECT 'market_data_indices_historique', COUNT(*)::bigint FROM public.market_data_indices_historique
ORDER BY tbl;
