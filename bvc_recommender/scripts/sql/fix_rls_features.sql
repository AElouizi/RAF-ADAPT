-- Analyste_IA_26 — tables features (étape 2/3) : lecture + écriture via clé anon
-- Exécuter APRÈS create_features_tables.sql

DO $$
DECLARE
  tbl text;
  pol record;
  tables text[] := ARRAY[
    'features_fondamentales',
    'features_techniques',
    'features_indices'
  ];
BEGIN
  FOREACH tbl IN ARRAY tables LOOP
    IF EXISTS (
      SELECT 1 FROM information_schema.tables
      WHERE table_schema = 'public' AND table_name = tbl
    ) THEN
      EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', tbl);
      FOR pol IN
        SELECT policyname FROM pg_policies
        WHERE schemaname = 'public' AND tablename = tbl
      LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I', pol.policyname, tbl);
      END LOOP;
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR SELECT TO anon, authenticated USING (true)',
        tbl || '_select_all', tbl
      );
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR INSERT TO anon, authenticated WITH CHECK (true)',
        tbl || '_insert_all', tbl
      );
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true)',
        tbl || '_update_all', tbl
      );
      EXECUTE format(
        'CREATE POLICY %I ON public.%I FOR DELETE TO anon, authenticated USING (true)',
        tbl || '_delete_all', tbl
      );
    END IF;
  END LOOP;
END $$;

NOTIFY pgrst, 'reload schema';

SELECT 'features_fondamentales' AS tbl, COUNT(*)::bigint AS n FROM public.features_fondamentales
UNION ALL SELECT 'features_techniques', COUNT(*)::bigint FROM public.features_techniques
UNION ALL SELECT 'features_indices', COUNT(*)::bigint FROM public.features_indices
ORDER BY tbl;
