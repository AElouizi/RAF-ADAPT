-- Analyste_IA_26 (pungyebagtntbocoewir)
-- La nouvelle table fondamentaux_rs_bis renvoie 0 ligne via l'API anon
-- (RLS activé par défaut sans policy). Exécuter ce script dans le SQL Editor.

DO $$
DECLARE
  pol record;
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'fondamentaux_rs_bis'
  ) THEN
    EXECUTE 'ALTER TABLE public.fondamentaux_rs_bis DISABLE ROW LEVEL SECURITY';
    FOR pol IN
      SELECT policyname FROM pg_policies
      WHERE schemaname = 'public' AND tablename = 'fondamentaux_rs_bis'
    LOOP
      EXECUTE format(
        'DROP POLICY IF EXISTS %I ON public.fondamentaux_rs_bis', pol.policyname
      );
    END LOOP;
  END IF;
END $$;

NOTIFY pgrst, 'reload schema';

-- Contrôle : doit renvoyer le nombre réel de lignes
SELECT 'fondamentaux_rs_bis' AS tbl, COUNT(*)::bigint AS n
FROM public.fondamentaux_rs_bis;
