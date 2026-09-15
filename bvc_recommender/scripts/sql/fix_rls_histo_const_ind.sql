-- Analyste_IA_26 — lecture histo_const_ind + indicateurs_financiers via clé anon
-- Nécessaire pour PE/PB (cours, nombre_titre) et mapping indicateurs EAV

GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT SELECT ON TABLE public.histo_const_ind TO anon, authenticated;
GRANT SELECT ON TABLE public.indicateurs_financiers TO anon, authenticated;

ALTER TABLE public.histo_const_ind ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.indicateurs_financiers ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "bvc_rs_anon_read_histo_const_ind" ON public.histo_const_ind;
CREATE POLICY "bvc_rs_anon_read_histo_const_ind"
  ON public.histo_const_ind
  FOR SELECT
  TO anon, authenticated
  USING (true);

DROP POLICY IF EXISTS "bvc_rs_anon_read_indicateurs_financiers" ON public.indicateurs_financiers;
CREATE POLICY "bvc_rs_anon_read_indicateurs_financiers"
  ON public.indicateurs_financiers
  FOR SELECT
  TO anon, authenticated
  USING (true);

NOTIFY pgrst, 'reload schema';

SELECT 'histo_const_ind' AS tbl, COUNT(*)::bigint AS n FROM public.histo_const_ind
UNION ALL SELECT 'indicateurs_financiers', COUNT(*)::bigint FROM public.indicateurs_financiers
ORDER BY tbl;
