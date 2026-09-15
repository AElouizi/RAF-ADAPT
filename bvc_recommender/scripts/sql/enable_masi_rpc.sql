-- Contournement RLS : fonction SECURITY DEFINER lisible par anon
-- Exécuter UNE FOIS dans Supabase SQL Editor (Run sur tout le fichier)

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

REVOKE ALL ON FUNCTION public.fetch_masi_historique() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.fetch_masi_historique() TO anon, authenticated, service_role;

-- Vérification (postgres)
SELECT COUNT(*) AS masi_total,
       MIN(date_index) AS masi_min,
       MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';
