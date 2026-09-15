-- Correction outliers market_data_cours_historique
-- Genere par diagnostics/fix_cours_outliers_sna_slf_tma_rds.py
-- A executer dans le SQL Editor Supabase (role postgres / service_role)

BEGIN;

-- 1) SNA : supprimer tout l'historique
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'SNA';

-- 2) SLF : supprimer cours autour de 26 (echelle basse)
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'SLF'
  AND prix_cloture <= 40.0;

-- 3) TMA : supprimer spikes isoles
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'TMA'
  AND date_cours::date IN ('2010-01-04', '2022-12-31', '2023-06-30', '2023-09-30', '2023-12-31', '2024-03-31');

-- 4) RDS : supprimer haute echelle aberrante en 2018
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'RDS'
  AND EXTRACT(YEAR FROM date_cours::date) = 2018
  AND prix_cloture >= 50.0;

COMMIT;

-- Pass 2 (residus)
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker)='TMA'
  AND date_cours::date BETWEEN '2022-10-01' AND '2022-10-23'
  AND prix_cloture <= 400.0;

DELETE FROM public.market_data_cours_historique
WHERE upper(ticker)='RDS'
  AND date_cours::date < '2024-05-01'
  AND prix_cloture >= 100.0;
