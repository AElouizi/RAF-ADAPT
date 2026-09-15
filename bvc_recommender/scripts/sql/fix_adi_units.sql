-- ADI : correction d'unité (KMAD) — CONSOLIDE semestriel 30/06/2015 → 30/06/2022
-- Actifs + Passifs : × 1 000
-- CPC RNPG seul : ÷ 1 000 (CA/RN/rex déjà en KMAD)

-- 1) Bilan Actifs & Passifs
UPDATE public.donnees_financieres df
SET valeur = df.valeur * 1000, updated_at = now()
FROM public.periodes p
WHERE df.entreprise_id = '93bf92ba-3730-496f-8a8a-85700819df47'
  AND df.type_compte = 'CONSOLIDE'
  AND df.type_etat_id IN (
    'fc14edf7-6caf-4dbf-b7ab-e9204617f079',  -- Actifs
    'e8a7cb89-643f-46fd-81f1-79f17030bdc9'   -- Passifs
  )
  AND df.periode_id = p.id
  AND p.type_periode = 'SEMESTRIELLE'
  AND p.date_fin >= '2015-06-30'
  AND p.date_fin <= '2022-06-30';

-- 2) CPC — RNPG sur-échelonné
UPDATE public.donnees_financieres df
SET valeur = df.valeur / 1000, updated_at = now()
FROM public.periodes p
WHERE df.entreprise_id = '93bf92ba-3730-496f-8a8a-85700819df47'
  AND df.type_compte = 'CONSOLIDE'
  AND df.type_etat_id = '1202c838-e080-4d87-97c5-314169a582a0'
  AND df.indicateur_id = 'f0a04583-1206-4d30-b1b3-bc988fbd451c'
  AND df.periode_id = p.id
  AND p.type_periode = 'SEMESTRIELLE'
  AND p.date_fin >= '2015-06-30'
  AND p.date_fin <= '2022-06-30';

-- Contrôles
SELECT p.date_fin, df.valeur AS capitaux_propres
FROM public.donnees_financieres df
JOIN public.periodes p ON p.id = df.periode_id
WHERE df.entreprise_id = '93bf92ba-3730-496f-8a8a-85700819df47'
  AND df.indicateur_id = 'b9c2dc9f-e1ba-448b-a84b-aefa42e620ea'
  AND df.type_etat_id = 'e8a7cb89-643f-46fd-81f1-79f17030bdc9'
  AND df.type_compte = 'CONSOLIDE'
  AND p.type_periode = 'SEMESTRIELLE'
  AND p.date_fin IN ('2022-06-30', '2023-06-30')
ORDER BY p.date_fin;

SELECT p.date_fin, df.valeur AS rnpg
FROM public.donnees_financieres df
JOIN public.periodes p ON p.id = df.periode_id
WHERE df.entreprise_id = '93bf92ba-3730-496f-8a8a-85700819df47'
  AND df.indicateur_id = 'f0a04583-1206-4d30-b1b3-bc988fbd451c'
  AND df.type_etat_id = '1202c838-e080-4d87-97c5-314169a582a0'
  AND df.type_compte = 'CONSOLIDE'
  AND p.type_periode = 'SEMESTRIELLE'
  AND p.date_fin IN ('2022-06-30', '2023-06-30')
ORDER BY p.date_fin;
