-- Migration Total_Actif : histo_total_actif → public.donnees_financieres
-- 1) Vérifier l'indicateur dans indicateurs_financiers
-- 2) Supprimer les anciennes lignes Total_Actif dans donnees_financieres
-- 3) Réinsérer depuis histo_total_actif (sans doublons)
--
-- Exécuter dans Supabase SQL Editor (rôle postgres / service).

-- ── Constantes ───────────────────────────────────────────────────────────────
-- indicateurs_financiers.code = 'Total_Actif'
-- type_etat = Actifs (bilan)

-- ── 0) Contrôle indicateur ───────────────────────────────────────────────────
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM public.indicateurs_financiers
    WHERE id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9'
      AND code = 'Total_Actif'
  ) THEN
    RAISE EXCEPTION 'Indicateur Total_Actif (86a4a181-…) introuvable dans indicateurs_financiers';
  END IF;
  RAISE NOTICE 'OK — indicateur Total_Actif présent';
END $$;

-- ── 1) Volumes avant migration ───────────────────────────────────────────────
SELECT 'donnees_financieres (avant)' AS source, count(*) AS n
FROM public.donnees_financieres
WHERE indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9'
UNION ALL
SELECT 'histo_total_actif', count(*)
FROM public.histo_total_actif;

-- ── 2) Vue normalisée (déduplication source) ─────────────────────────────────
CREATE OR REPLACE VIEW public.v_histo_total_actif_norm AS
WITH raw AS (
  SELECT
    upper(trim(h.ticker)) AS ticker_norm,
    upper(trim(h.type_compte)) AS type_compte_norm,
    lower(replace(trim(coalesce(h.item_id, 'Total_Actif')), ' ', '_')) AS item_norm,
    CASE
      WHEN h.date IS NULL THEN NULL
      WHEN pg_typeof(h.date)::text = 'date' THEN h.date::date
      WHEN h.date::text ~ '^\d{4}-\d{2}-\d{2}' THEN left(h.date::text, 10)::date
      WHEN h.date::text ~ '^\d{2}/\d{2}/\d{4}' THEN to_date(h.date::text, 'DD/MM/YYYY')
      ELSE h.date::date
    END AS date_fin,
    NULLIF(trim(replace(replace(h.montant::text, ' ', ''), ',', '.')), '')::double precision AS montant_num
  FROM public.histo_total_actif h
  WHERE nullif(trim(h.montant::text), '') IS NOT NULL
    AND upper(trim(h.type_compte)) IN ('CONSOLIDE', 'SOCIAL')
    AND lower(replace(trim(coalesce(h.item_id, 'Total_Actif')), ' ', '_'))
        IN ('total_actif', 'total_actif_buk1', 'total_actif_buk2', 'total_actif_buk3')
)
SELECT DISTINCT ON (e.id, p.id, r.type_compte_norm)
  e.id AS entreprise_id,
  e.ticker,
  p.id AS periode_id,
  p.date_fin,
  r.type_compte_norm AS type_compte,
  r.montant_num AS valeur_kmad
FROM raw r
JOIN public.entreprises e
  ON upper(trim(e.ticker)) = r.ticker_norm
JOIN public.periodes p
  ON p.date_fin = r.date_fin
WHERE r.date_fin IS NOT NULL
  AND r.montant_num IS NOT NULL
  AND r.montant_num <> 0
ORDER BY e.id, p.id, r.type_compte_norm, r.date_fin DESC;

-- Lignes histo non mappées (ticker ou date absent de periodes/entreprises)
SELECT h.ticker, h.date, h.type_compte, h.montant, h.item_id
FROM public.histo_total_actif h
WHERE NOT EXISTS (
  SELECT 1
  FROM public.v_histo_total_actif_norm v
  JOIN public.entreprises e ON e.id = v.entreprise_id
  WHERE upper(trim(e.ticker)) = upper(trim(h.ticker))
    AND v.date_fin = CASE
      WHEN pg_typeof(h.date)::text = 'date' THEN h.date::date
      ELSE left(h.date::text, 10)::date
    END
    AND v.type_compte = upper(trim(h.type_compte))
)
ORDER BY 1, 2
LIMIT 100;

-- ── 3) Suppression anciennes lignes Total_Actif ────────────────────────────────
DELETE FROM public.donnees_financieres
WHERE indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9';

-- ── 4) Insertion (clé métier = entreprise + période + type_etat + type_compte) ─
INSERT INTO public.donnees_financieres (
  entreprise_id,
  indicateur_id,
  periode_id,
  source_id,
  type_etat_id,
  valeur,
  unite,
  type_compte,
  score_qualite,
  statut_validation
)
SELECT
  v.entreprise_id,
  '86a4a181-e1d3-4189-9c79-9d72e3d8eae9',
  v.periode_id,
  'bdd88eb6-09fa-497e-820e-1b7c64711900',
  'fc14edf7-6caf-4dbf-b7ab-e9204617f079',
  v.valeur_kmad,
  'KMAD',
  v.type_compte,
  0,
  'VALIDE'
FROM public.v_histo_total_actif_norm v
WHERE NOT EXISTS (
  SELECT 1
  FROM public.donnees_financieres df
  WHERE df.entreprise_id = v.entreprise_id
    AND df.indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9'
    AND df.periode_id = v.periode_id
    AND df.type_etat_id = 'fc14edf7-6caf-4dbf-b7ab-e9204617f079'
    AND df.type_compte = v.type_compte
);

-- ── 5) Contrôles post-migration ──────────────────────────────────────────────
SELECT 'donnees_financieres (après)' AS source, count(*) AS n
FROM public.donnees_financieres
WHERE indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9';

SELECT
  e.ticker,
  count(*) AS periodes,
  min(p.date_fin) AS depuis,
  max(p.date_fin) AS jusqu_a
FROM public.donnees_financieres df
JOIN public.entreprises e ON e.id = df.entreprise_id
JOIN public.periodes p ON p.id = df.periode_id
WHERE df.indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9'
  AND df.type_compte = 'CONSOLIDE'
  AND df.statut_validation = 'VALIDE'
GROUP BY e.ticker
ORDER BY e.ticker;

-- IAM détail
SELECT
  e.ticker,
  p.date_fin,
  p.type_periode,
  df.type_compte,
  df.valeur
FROM public.donnees_financieres df
JOIN public.entreprises e ON e.id = df.entreprise_id
JOIN public.periodes p ON p.id = df.periode_id
WHERE e.ticker = 'IAM'
  AND df.indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9'
ORDER BY p.date_fin, df.type_compte;
