-- Alimentation Total_Actif → public.donnees_financieres
-- Source : wafanalytics_v2.comptes_financiers_staging
--
-- Prérequis : exécuter dans Supabase SQL Editor (rôle service / postgres).
-- Idempotent : INSERT si absent, UPDATE si valeur différente (pas de doublon).
--
-- Clé métier : (entreprise_id, indicateur_id, periode_id, type_etat_id, type_compte)

-- ── Constantes ─────────────────────────────────────────────────────────────
-- indicateurs_financiers.code = 'Total_Actif'
-- type_etat = Actifs (bilan)

-- ── 0) Pré-contrôle staging ──────────────────────────────────────────────────
SELECT
  count(*) AS lignes_staging,
  count(DISTINCT titre) AS titres,
  count(DISTINCT date) AS dates
FROM wafanalytics_v2.comptes_financiers_staging s
WHERE lower(replace(trim(s.item_id), ' ', '_')) IN ('total_actif')
  AND nullif(trim(s.montant), '') IS NOT NULL;

-- ── 1) Vue normalisée (staging → clés cibles) ───────────────────────────────
CREATE OR REPLACE VIEW wafanalytics_v2.v_total_actif_a_charger AS
WITH raw AS (
  SELECT
    s.*,
    upper(trim(s.titre)) AS titre_norm,
    upper(trim(s.type_compte)) AS type_compte_norm,
    lower(replace(trim(s.item_id), ' ', '_')) AS item_norm,
    CASE
      WHEN s.date ~ '^\d{4}-\d{2}-\d{2}' THEN s.date::date
      WHEN s.date ~ '^\d{2}/\d{2}/\d{4}' THEN to_date(s.date, 'DD/MM/YYYY')
      WHEN s.date ~ '^\d{2}-\d{2}-\d{4}' THEN to_date(s.date, 'DD-MM-YYYY')
      ELSE NULL
    END AS date_fin,
    NULLIF(trim(replace(replace(s.montant, ' ', ''), ',', '.')), '')::double precision AS montant_num
  FROM wafanalytics_v2.comptes_financiers_staging s
  WHERE lower(replace(trim(s.item_id), ' ', '_')) IN ('total_actif')
    AND nullif(trim(s.montant), '') IS NOT NULL
    AND (
      s.type_etat_financier IS NULL
      OR s.type_etat_financier ILIKE '%actif%'
    )
    AND upper(trim(s.type_compte)) IN ('CONSOLIDE', 'SOCIAL')
)
SELECT DISTINCT ON (e.id, p.id, r.type_compte_norm)
  e.id AS entreprise_id,
  e.ticker,
  p.id AS periode_id,
  p.date_fin,
  r.type_compte_norm AS type_compte,
  CASE
    WHEN r.unite IS NOT NULL AND upper(r.unite) LIKE '%MAD%'
         AND upper(r.unite) NOT LIKE '%K%' THEN r.montant_num / 1000.0
    ELSE r.montant_num
  END AS valeur_kmad
FROM raw r
JOIN public.entreprises e
  ON upper(trim(e.ticker)) = r.titre_norm
  OR upper(trim(e.code_original::text)) = r.titre_norm
JOIN public.periodes p
  ON p.date_fin = r.date_fin
WHERE r.date_fin IS NOT NULL
  AND r.montant_num IS NOT NULL
  AND r.montant_num <> 0
ORDER BY e.id, p.id, r.type_compte_norm, r.date_fin DESC;

-- ── 2) Upsert (INSERT + UPDATE) ─────────────────────────────────────────────
DO $$
DECLARE
  rec record;
  row_id uuid;
  existing_valeur double precision;
  v_insert int := 0;
  v_update int := 0;
  v_skip int := 0;
BEGIN
  FOR rec IN
    SELECT *
    FROM wafanalytics_v2.v_total_actif_a_charger
    ORDER BY ticker, date_fin, type_compte
  LOOP
    row_id := NULL;
    existing_valeur := NULL;

    SELECT df.id, df.valeur
    INTO row_id, existing_valeur
    FROM public.donnees_financieres df
    WHERE df.entreprise_id = rec.entreprise_id
      AND df.indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9'
      AND df.periode_id = rec.periode_id
      AND df.type_etat_id = 'fc14edf7-6caf-4dbf-b7ab-e9204617f079'
      AND df.type_compte = rec.type_compte
    LIMIT 1;

    IF row_id IS NOT NULL THEN
      IF abs(coalesce(existing_valeur, 0) - rec.valeur_kmad) < 0.01 THEN
        v_skip := v_skip + 1;
      ELSE
        UPDATE public.donnees_financieres
        SET valeur = rec.valeur_kmad, updated_at = now()
        WHERE id = row_id;
        v_update := v_update + 1;
      END IF;
    ELSE
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
      ) VALUES (
        rec.entreprise_id,
        '86a4a181-e1d3-4189-9c79-9d72e3d8eae9',
        rec.periode_id,
        'bdd88eb6-09fa-497e-820e-1b7c64711900',
        'fc14edf7-6caf-4dbf-b7ab-e9204617f079',
        rec.valeur_kmad,
        'KMAD',
        rec.type_compte,
        0,
        'VALIDE'
      );
      v_insert := v_insert + 1;
    END IF;
  END LOOP;

  RAISE NOTICE 'Total_Actif — insert=%, update=%, skip=%', v_insert, v_update, v_skip;
END $$;

-- ── 3) Contrôle IAM ───────────────────────────────────────────────────────────
SELECT
  e.ticker,
  p.date_fin,
  df.type_compte,
  df.valeur,
  df.statut_validation
FROM public.donnees_financieres df
JOIN public.entreprises e ON e.id = df.entreprise_id
JOIN public.periodes p ON p.id = df.periode_id
WHERE e.ticker = 'IAM'
  AND df.indicateur_id = '86a4a181-e1d3-4189-9c79-9d72e3d8eae9'
  AND df.type_etat_id = 'fc14edf7-6caf-4dbf-b7ab-e9204617f079'
ORDER BY p.date_fin, df.type_compte;

-- ── 4) Lignes staging non jointes (à corriger manuellement) ───────────────────
SELECT DISTINCT s.titre, s.date, s.type_compte, s.montant
FROM wafanalytics_v2.comptes_financiers_staging s
WHERE lower(replace(trim(s.item_id), ' ', '_')) IN ('total_actif')
  AND nullif(trim(s.montant), '') IS NOT NULL
  AND NOT EXISTS (
    SELECT 1
    FROM wafanalytics_v2.v_total_actif_a_charger v
    JOIN public.entreprises e ON e.id = v.entreprise_id
    WHERE upper(trim(e.ticker)) = upper(trim(s.titre))
       OR upper(trim(e.code_original::text)) = upper(trim(s.titre))
  )
ORDER BY 1, 2
LIMIT 50;
