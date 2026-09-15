-- Alimenter market_data_cours_historique.titres_echanges
-- depuis public.histo_volume, UNIQUEMENT là où titres_echanges est NULL.
--
-- Exécuter tout le fichier dans Supabase SQL Editor (rôle postgres).
-- Ne touche pas aux volumes déjà renseignés (après 2022-10-24 notamment).

-- ── 0) Schéma source ────────────────────────────────────────────────────────
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'histo_volume'
ORDER BY ordinal_position;

SELECT count(*) AS n_histo_volume FROM public.histo_volume;

SELECT
  count(*) AS n_cours,
  count(titres_echanges) AS n_titres_deja_remplis,
  count(*) FILTER (WHERE titres_echanges IS NULL) AS n_titres_vides
FROM public.market_data_cours_historique;

-- ── 1) Mapping colonnes + UPDATE ────────────────────────────────────────────
DO $$
DECLARE
  col_ticker text;
  col_date   text;
  col_qty    text;
  n_src      bigint;
  n_upd      bigint;
  n_nomatch  bigint;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'histo_volume'
  ) THEN
    RAISE EXCEPTION 'Table public.histo_volume introuvable';
  END IF;

  SELECT c.column_name INTO col_ticker
  FROM information_schema.columns c
  WHERE c.table_schema = 'public'
    AND c.table_name = 'histo_volume'
    AND (
      lower(c.column_name) IN ('ticker', 'titre', 'code_ticker', 'symbole', 'symbol', 'code')
      OR lower(c.column_name) LIKE '%ticker%'
    )
  ORDER BY CASE lower(c.column_name)
    WHEN 'ticker' THEN 0
    WHEN 'titre' THEN 1
    ELSE 2
  END
  LIMIT 1;

  SELECT c.column_name INTO col_date
  FROM information_schema.columns c
  WHERE c.table_schema = 'public'
    AND c.table_name = 'histo_volume'
    AND (
      lower(c.column_name) IN (
        'date', 'date_cours', 'date_volume', 'jour', 'seance', 'date_seance'
      )
      OR lower(c.column_name) LIKE '%date%'
    )
  ORDER BY CASE lower(c.column_name)
    WHEN 'date_cours' THEN 0
    WHEN 'date' THEN 1
    ELSE 2
  END
  LIMIT 1;

  SELECT c.column_name INTO col_qty
  FROM information_schema.columns c
  WHERE c.table_schema = 'public'
    AND c.table_name = 'histo_volume'
    AND (
      lower(c.column_name) IN (
        'titres_echanges', 'titres_échangés', 'volume', 'nb_titres', 'quantite', 'qte'
      )
      OR lower(c.column_name) LIKE '%echang%'
      OR lower(c.column_name) LIKE '%échange%'
      OR c.column_name ILIKE '%titres%'
    )
  ORDER BY CASE
    WHEN lower(c.column_name) LIKE '%echang%' OR lower(c.column_name) LIKE '%échange%' THEN 0
    WHEN c.column_name ILIKE '%titres%' THEN 1
    WHEN lower(c.column_name) = 'volume' THEN 2
    ELSE 3
  END
  LIMIT 1;

  IF col_ticker IS NULL OR col_date IS NULL OR col_qty IS NULL THEN
    RAISE EXCEPTION
      'Colonnes histo_volume non identifiées (ticker=%, date=%, qty=%). Vérifier le SELECT information_schema ci-dessus.',
      col_ticker, col_date, col_qty;
  END IF;

  RAISE NOTICE 'Mapping histo_volume → ticker=%, date=%, qty=%',
    col_ticker, col_date, col_qty;

  EXECUTE format(
    $sql$
    DROP TABLE IF EXISTS tmp_hv;
    CREATE TEMP TABLE tmp_hv AS
    SELECT DISTINCT ON (ticker_n, dt)
      ticker_n,
      dt,
      qty
    FROM (
      SELECT
        upper(trim(%1$I::text)) AS ticker_n,
        CASE
          WHEN trim(%2$I::text) ~ '^\d{4}-\d{2}-\d{2}'
            THEN left(trim(%2$I::text), 10)::date
          WHEN trim(%2$I::text) ~ '^\d{2}/\d{2}/\d{4}'
            THEN to_date(trim(%2$I::text), 'DD/MM/YYYY')
          ELSE %2$I::date
        END AS dt,
        NULLIF(
          replace(replace(trim(%3$I::text), ' ', ''), ',', '.'),
          ''
        )::double precision AS qty
      FROM public.histo_volume
      WHERE %1$I IS NOT NULL
        AND %2$I IS NOT NULL
        AND %3$I IS NOT NULL
    ) s
    WHERE ticker_n <> ''
      AND dt IS NOT NULL
      AND qty IS NOT NULL
      AND qty >= 0
    ORDER BY ticker_n, dt, qty DESC
    $sql$,
    col_ticker, col_date, col_qty
  );

  EXECUTE 'CREATE INDEX ON tmp_hv (ticker_n, dt)';
  EXECUTE 'SELECT count(*) FROM tmp_hv' INTO n_src;
  RAISE NOTICE 'Lignes histo_volume normalisées : %', n_src;

  UPDATE public.market_data_cours_historique AS c
  SET titres_echanges = t.qty
  FROM tmp_hv AS t
  WHERE upper(trim(c.ticker)) = t.ticker_n
    AND c.date_cours::date = t.dt
    AND c.titres_echanges IS NULL
    AND t.qty IS NOT NULL;

  GET DIAGNOSTICS n_upd = ROW_COUNT;
  RAISE NOTICE 'Lignes cours mises à jour (titres_echanges était NULL) : %', n_upd;

  EXECUTE $sql$
    SELECT count(*)
    FROM tmp_hv t
    WHERE NOT EXISTS (
      SELECT 1
      FROM public.market_data_cours_historique c
      WHERE upper(trim(c.ticker)) = t.ticker_n
        AND c.date_cours::date = t.dt
    )
  $sql$ INTO n_nomatch;
  RAISE NOTICE 'Lignes histo_volume sans séance correspondante dans cours : %', n_nomatch;
END $$;

-- ── 2) Contrôles ────────────────────────────────────────────────────────────
SELECT
  count(*) AS n_cours,
  count(titres_echanges) AS n_titres_remplis,
  count(*) FILTER (WHERE titres_echanges IS NULL) AS n_titres_encore_vides,
  min(date_cours) FILTER (WHERE titres_echanges IS NOT NULL) AS min_date_volume,
  max(date_cours) FILTER (WHERE titres_echanges IS NOT NULL) AS max_date_volume
FROM public.market_data_cours_historique;

-- ── 3) Lecture API (optionnel, pour le pipeline Python) ─────────────────────
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT SELECT ON TABLE public.histo_volume TO anon, authenticated;
ALTER TABLE public.histo_volume ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "bvc_rs_anon_read_histo_volume" ON public.histo_volume;
CREATE POLICY "bvc_rs_anon_read_histo_volume"
  ON public.histo_volume
  FOR SELECT
  TO anon, authenticated
  USING (true);

NOTIFY pgrst, 'reload schema';
