-- Schéma attendu pour l'étape 2 (à exécuter dans Supabase SQL Editor)
-- Puis activer RLS + policy INSERT/UPDATE pour la clé anon si écriture via API.

CREATE TABLE IF NOT EXISTS public.features_fondamentales (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker text NOT NULL,
    date_fin date NOT NULL,
    publication_date date,
    sector_type text,
    pe double precision,
    pb double precision,
    ev_ebitda double precision,
    roe double precision,
    marge_ebitda double precision,
    marge_nette double precision,
    ca_growth_yoy double precision,
    resultat_growth_yoy double precision,
    ebitda_growth_yoy double precision,
    gearing double precision,
    piotroski_score double precision,
    score_sante double precision,
    computed_at timestamptz,
    UNIQUE (ticker, date_fin)
);

CREATE TABLE IF NOT EXISTS public.features_techniques (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker text NOT NULL,
    date_cours date NOT NULL,
    ret_1m double precision,
    ret_3m double precision,
    ret_6m double precision,
    ret_12m double precision,
    mom_12_1 double precision,
    rsi_14 double precision,
    dist_ma20 double precision,
    dist_ma50 double precision,
    dist_ma200 double precision,
    vol_20d double precision,
    vol_60d double precision,
    vol_baissiere_20d double precision,
    beta_masi double precision,
    vmq_20j double precision,
    vmq_60j double precision,
    turnover_ratio double precision,
    indicateur_liquidite double precision,
    computed_at timestamptz,
    UNIQUE (ticker, date_cours)
);

CREATE TABLE IF NOT EXISTS public.features_indices (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    date_cours date NOT NULL UNIQUE,
    masi_mom_3m double precision,
    return_dispersion double precision,
    breadth_ma50 double precision,
    vol_universe_mean double precision,
    herfindahl_volume double precision,
    -- Régime discret à seuils (Composante A) — one-hot journalier
    is_bull smallint,
    is_neutral smallint,
    is_sideways smallint,  -- miroir is_neutral (rétrocompat)
    is_bear smallint,
    is_rebalancing_date boolean,
    computed_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_features_fond_ticker ON public.features_fondamentales (ticker);
CREATE INDEX IF NOT EXISTS idx_features_tech_ticker_date ON public.features_techniques (ticker, date_cours);
CREATE INDEX IF NOT EXISTS idx_features_indices_date ON public.features_indices (date_cours);
