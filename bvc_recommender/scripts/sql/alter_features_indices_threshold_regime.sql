-- Migration : régime discret à seuils (remplace HMM en production)
-- Ajoute is_neutral ; conserve is_sideways en miroir pour rétrocompat.
-- Archive HMM : table features_indices_hmm_archive (snapshot one-hot HMM).

ALTER TABLE public.features_indices
    ADD COLUMN IF NOT EXISTS is_bull smallint,
    ADD COLUMN IF NOT EXISTS is_sideways smallint,
    ADD COLUMN IF NOT EXISTS is_neutral smallint,
    ADD COLUMN IF NOT EXISTS is_bear smallint,
    ADD COLUMN IF NOT EXISTS is_rebalancing_date boolean;

-- Si is_neutral encore vide mais is_sideways présent (transition)
UPDATE public.features_indices
SET is_neutral = is_sideways
WHERE is_neutral IS NULL AND is_sideways IS NOT NULL;

CREATE TABLE IF NOT EXISTS public.features_indices_hmm_archive (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    date_cours date NOT NULL UNIQUE,
    masi_mom_3m double precision,
    return_dispersion double precision,
    breadth_ma50 double precision,
    is_bull smallint,
    is_sideways smallint,
    is_bear smallint,
    is_rebalancing_date boolean,
    archived_at timestamptz DEFAULT now(),
    note text DEFAULT 'GaussianHMM 3-states snapshot before threshold regime'
);

CREATE INDEX IF NOT EXISTS idx_features_indices_hmm_archive_date
    ON public.features_indices_hmm_archive (date_cours);
