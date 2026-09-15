-- Migration historique : régime discret HMM dans features_indices
-- Remplace l'ancien vecteur flou mu_* (s'il existait) par is_* (0/1).
-- SUPERSEDE : voir alter_features_indices_threshold_regime.sql (is_neutral + archive HMM).
-- is_rebalancing_date : dernier jour de bourse du mois (inchangé sémantiquement).

ALTER TABLE public.features_indices
    ADD COLUMN IF NOT EXISTS is_bull smallint,
    ADD COLUMN IF NOT EXISTS is_sideways smallint,
    ADD COLUMN IF NOT EXISTS is_bear smallint,
    ADD COLUMN IF NOT EXISTS is_rebalancing_date boolean;

-- Nettoyage éventuel des colonnes floues obsolètes (no-op si absentes)
ALTER TABLE public.features_indices DROP COLUMN IF EXISTS mu_bull;
ALTER TABLE public.features_indices DROP COLUMN IF EXISTS mu_sideways;
ALTER TABLE public.features_indices DROP COLUMN IF EXISTS mu_bear;
