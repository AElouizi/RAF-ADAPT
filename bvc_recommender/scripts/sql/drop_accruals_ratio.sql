-- Suppression de accruals_ratio (retiré du pipeline fondamental)
-- Exécuter une fois dans Supabase SQL Editor.

ALTER TABLE public.features_fondamentales
  DROP COLUMN IF EXISTS accruals_ratio;

NOTIFY pgrst, 'reload schema';
