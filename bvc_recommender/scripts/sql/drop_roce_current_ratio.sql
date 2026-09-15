-- Suppression des indicateurs roce et current_ratio
-- Exécuter une fois dans Supabase SQL Editor.

ALTER TABLE public.features_fondamentales
  DROP COLUMN IF EXISTS roce,
  DROP COLUMN IF EXISTS current_ratio;

NOTIFY pgrst, 'reload schema';
