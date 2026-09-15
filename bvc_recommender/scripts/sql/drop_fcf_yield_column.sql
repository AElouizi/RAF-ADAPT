-- Suppression de l'indicateur fcf_yield (remplacé par ev_ebitda)
-- Exécuter une fois dans Supabase SQL Editor.

ALTER TABLE public.features_fondamentales
  DROP COLUMN IF EXISTS fcf_yield;

NOTIFY pgrst, 'reload schema';

-- Contrôle : la colonne ne doit plus exister
SELECT column_name
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'features_fondamentales'
ORDER BY ordinal_position;
