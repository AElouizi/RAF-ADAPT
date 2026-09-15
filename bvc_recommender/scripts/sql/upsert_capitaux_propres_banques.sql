-- Capitaux propres (Passifs, CONSOLIDE) — BCP, BOA, SBM, ATW
-- Clôtures annuelles 2015–2021 (valeurs en milliers MAD / KMAD)
-- Exécuter dans Supabase SQL Editor si l'import Python n'est pas disponible.

DO $$
DECLARE
  rec record;
  row_id uuid;
BEGIN
  FOR rec IN
    SELECT * FROM (VALUES
      ('BCP', 'ad996576-6fd2-4372-adf7-853773a6e546'::uuid, '763027be-df94-491b-bfd8-749c80e09cca'::uuid, 38839564::double precision),
      ('BCP', 'ad996576-6fd2-4372-adf7-853773a6e546'::uuid, 'c682a56a-1cad-407c-a312-5caf4f2a627d'::uuid, 41370924::double precision),
      ('BCP', 'ad996576-6fd2-4372-adf7-853773a6e546'::uuid, '04fa9b3d-edd4-402c-a646-2079b99a6d11'::uuid, 43483573::double precision),
      ('BOA', 'eb78b221-7006-4de6-9572-f20a7894f565'::uuid, 'c682a56a-1cad-407c-a312-5caf4f2a627d'::uuid, 23582687::double precision),
      ('BOA', 'eb78b221-7006-4de6-9572-f20a7894f565'::uuid, '04fa9b3d-edd4-402c-a646-2079b99a6d11'::uuid, 24684424::double precision),
      ('BOA', 'eb78b221-7006-4de6-9572-f20a7894f565'::uuid, 'dbcb2063-fbc9-49f9-aa63-27cbec8282ca'::uuid, 23841511::double precision),
      ('BOA', 'eb78b221-7006-4de6-9572-f20a7894f565'::uuid, 'ff4f2660-d300-4354-8529-ce6565dd4390'::uuid, 29435162::double precision),
      ('BOA', 'eb78b221-7006-4de6-9572-f20a7894f565'::uuid, 'a103de89-fb59-4e2d-a3a7-3926bc507b34'::uuid, 29943306::double precision),
      ('BOA', 'eb78b221-7006-4de6-9572-f20a7894f565'::uuid, '58243760-2eea-408c-bd49-063de2bb38a3'::uuid, 31390520::double precision),
      ('SBM', '23ac7cc4-5784-43f5-8b8b-2e5acfe82e57'::uuid, '763027be-df94-491b-bfd8-749c80e09cca'::uuid, 1587653::double precision),
      ('SBM', '23ac7cc4-5784-43f5-8b8b-2e5acfe82e57'::uuid, 'c682a56a-1cad-407c-a312-5caf4f2a627d'::uuid, 1660155::double precision),
      ('SBM', '23ac7cc4-5784-43f5-8b8b-2e5acfe82e57'::uuid, '04fa9b3d-edd4-402c-a646-2079b99a6d11'::uuid, 1741968::double precision),
      ('SBM', '23ac7cc4-5784-43f5-8b8b-2e5acfe82e57'::uuid, 'dbcb2063-fbc9-49f9-aa63-27cbec8282ca'::uuid, 1673601::double precision),
      ('SBM', '23ac7cc4-5784-43f5-8b8b-2e5acfe82e57'::uuid, 'ff4f2660-d300-4354-8529-ce6565dd4390'::uuid, 1686386::double precision),
      ('SBM', '23ac7cc4-5784-43f5-8b8b-2e5acfe82e57'::uuid, 'a103de89-fb59-4e2d-a3a7-3926bc507b34'::uuid, 1537640::double precision),
      ('ATW', '245923cb-9950-45df-9cb8-c7a08c849108'::uuid, '763027be-df94-491b-bfd8-749c80e09cca'::uuid, 40401978::double precision),
      ('ATW', '245923cb-9950-45df-9cb8-c7a08c849108'::uuid, 'c682a56a-1cad-407c-a312-5caf4f2a627d'::uuid, 47411083::double precision),
      ('ATW', '245923cb-9950-45df-9cb8-c7a08c849108'::uuid, '04fa9b3d-edd4-402c-a646-2079b99a6d11'::uuid, 46058720::double precision)
    ) AS v(ticker, entreprise_id, periode_id, valeur)
  LOOP
    SELECT id INTO row_id
    FROM public.donnees_financieres
    WHERE entreprise_id = rec.entreprise_id
      AND indicateur_id = 'b9c2dc9f-e1ba-448b-a84b-aefa42e620ea'
      AND periode_id = rec.periode_id
      AND type_etat_id = 'e8a7cb89-643f-46fd-81f1-79f17030bdc9'
      AND type_compte = 'CONSOLIDE'
    LIMIT 1;

    IF row_id IS NOT NULL THEN
      UPDATE public.donnees_financieres
      SET valeur = rec.valeur, updated_at = now()
      WHERE id = row_id;
      RAISE NOTICE 'UPDATE % periode=% valeur=%', rec.ticker, rec.periode_id, rec.valeur;
    ELSE
      INSERT INTO public.donnees_financieres (
        entreprise_id, indicateur_id, periode_id, source_id, type_etat_id,
        valeur, unite, type_compte, score_qualite, statut_validation
      ) VALUES (
        rec.entreprise_id,
        'b9c2dc9f-e1ba-448b-a84b-aefa42e620ea',
        rec.periode_id,
        'bdd88eb6-09fa-497e-820e-1b7c64711900',
        'e8a7cb89-643f-46fd-81f1-79f17030bdc9',
        rec.valeur,
        'KMAD',
        'CONSOLIDE',
        0,
        'VALIDE'
      );
      RAISE NOTICE 'INSERT % periode=% valeur=%', rec.ticker, rec.periode_id, rec.valeur;
    END IF;
  END LOOP;
END $$;

-- Contrôle rapide
SELECT e.ticker, p.date_fin, df.valeur
FROM public.donnees_financieres df
JOIN public.entreprises e ON e.id = df.entreprise_id
JOIN public.periodes p ON p.id = df.periode_id
WHERE e.ticker IN ('BCP', 'BOA', 'SBM', 'ATW')
  AND df.indicateur_id = 'b9c2dc9f-e1ba-448b-a84b-aefa42e620ea'
  AND df.type_etat_id = 'e8a7cb89-643f-46fd-81f1-79f17030bdc9'
  AND df.type_compte = 'CONSOLIDE'
  AND p.type_periode = 'ANNUELLE'
  AND p.date_fin BETWEEN '2015-12-31' AND '2021-12-31'
ORDER BY e.ticker, p.date_fin;
