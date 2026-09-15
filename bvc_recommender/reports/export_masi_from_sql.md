# Export MASI depuis SQL Editor → import pipeline

Utilisez ce guide si SQL Editor affiche des lignes MASI 2010+ mais l’API REST (`tlrborylhfwxrrryyism`) n’en voit que 741 (depuis 2022-11-21) — ou si vous voulez simplement alimenter le pipeline sans dépendre de l’API.

---

## 0. Vérifier que vous êtes sur le bon projet Supabase

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Barre d’adresse du Dashboard Supabase                                  │
│                                                                         │
│  https://supabase.com/dashboard/project/tlrborylhfwxrrryyism/...        │
│                                        ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲              │
│                                        project-ref attendu par .env     │
└─────────────────────────────────────────────────────────────────────────┘
```

| Où regarder | Valeur attendue (ce dépôt) |
|-------------|------------------------------|
| URL dashboard après `/project/` | `tlrborylhfwxrrryyism` |
| Settings → API → Project URL | `https://tlrborylhfwxrrryyism.supabase.co` |
| JWT anon décodé (`ref`) | `tlrborylhfwxrrryyism` |

Si le ref dans l’URL ≠ `tlrborylhfwxrrryyism`, le SQL Editor et `.env` ne pointent **pas** vers le même projet — mettez à jour `.env` ou exportez depuis le bon projet.

---

## 1. Requête SQL (SQL Editor)

```sql
SELECT
  code_index,
  nom_index,
  date_index,
  valeur_index,
  valeur_haut,
  valeur_bas,
  variation_veille,
  variation_annee,
  capitalisation_marchande,
  drupal_id,
  scraped_at
FROM public.market_data_indices_historique
WHERE code_index = 'MASI'
  AND date_index >= '2010-01-01'
ORDER BY date_index;
```

Contrôle rapide :

```sql
SELECT MIN(date_index), MAX(date_index), COUNT(*)
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';
```

---

## 2. Export CSV depuis SQL Editor

1. Exécuter la requête ci-dessus.
2. Onglet **Results** → bouton **Download** (ou **Export CSV** selon la version).
3. Enregistrer sous `masi_historique.csv`.

**Alternative COPY (psql / connexion directe)** — si vous avez `DATABASE_URL` :

```sql
\copy (
  SELECT code_index, nom_index, date_index, valeur_index,
         valeur_haut, valeur_bas, variation_veille, variation_annee,
         capitalisation_marchande, drupal_id, scraped_at
  FROM public.market_data_indices_historique
  WHERE code_index = 'MASI' AND date_index >= '2010-01-01'
  ORDER BY date_index
) TO 'masi_historique.csv' WITH (FORMAT CSV, HEADER true);
```

Colonnes minimales pour l’import : `date_index` + `valeur_index` (le script complète `code_index = MASI`).

---

## 3. Import via le pipeline

Depuis la racine du projet :

```bash
# Simulation (aucune écriture)
py -3 -m bvc_recommender.scripts.import_masi_history --file chemin/vers/masi_historique.csv --dry-run

# Import réel (upsert sur code_index + date_index)
py -3 -m bvc_recommender.scripts.import_masi_history --file chemin/vers/masi_historique.csv
```

Si RLS bloque l’écriture avec la clé anon :

1. **Option A** — policy SQL (recommandé pour anon) : voir message d’erreur de `import_masi_history.py`.
2. **Option B** — clé `service_role` : Dashboard → API → service_role (secret), à utiliser hors dépôt ; le script d’import actuel passe par la clé anon (`get_supabase_client`). Pour un import batch avec service_role, exécutez l’upsert depuis SQL Editor ou étendez le script pour accepter `SUPABASE_SERVICE_ROLE_KEY`.

---

## 4. Vérification post-import

```bash
py -3 -m bvc_recommender.scripts.probe_rls_masi
```

Attendu : `masi_gte_2010` > 741, `date_min` ≤ 2010-01-04.
