# Diagnostic MASI — Supabase `tlrborylhfwxrrryyism`

**Dernière re-vérification** : 2026-06-09 (probe RLS + service_role + loader)  
**Méthode** : API REST anon, clé `service_role` (bypass RLS), pagination complète, comparaison UUID utilisateur.

---

## Verdict (cause racine)

| Question | Réponse |
|----------|---------|
| **MASI 2010+ visible via REST anon ?** | **Non** — 741 lignes, min `2022-11-21` |
| **MASI 2010+ visible via service_role ?** | **Non** — mêmes 741 lignes (RLS **écarté**) |
| **UUID screenshot (`9c49e61a-…`) trouvé en base ?** | **Non** — ni anon ni service_role |
| **RLS filtre les lignes anciennes ?** | **Peu probable sur ce projet** — service_role = anon (15 526 / 741 MASI) |
| **Deux jeux de données (UUID vs integer) ?** | **Non** — 100 % UUID côté API |
| **Bug pagination / filtre loader ?** | **Non** — count exact = fetch paginé |
| **Contradiction SQL Editor utilisateur ?** | **À confirmer** — voir section ci-dessous |

### Cause racine la plus probable

Les données MASI 2010+ **ne sont pas présentes** dans le projet `tlrborylhfwxrrryyism` tel qu’accessible via l’API (anon **et** service_role).  
Si le SQL Editor affiche 2010-01-04 avec `valeur_index` ~10 312, vérifier impérativement le **project-ref** en haut à gauche du dashboard et exécuter les requêtes de contrôle ci-dessous **dans ce même projet**.

---

## Preuves collectées (2026-06-09)

### 1. Probe REST anon (`probe_rls_masi.py`)

Fichier : `bvc_recommender/reports/masi_rls_probe.json`

| Métrique | Valeur anon |
|----------|-------------|
| Total indices | 15 526 |
| MASI | 741 |
| MASI `date_index < 2022-01-01` | **0** |
| MASI `date_index >= 2010-01-01` | 741 (= toutes les lignes visibles) |
| Première date MASI (ordre asc) | **2022-11-21** |
| Type `id` | 741 UUID, 0 integer |
| UUID utilisateur `9c49e61a-e4e5-40a4-b58d-80ba95b869ea` | **0 ligne** |

### 2. Cross-check service_role (bypass RLS)

Même projet, même table — résultats **identiques** à anon :

| Métrique | anon | service_role |
|----------|------|--------------|
| Total | 15 526 | 15 526 |
| MASI | 741 | 741 |
| MASI avant 2022 | 0 | 0 |
| UUID `9c49e61a-…` | absent | absent |
| MASI 2010-01-04 | absent | absent |

→ **RLS ne peut pas expliquer l’écart** si service_role ne voit pas non plus les lignes 2010.

### 3. Loader / step1 — pas de filtre erroné

`fetch_table` dans `loader.py` charge `market_data_indices_historique` **sans** filtre date ni `code_index`.  
Pagination : 15 526/15 526 (ordre `id` et `date_index`). **Aucun bug code identifié.**

---

## Contradiction avec le screenshot utilisateur

L’utilisateur rapporte dans le SQL Editor :

```sql
SELECT * FROM market_data_indices_historique
WHERE code_index='MASI' ORDER BY date_index;
-- Résultats : 2010-01-04, 2010-01-05… valeur_index ~10312, id UUID 9c49e61a-…
```

**Nos sondes sur `tlrborylhfwxrrryyism` ne retrouvent pas ces lignes** (même avec service_role).

| Hypothèse | Action de vérification |
|-----------|------------------------|
| **Mauvais projet Supabase** | Confirmer ref `tlrborylhfwxrrryyism` dans l’URL du dashboard |
| **Autre table / schéma** | Vérifier `public.market_data_indices_historique` explicitement |
| **Données insérées localement non persistées** | Re-exécuter COUNT/MIN après refresh |
| **RLS + rôle SQL Editor restreint** | Peu probable — postgres bypass RLS ; service_role confirme l’absence |
| **Projet alternatif `pungyebagtntbocoewir`** | Ancien .env vide — clé anon absente |

**SQL de contrôle à coller dans le dashboard** (section 5 de `verify_masi_user.py`) :

```sql
SELECT COUNT(*) AS masi_lignes, MIN(date_index) AS masi_min, MAX(date_index) AS masi_max
FROM public.market_data_indices_historique
WHERE code_index = 'MASI';

SELECT id, date_index, valeur_index
FROM public.market_data_indices_historique
WHERE id = '9c49e61a-e4e5-40a4-b58d-80ba95b869ea';

SELECT COUNT(*) AS masi_avant_2022
FROM public.market_data_indices_historique
WHERE code_index = 'MASI' AND date_index < '2022-01-01';
```

**Si SQL Editor montre MIN ≤ 2010-01-04 mais REST anon montre 2022-11-21** → appliquer `fix_rls_indices.sql` (RLS restrictif sur anon uniquement — cas rare vu service_role).

**Si SQL Editor montre aussi MIN = 2022-11-21** → importer l’historique MASI 2010+ (`import_masi_history.py`).

---

## Ce qui existe via API (état actuel)

### MASI (`code_index = 'MASI'`)

| Métrique | Valeur |
|----------|--------|
| Lignes | 741 |
| Date min | 2022-11-21 |
| Date max | 2025-11-17 |
| `scraped_at` | 2025-11-19 (lot unique) |
| Type `id` | UUID uniquement |

### Parquet local step1

Identique à Supabase REST : 15 526 lignes indices, MASI 741 lignes, 2022-11-21 → 2025-11-17.

---

## Impact `beta_masi` (inchangé — step2 non relancé)

- Couverture `beta_masi` : **7,3 %** (`step2_validation_report.json`)
- Dernières dates ATW/IAM/BCP : `beta_masi = NaN` (MASI s’arrête 2025-11-17)
- **step1 / step2 non relancés** — données source inchangées côté API

---

## Actions utilisateur

### A. Si RLS confirmé (SQL Editor >> REST anon, service_role voit tout)

Exécuter dans SQL Editor :

`bvc_recommender/scripts/sql/fix_rls_indices.sql`

Puis :

```bash
python -m bvc_recommender.scripts.probe_rls_masi
python -m bvc_recommender.scripts.run_step1
python -m bvc_recommender.scripts.run_step2 --tickers ATW IAM BCP
```

Critères succès : `masi_min <= 2010-01-04`, `beta_masi` couverture >> 7,3 %.

### B. Si données absentes (SQL Editor = REST = 741 lignes)

Importer CSV historique :

```bash
python -m bvc_recommender.scripts.import_masi_history --file chemin/masi_2010.csv
```

Template : `bvc_recommender/data/templates/masi_history_template.csv`

### C. Option Postgres direct (contournement)

Décommenter `DATABASE_URL` dans `.env` (Dashboard → Settings → Database), ou utiliser `SUPABASE_SERVICE_ROLE_KEY` hors dépôt pour imports batch.

---

## Fichiers de référence

| Fichier | Rôle |
|---------|------|
| `bvc_recommender/scripts/probe_rls_masi.py` | Probe anon + inférence RLS |
| `bvc_recommender/reports/masi_rls_probe.json` | Résultats bruts probe |
| `bvc_recommender/scripts/sql/fix_rls_indices.sql` | Policies SELECT anon/authenticated |
| `bvc_recommender/scripts/masi_diagnostic.py` | Diagnostic complet pagination |
| `bvc_recommender/reports/masi_diagnostic_raw.json` | Données brutes diagnostic |

---

## Actions effectuées (2026-06-09)

| Action | Résultat |
|--------|----------|
| `probe_rls_masi` | anon : 741 MASI, UUID user absent, 0 pré-2022 |
| Probe service_role | Identique anon → **RLS écarté** pour ce projet |
| Analyse id UUID vs int | 100 % UUID, pas de split datasets |
| Revue `loader.py` / `fetch_table` | Pas de filtre erroné |
| `fix_rls_indices.sql` | Créé — à exécuter si SQL Editor ≠ REST |
| Re-run step1 / step2 | **Non** — données API inchangées |
