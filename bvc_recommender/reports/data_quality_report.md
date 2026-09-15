# Rapport qualité des données — Étape 1

- **Projet** : BVC Recommender — Étape 1
- **Généré le** : 2026-06-10 14:58 UTC

## Synthèse par table

| Table | Lignes | Colonnes | Date min | Date max | Statut |
|---|---:|---:|---|---|---|
| raw/fondamentaux_rs | 14,040 | 5 | 2014-06-30 | 2025-12-31 | OK |
| raw/donnees_financieres | 360,133 | 18 | — | — | OK |
| raw/market_data_cours_historique | 258,086 | 17 | 2010-01-04 | 2026-05-18 | WARNING |
| raw/market_data_indices_historique | 18,886 | 12 | 2010-01-04 | 2025-11-17 | WARNING |
| clean/fondamentaux_rs | 14,040 | 7 | 2014-06-30 | 2025-12-31 | OK |
| clean/donnees_financieres | 138,010 | 18 | — | — | OK |
| clean/market_data_cours_historique | 258,086 | 17 | 2010-01-04 | 2026-05-18 | WARNING |
| clean/market_data_indices_historique | 18,886 | 12 | 2010-01-04 | 2025-11-17 | WARNING |

## Valeurs manquantes (% par colonne)

### raw/fondamentaux_rs

- `montant` : 20.3 %
- `id` : 0.0 %
- `ticker` : 0.0 %
- `item_id` : 0.0 %
- `date` : 0.0 %

### raw/donnees_financieres

- `commentaire_validation` : 100.0 %
- `valide_par` : 100.0 %
- `valide_le` : 100.0 %
- `methode_calcul` : 100.0 %
- `donnees_brutes` : 100.0 %
- `id` : 0.0 %
- `entreprise_id` : 0.0 %
- `indicateur_id` : 0.0 %
- `periode_id` : 0.0 %
- `source_id` : 0.0 %
- `type_etat_id` : 0.0 %
- `valeur` : 0.0 %
- `unite` : 0.0 %
- `type_compte` : 0.0 %
- `score_qualite` : 0.0 %

### raw/market_data_cours_historique

- `drupal_id` : 99.4 %
- `prix_ouverture` : 80.8 %
- `prix_haut` : 80.8 %
- `prix_bas` : 80.8 %
- `volume` : 80.8 %
- `titres_echanges` : 80.8 %
- `nombre_transactions` : 76.4 %
- `capitalisation` : 75.5 %
- `prix_ajuste` : 75.5 %
- `ratio_consolide` : 75.5 %
- `prix_cloture` : 0.3 %
- `id` : 0.0 %
- `instrument_id` : 0.0 %
- `ticker` : 0.0 %
- `date_cours` : 0.0 %

### raw/market_data_indices_historique

- `nom_index` : 100.0 %
- `valeur_haut` : 100.0 %
- `valeur_bas` : 100.0 %
- `variation_veille` : 100.0 %
- `variation_annee` : 100.0 %
- `capitalisation_marchande` : 100.0 %
- `drupal_id` : 17.8 %
- `id` : 0.0 %
- `code_index` : 0.0 %
- `date_index` : 0.0 %
- `valeur_index` : 0.0 %
- `scraped_at` : 0.0 %

### clean/fondamentaux_rs

- `montant` : 14.0 %
- `id` : 0.0 %
- `ticker` : 0.0 %
- `item_id` : 0.0 %
- `date` : 0.0 %
- `date_fin` : 0.0 %
- `publication_date` : 0.0 %

### clean/donnees_financieres

- `commentaire_validation` : 100.0 %
- `valide_par` : 100.0 %
- `valide_le` : 100.0 %
- `methode_calcul` : 100.0 %
- `donnees_brutes` : 100.0 %
- `id` : 0.0 %
- `entreprise_id` : 0.0 %
- `indicateur_id` : 0.0 %
- `periode_id` : 0.0 %
- `source_id` : 0.0 %
- `type_etat_id` : 0.0 %
- `valeur` : 0.0 %
- `unite` : 0.0 %
- `type_compte` : 0.0 %
- `score_qualite` : 0.0 %

### clean/market_data_cours_historique

- `drupal_id` : 99.4 %
- `nombre_transactions` : 76.4 %
- `ratio_consolide` : 75.5 %
- `prix_ouverture` : 75.5 %
- `prix_haut` : 75.5 %
- `prix_bas` : 75.5 %
- `volume` : 75.5 %
- `titres_echanges` : 75.5 %
- `capitalisation` : 75.3 %
- `prix_ajuste` : 75.3 %
- `id` : 0.0 %
- `instrument_id` : 0.0 %
- `ticker` : 0.0 %
- `date_cours` : 0.0 %
- `prix_cloture` : 0.0 %

### clean/market_data_indices_historique

- `nom_index` : 100.0 %
- `valeur_haut` : 100.0 %
- `valeur_bas` : 100.0 %
- `variation_veille` : 100.0 %
- `variation_annee` : 100.0 %
- `capitalisation_marchande` : 100.0 %
- `drupal_id` : 17.8 %
- `id` : 0.0 %
- `code_index` : 0.0 %
- `date_index` : 0.0 %
- `valeur_index` : 0.0 %
- `scraped_at` : 0.0 %

## Métriques globales

- **tickers_fondamentaux** : 48
- **tickers_cours** : 78
- **indices_codes** : ['ESGI', 'FTSE_CSE_MOROCCO_15', 'MASI', 'MASI_AGROALIMENTAIRE', 'MASI_ASSURANCES', 'MASI_BANQUES', 'MASI_BATIMENT', 'MASI_BOISSONS', 'MASI_CHIMIE', 'MASI_DISTRIBUTEURS', 'MASI_ELECTRICITE', 'MASI_EUR', 'MASI_INGENIERIES', 'MASI_LOISIRS_HOTELS', 'MASI_MID_SMALL_CAP', 'MASI_MINES', 'MASI_PETROLE_GAZ', 'MASI_RENTABILITE_NET', 'MASI_SANTE', 'MASI_TRANSPORT', 'MASI_USD']
- **entreprises_financieres** : 63
- **publication_date_appliquee** : fondamentaux_rs (S1/S2)
- **masi_coverage** : {'code_index': 'MASI', 'expected_min_date': '2010-01-01', 'row_count': 4101, 'date_min': '2010-01-04', 'date_max': '2025-11-17', 'covers_from_2010': True, 'status': 'OK', 'notes': ['Historique MASI couvre depuis 2010-01-04 (4,101 lignes)']}
- **masi_covers_from_2010** : True
- **supabase_url** : https://pungyebagtntbocoewir.supabase.co
- **tables_chargees** : 651145

## Vérification MASI (benchmark)

- **code_index** : MASI
- **Lignes MASI** : 4,101
- **Date min** : 2010-01-04
- **Date max** : 2025-11-17
- **Couverture depuis 2010** : Oui
- **Statut** : OK
- **Note** : Historique MASI couvre depuis 2010-01-04 (4,101 lignes)

## Règles appliquées (Étape 1)

- **Look-ahead** : S1 (juin N) → disponible fin septembre N ; S2 (décembre N) → disponible fin avril N+1 (`publication_date`)
- **Valeurs manquantes** : carry-forward (ffill) uniquement, pas de bfill
- **Données financières** : filtre `statut_validation=VALIDE`, comptes `CONSOLIDE` pour donnees_financieres
- **Cours** : périmètre depuis `COURS_MIN_DATE` (défaut 2010-01-01)
- **random_state** : 42 (reproductibilité)
## Vérification MASI (Étape 2)

- **code_index** : MASI
- **Lignes MASI** : 4,101
- **Date min** : 2010-01-04
- **Date max** : 2025-11-17
- **Couverture depuis 2010** : Oui
- **Statut** : OK
- **Note** : Historique MASI couvre depuis 2010-01-04 (4,101 lignes)
