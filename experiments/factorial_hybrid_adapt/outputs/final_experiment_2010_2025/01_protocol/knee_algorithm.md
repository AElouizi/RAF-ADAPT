# Algorithme Knee / distance à l'utopie (méthode principale)

Implémentation figée : `_knee_point_index` dans `bvc_recommender/models/portfolio_optimizer.py`.
La règle `ideal_distance` est **identique** (même index). Le profil `P_equilibre` coïncide donc avec `P_selected` lorsque `selection_rule=knee`.

## Informations utilisées (as-of t uniquement)

Sur le front de Pareto du mois t, chaque solution i a un vecteur d'objectifs **tous exprimés à minimiser** :

- F1 = − Alpha (score_opt pondéré)
- F2 = CVaR 95 % historique (rendements ≤ t)
- F3 = − L (si mode Pareto 3 objectifs)

Aucun rendement réalisé après t n'entre dans F.

## Normalisation min-max

Pour chaque colonne j de F :

    Z_{i,j} = (F_{i,j} − min_i F_{·,j}) / (max_i F_{·,j} − min_i F_{·,j})

Si max = min, Z_{·,j} = 0.

Le point d'utopie dans l'espace normalisé est **0**.

## Sélection

    i* = argmin_i  || Z_i ||_2

C'est le portefeuille **Knee / Utopie** (`P_selected`).

Les règles alternatives (robustesse uniquement, jamais substituts) :

- B : max Alpha
- C : max L
- D : min CVaR
- E : score composite déjà implémenté = même knee (`P_equilibre`)
