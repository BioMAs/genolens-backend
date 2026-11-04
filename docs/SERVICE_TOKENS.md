# Jetons de service

Un jeton de service permet à un client machine (le pipeline d'analyse) d'appeler l'API de GenoLens sans jeton de connexion d'utilisateur. Il agit au nom d'un utilisateur, son **propriétaire**, et seulement sur quelques routes.

## Ce qu'est un jeton

- Il commence par `glst_`, suivi de 43 caractères base64url (`secrets.token_urlsafe(32)`). Le préfixe le distingue d'un JWT avant toute lecture : un jeton de service n'est jamais décodé comme un JWT, ni par l'authentification, ni par les middlewares (suivi des connexions, limitation de débit).
- Il s'envoie dans l'en-tête `Authorization: Bearer glst_…`, comme un JWT.
- La base n'en garde que l'empreinte sha256 (`service_tokens.token_sha256`). Le jeton lui-même n'est affiché qu'une fois, à sa création. Il n'apparaît ensuite dans aucun journal, aucune réponse et aucun message d'erreur.
- Il porte les droits, la licence, le plan et les modules de son propriétaire. Toutes les gardes s'appliquent sans changement : licence de l'application, compte suspendu ou expiré, plan TEAM ou ON_PREMISE pour les rapports PDF, module cosmétique pour le rapport éditorial, quota d'analyses DEG, accès au projet. Un jeton ne crée jamais de compte et ne modifie jamais celui de son propriétaire (rôle, statut).

## Droits (`scopes`)

| Droit | Routes ouvertes |
|---|---|
| `projects:create` | `POST /api/v1/projects/` |
| `datasets:upload` | `POST /api/v1/datasets/upload`, `GET /api/v1/datasets/{id}` (suivi du traitement) |
| `reports:editorial` | `POST /api/v1/datasets/{id}/report`, `GET …/report/status`, `GET …/report/download`, chacune **avec `template=editorial`** |

Toute autre route est refusée. Une route s'ouvre aux jetons par le décorateur `@service_token_scope(...)`, placé sous celui du routeur (`app/api/deps/service_token.py`). Sans ce décorateur, la route refuse les jetons.

Réponses d'un jeton :

| Cas | Code | `detail` |
|---|---|---|
| Jeton mal formé, inconnu ou révoqué, ou propriétaire supprimé | 401 | `Invalid or revoked service token` |
| Jeton valide, route non ouverte aux jetons | 403 | `Service tokens are not accepted on this route` |
| Jeton valide, droit manquant | 403 | `This service token lacks the scope <droit>` |
| Rapport demandé, suivi ou téléchargé sans `template=editorial` | 403 | `A service token is accepted on this route only with template=editorial` |
| Garde du propriétaire (licence, compte, plan, module, quota) | celui de la garde | celui de la garde, comme pour l'utilisateur lui-même |

Sur une route publique (tarifs, santé, état de la licence), l'en-tête est ignoré comme pour tout appel anonyme.

## Rapport éditorial demandé par un jeton

- Il est rendu **sans l'habillage enregistré du propriétaire** (logo, institut, couleurs, page de garde). Le contenu personnalisé envoyé dans la demande (conclusion, matériel et méthodes, page de garde) est ignoré. Le rapport livré à un client ne porte donc jamais l'habillage du compte interne.
- Le job garde l'identifiant du jeton (`report_jobs.service_token_id`). Un jeton ne voit, au suivi comme au téléchargement, que les jobs demandés par un jeton : jamais un rapport habillé fait par le propriétaire dans l'interface.

## Interrupteur IA (`use_ai`)

La demande de rapport (`POST /api/v1/datasets/{id}/report`) accepte un corps JSON `{"use_ai": false}`. La valeur par défaut est `true`, ce qui laisse l'interface inchangée. Le champ est lu pour tout appelant, qu'il ait le module de personnalisation ou non.

Avec `false` :

- aucun LLM n'est appelé ;
- aucune interprétation en cache n'est lue (`ai_interpretations`, `cosmetic_interpretations`) ;
- le rapport ne porte aucun texte ni libellé IA : pas d'encadré d'interprétation, pas de modèle dans la traçabilité, pas de mention de l'IA dans le guide de lecture ni dans la méthode.

`use_ai` est enregistré sur le job (`report_jobs.use_ai`) et rendu par `GET …/report/status`. Il fait partie de la clé de réutilisation d'un job en cours (15 dernières minutes), avec le format et le type de demandeur (utilisateur ou jeton). Changer `use_ai` ne renvoie donc jamais un job fait avec l'autre réglage. Le rapport LaTeX suit le même interrupteur.

## Commande

Dans l'image de l'API (variables d'environnement de l'application chargées) :

```bash
# Créer : le jeton est écrit seul sur la sortie standard, le reste sur la sortie d'erreur.
python -m app.scripts.service_token create --owner <e-mail ou id> --name pipeline
python -m app.scripts.service_token create --owner <e-mail ou id> --name essai \
    --scope projects:create --scope datasets:upload   # droits choisis (défaut : les trois)

# Lister : identifiant, nom, propriétaire, droits, dates. Jamais le jeton ni son empreinte.
python -m app.scripts.service_token list

# Révoquer, définitivement (401 dès la requête suivante).
python -m app.scripts.service_token revoke <id du jeton>
```

Un jeton perdu ne peut pas être réaffiché : on le révoque et on en crée un autre. `last_used_at` note la dernière requête acceptée qui a abouti.

## Bonnes pratiques

- Un propriétaire dédié : un compte interne **actif**, avec le plan et les modules dont le pipeline a besoin (TEAM et module cosmétique pour le rapport éditorial), qui ne sert pas à travailler dans l'interface. Un jeton n'active jamais une invitation en attente : avec un propriétaire `pending`, le suivi et le téléchargement du rapport répondent 403 (`User account is inactive`).
- Le jeton se range dans un gestionnaire de secrets ou dans une variable d'environnement. Jamais dans un fichier de configuration versionné, ni sur une ligne de commande.
- Un jeton par usage. On révoque un jeton dès qu'il n'est plus utilisé, ou au moindre doute.
