# Jetons de service

Un jeton de service permet à un client machine (le pipeline d'analyse) d'appeler l'API de GenoLens sans jeton de connexion d'utilisateur. Il agit au nom d'un utilisateur, son **propriétaire**, et seulement sur quelques routes.

## Ce qu'est un jeton

- Il commence par `glst_`, suivi de 43 caractères base64url (`secrets.token_urlsafe(32)`). Le préfixe le distingue d'un JWT avant toute lecture : un jeton de service n'est jamais décodé comme un JWT, ni par l'authentification, ni par les middlewares (suivi des connexions, limitation de débit).
- Il s'envoie dans l'en-tête `Authorization: Bearer glst_…`, comme un JWT.
- La base n'en garde que l'empreinte sha256 (`service_tokens.token_sha256`). Le jeton lui-même n'est affiché qu'une fois, à sa création. Il n'apparaît ensuite dans aucun journal, aucune réponse et aucun message d'erreur.
- Il porte les droits, la licence, le plan et les modules de son propriétaire. Toutes les gardes s'appliquent sans changement : licence de l'application, abonnement expiré, plan TEAM ou ON_PREMISE pour les rapports PDF, module cosmétique pour le rapport éditorial, quota d'analyses DEG, accès au projet. Un jeton ne crée jamais de compte et ne modifie jamais celui de son propriétaire (rôle, statut).
- Son propriétaire doit être un compte **actif** (`active`). Un compte en attente d'invitation (`pending`), suspendu ou résilié n'obtient pas de jeton. Le jeton d'un compte qui cesse d'être actif est refusé sur toutes ses routes, avant toute création de projet, tout dépôt, toute unité de quota et toute demande de rapport. Un jeton n'active jamais une invitation en attente.

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
| Propriétaire non actif (en attente, suspendu, résilié) | 403 | `{"error": "account_inactive", "status": "<statut>", "message": …}` |
| Garde du propriétaire (licence, abonnement expiré, plan, module, quota) | celui de la garde | celui de la garde, comme pour l'utilisateur lui-même |

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
# Le propriétaire doit être actif (voir « Préparer le compte propriétaire ») ; sinon, refus.
python -m app.scripts.service_token create --owner <e-mail ou id> --name pipeline
python -m app.scripts.service_token create --owner <e-mail ou id> --name essai \
    --scope projects:create --scope datasets:upload   # droits choisis (défaut : les trois)

# Lister : identifiant, nom, propriétaire, droits, dates. Jamais le jeton ni son empreinte.
python -m app.scripts.service_token list

# Révoquer, définitivement (401 dès la requête suivante).
python -m app.scripts.service_token revoke <id du jeton>
```

Un jeton perdu ne peut pas être réaffiché : on le révoque et on en crée un autre. `last_used_at` note la dernière requête acceptée qui a abouti.

## Préparer le compte propriétaire

Un compte invité par un administrateur reste `pending` tant que la personne invitée ne s'est pas connectée. Un compte machine ne se connecte jamais : un administrateur doit donc le rendre actif lui-même avant de créer le jeton.

1. Inviter le compte dédié depuis l'administration des utilisateurs, avec le plan TEAM (ou ON_PREMISE) et le module cosmétique s'il doit demander le rapport éditorial.
2. Le passer au statut actif. Dans l'administration des utilisateurs, choisir le statut « actif » sur sa ligne. Ou, par l'API, avec un jeton de connexion d'administrateur : `PATCH /api/v1/admin/users/<id>/status` avec le corps `{"status": "active"}`. La réponse rend `"status": "active"`.
3. Créer le jeton : `python -m app.scripts.service_token create --owner <e-mail du compte> --name pipeline`.

Pour une instance de recette dont la base est remplie par un script, poser directement `status = 'active'` sur la ligne du compte dans `users` revient au même.

## Bonnes pratiques

- Un propriétaire dédié : un compte interne actif, avec le plan et les modules dont le pipeline a besoin (TEAM et module cosmétique pour le rapport éditorial), qui ne sert pas à travailler dans l'interface.
- Le jeton se range dans un gestionnaire de secrets ou dans une variable d'environnement. Jamais dans un fichier de configuration versionné, ni sur une ligne de commande.
- Un jeton par usage. On révoque un jeton dès qu'il n'est plus utilisé, ou au moindre doute.
