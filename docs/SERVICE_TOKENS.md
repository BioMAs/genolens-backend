# Jetons de service

Un jeton de service permet à un client machine (le pipeline d'analyse) d'appeler l'API de GenoLens sans jeton de connexion d'utilisateur. Il agit au nom d'un utilisateur, son **propriétaire**, et seulement sur quelques routes.

## Ce qu'est un jeton

- Il commence par `glst_`, suivi de 43 caractères base64url (`secrets.token_urlsafe(32)`). Le préfixe le distingue d'un JWT avant toute lecture : un jeton de service n'est jamais décodé comme un JWT, ni par l'authentification, ni par les middlewares (suivi des connexions, limitation de débit).
- Il s'envoie dans l'en-tête `Authorization: Bearer glst_…`, comme un JWT.
- La base n'en garde que l'empreinte sha256 (`service_tokens.token_sha256`). Le jeton lui-même n'est affiché qu'une fois, à sa création. Il n'apparaît ensuite dans aucun journal, aucune réponse et aucun message d'erreur. Le SQL que DEBUG fait écrire au journal garde ses paramètres masqués, l'empreinte comprise, sauf si `SQL_ECHO_PARAMETERS` les demande (en développement seulement : le compose de développement transmet cette variable). `docker-compose.prod.yml` pose `DEBUG=false` dans l'api, le worker, le beat et le r-worker : en production, le SQL n'est plus écrit au journal.
- Il porte les droits, la licence, le plan et les modules de son propriétaire. Les gardes de son propriétaire s'appliquent : licence de l'application, abonnement expiré, plan TEAM ou ON_PREMISE pour les rapports PDF, module cosmétique pour le rapport éditorial, quota d'analyses DEG, accès au projet. Un jeton ne crée jamais de compte et ne modifie jamais celui de son propriétaire (rôle, statut).
- Deux règles s'y ajoutent, propres aux jetons. D'une part, son propriétaire ne peut pas être un administrateur (`ADMIN` ou `SCILICIUM_ADMIN`) : un jeton agit comme son propriétaire, et les droits d'un administrateur dépassent de loin les trois droits du jeton (moindre privilège). La commande `create` refuse un tel propriétaire (`an admin account cannot own a service token`) et n'écrit aucune ligne. Le contrôle a lieu à la création, pas à chaque requête : le jeton d'un compte devenu administrateur depuis se révoque. D'autre part, son propriétaire doit être un compte **actif** (`active`). Un compte en attente d'invitation (`pending`), suspendu ou résilié n'obtient pas de jeton. Le jeton d'un compte qui cesse d'être actif est refusé sur toutes ses routes, avant toute création de projet, tout dépôt, toute unité de quota et toute demande de rapport. Un jeton n'active jamais une invitation en attente.

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
- Le job garde l'identifiant du jeton (`report_jobs.service_token_id`). Un jeton ne voit, au suivi comme au téléchargement, que **ses** jobs : jamais un rapport habillé fait par le propriétaire dans l'interface, ni le job d'un autre jeton, même du même propriétaire. Une demande identique de deux jetons donne donc deux jobs.
- `GET …/report/status` et `GET …/report/download` acceptent un paramètre facultatif `job_id` (UUID, celui que rend la demande). Donné, il désigne ce job-là, qui doit appartenir au dataset, à la comparaison, au format (`template`) et à la visibilité de l'appelant, et être terminé pour le téléchargement. Sinon : 404, avec le message habituel de la route (`No report job found for this comparison` au suivi, `No completed report available` au téléchargement). Absent, la route rend le dernier job, comme avant. Un `job_id` qui n'est pas un UUID donne 422. Le client qui a demandé un rapport suit ainsi son job, et non le dernier du même dataset.

## Interrupteur IA (`use_ai`)

La demande de rapport (`POST /api/v1/datasets/{id}/report`) accepte un corps JSON `{"use_ai": false}`. La valeur par défaut est `true`, ce qui laisse l'interface inchangée. Le champ est lu pour tout appelant, qu'il ait le module de personnalisation ou non.

Avec `false` :

- aucun LLM n'est appelé ;
- aucune interprétation en cache n'est lue (`ai_interpretations`, `cosmetic_interpretations`) ;
- le rapport ne porte aucun texte ni libellé IA : pas d'encadré d'interprétation, pas de modèle dans la traçabilité, pas de mention de l'IA dans le guide de lecture ni dans la méthode.

`use_ai` est enregistré sur le job (`report_jobs.use_ai`) et rendu par `GET …/report/status`. Il fait partie de la clé de réutilisation d'un job en cours (15 dernières minutes), avec le format et le demandeur (l'utilisateur, ou tel jeton : jamais le job d'un autre jeton). Changer `use_ai` ne renvoie donc jamais un job fait avec l'autre réglage. Le rapport LaTeX suit le même interrupteur.

### Textes IA imprimés (`ai_texts`)

Chaque texte IA d'un rapport porte le libellé `AI-generated text (model <id>, generated <AAAA-MM-JJ>)` : le modèle qui l'a écrit et la date de génération (UTC). Le libellé ne dit rien d'une relecture.

Le job enregistre les textes réellement imprimés dans `report_jobs.ai_texts`, une entrée `{kind, model, generated_at}` par texte (`kind` vaut `nova` pour la lecture de la nova, `comparison` pour l'interprétation de la comparaison). `GET …/report/status` les rend :

- `[]` : aucun texte imprimé (`use_ai` faux, ou aucun texte à jour) ;
- `null` : non enregistré (job pas terminé, ou fait par un serveur d'avant cette version).

Un texte en cache écrit sur d'autres données n'est ni imprimé ni listé. La lecture de la nova est liée à l'empreinte de la nova qui l'a produite (`cosmetic_interpretations.inputs_sha256`) ; l'interprétation de la comparaison, aux comptes de gènes et de voies qu'elle a enregistrés. Un texte d'avant l'empreinte n'a pas d'empreinte, donc il est périmé : le rapport éditorial le dit dans la traçabilité (« not included (generated on other data) »).

## Commande

Dans l'image de l'API (variables d'environnement de l'application chargées) :

```bash
# Créer : le jeton est écrit seul sur la sortie standard, le reste sur la sortie d'erreur, même avec DEBUG
# (la commande n'affiche pas le SQL).
# Le propriétaire doit être actif et non administrateur (voir « Préparer le compte propriétaire ») ; sinon, refus.
python -m app.scripts.service_token create --owner <e-mail ou id> --name pipeline
python -m app.scripts.service_token create --owner <e-mail ou id> --name essai \
    --scope projects:create --scope datasets:upload   # droits choisis (défaut : les trois)

# Lister : identifiant, nom, propriétaire, droits, dates. Jamais le jeton ni son empreinte.
python -m app.scripts.service_token list

# Révoquer, définitivement (401 dès la requête suivante).
python -m app.scripts.service_token revoke <id du jeton>
```

Un jeton perdu ne peut pas être réaffiché : on le révoque et on en crée un autre. `last_used_at` note la dernière requête acceptée qui a abouti, à la minute près : il n'est réécrit que s'il est vide ou plus vieux que 60 secondes (`LAST_USED_RESOLUTION`), pour qu'un client actif n'écrive pas la ligne à chaque requête.

## Préparer le compte propriétaire

Un compte invité par un administrateur reste `pending` tant que la personne invitée ne s'est pas connectée. Un compte machine ne se connecte jamais : un administrateur doit donc le rendre actif lui-même avant de créer le jeton.

1. Inviter le compte dédié depuis l'administration des utilisateurs, avec un rôle autre qu'administrateur, le plan TEAM (ou ON_PREMISE) et le module cosmétique s'il doit demander le rapport éditorial. Un administrateur a tous les modules sans les avoir, mais la commande refuse son compte : le compte machine reçoit les siens explicitement.
2. Le passer au statut actif par l'API, avec un jeton de connexion d'administrateur : `PATCH /api/v1/admin/users/<id>/status` avec le corps `{"status": "active"}`. La réponse rend `"status": "active"`. Pour un compte en attente, c'est la seule voie : l'administration des utilisateurs ne propose alors que de renvoyer l'invitation. Elle n'offre « Activer le compte » qu'une fois le compte sorti de l'attente (suspendu ou résilié).
3. Créer le jeton : `python -m app.scripts.service_token create --owner <e-mail du compte> --name pipeline`.

Pour une instance de recette dont la base est remplie par un script, poser directement `status = 'active'` sur la ligne du compte dans `users` revient au même.

## Bonnes pratiques

- Un propriétaire dédié : un compte interne actif et non administrateur, avec le plan et les modules dont le pipeline a besoin (TEAM et module cosmétique pour le rapport éditorial), qui ne sert pas à travailler dans l'interface.
- Le jeton se range dans un gestionnaire de secrets ou dans une variable d'environnement. Jamais dans un fichier de configuration versionné, ni sur une ligne de commande.
- Un jeton par usage. On révoque un jeton dès qu'il n'est plus utilisé, ou au moindre doute.
