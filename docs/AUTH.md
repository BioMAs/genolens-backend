# Authentification autonome

GenoLens sait authentifier ses utilisateurs lui-même : comptes, mots de passe (argon2id), second facteur (TOTP et codes de secours), sessions révocables, journal d'audit. Ce code remplace, à terme, le fournisseur d'identité externe (Supabase Auth). Il est **inerte** tant que l'instance reste en `AUTH_MODE=supabase` avec `AUTH_LOCAL_ENABLED=false` : c'est le réglage par défaut du code, de `docker-compose.prod.yml` et de ce dépôt, et il garde le comportement de production d'avant.

Ce document dit à quoi servent les réglages, comment gérer les clés, quelles commandes l'exploitation a à sa disposition, et ce que les jetons de service font de la 2FA. La procédure de bascule d'une instance existante n'y figure pas : elle a son propre runbook.

## Les deux drapeaux

Deux réglages gouvernent toute la transition. `AUTH_MODE` dit **quel fournisseur fait foi** ; `AUTH_LOCAL_ENABLED` ouvre le système local **en plus**, pendant la fenêtre de migration où les deux répondent. Chaque chemin Supabase est conservé derrière `AUTH_MODE=supabase` jusqu'à sa suppression : le retour en arrière reste donc cohérent.

| | `AUTH_MODE=supabase` (défaut : avant la bascule, et retour arrière) | `AUTH_MODE=local` (à partir de la bascule, et instances on-premise) |
|---|---|---|
| Jetons acceptés | Supabase HS256, plus les jetons locaux si `AUTH_LOCAL_ENABLED=true` (pré-lancement) | locaux uniquement |
| Rôle d'un utilisateur | comportement actuel (`profiles`, synchronisé dans `users`) | `users.role` seul |
| Ligne `users` d'un compte inconnu | créée à la première requête (conservé) | jamais créée : ligne absente → `401` |
| Administration : inviter, créer, supprimer, lister, chercher par e-mail | code Supabase actuel | code local |
| Endpoints `/api/v1/auth/*` | `404` tant que `AUTH_LOCAL_ENABLED=false` (sauf `jwks.json`) ; ouverts s'il est `true` | ouverts |
| `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_JWT_SECRET` | **obligatoires** : sans eux le service refuse de démarrer | facultatifs |

Ordre de résolution d'un jeton : (1) un jeton `glst_` passe par la résolution des jetons de service, sans changement ; (2) un jeton local ES256 ; (3) un jeton Supabase HS256, seulement en mode `supabase`. Les jetons locaux sont acceptés quand `AUTH_MODE=local` **ou** `AUTH_LOCAL_ENABLED=true`.

`GET /api/v1/auth/jwks.json` répond toujours. Tant que le système local est fermé et qu'aucune clé n'est configurée, il rend un jeu vide plutôt qu'une erreur.

## Variables

| Variable | Défaut | Rôle |
|---|---|---|
| `AUTH_MODE` | `supabase` | Fournisseur qui fait foi : `supabase` ou `local`. |
| `AUTH_LOCAL_ENABLED` | `false` | Ouvre les endpoints de connexion locaux et accepte les jetons locaux quel que soit `AUTH_MODE`. |
| `AUTH_ISSUER` | `http://localhost:8001` dans le code ; `docker-compose.prod.yml` met l'hôte de l'API de production | Valeur du `iss` des jetons d'accès émis, vérifiée au retour. Elle doit être l'URL publique de l'API, en `https://` hors développement. Le frontend et l'admin-front vérifient le même `iss` (variable `AUTH_ISSUER`, côté serveur) et lisent les clés publiques à `AUTH_JWKS_URL`. |
| `AUTH_JWT_PRIVATE_KEYS` | vide | Clés de signature ES256 : PEM PKCS#8 (courbe P-256) encodé en base64, séparés par des virgules. La première signe, toutes vérifient. |
| `AUTH_ENCRYPTION_KEYS` | vide | Clés Fernet qui chiffrent les secrets TOTP au repos (et dont dérive la clé des codes de secours), séparées par des virgules. La première chiffre, toutes déchiffrent. |
| `AUTH_COOKIE_DOMAIN` | vide | Domaine des cookies `gl_at` et `gl_session`, partagé avec les applications web (par exemple `.exemple.test`). Vide : l'hôte de l'API seul. |
| `AUTH_COOKIE_SECURE` | `true` | Cookies envoyés en HTTPS seulement. À désactiver uniquement en développement en HTTP. |
| `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_JWT_SECRET` | aucun | Fournisseur externe. **Obligatoires seulement en `AUTH_MODE=supabase`** : le démarrage échoue avec un message qui nomme ceux qui manquent. Leur valeur n'est jamais écrite dans le message. |
| `SUPABASE_SERVICE_ROLE_KEY` | vide | Opérations d'administration du fournisseur externe (mode `supabase`). |
| `SUPABASE_DB_URL` | vide | Chaîne de connexion de la base du fournisseur externe, pour le seul script de migration des comptes (voir plus bas). Elle contient un mot de passe : elle n'est transmise qu'à cette commande, pas à l'environnement d'un service. |
| `APP_URL` | `http://localhost:3000` | URL publique du frontend, utilisée dans les liens des e-mails de sécurité. |
| `SMTP_*`, `EMAIL_FROM_ADDRESS` | vides | Les e-mails de sécurité (invitation, vérification, réinitialisation, mot de passe changé, 2FA activée ou réinitialisée, code de secours utilisé) sont envoyés en mode **obligatoire** : sans SMTP configuré, l'envoi échoue au lieu de passer sous silence. |

`.env.example` liste tous ces réglages avec un exemple sûr. `docker-compose.prod.yml` transmet les `AUTH_*` à `api`, `worker` et `beat`, avec les défauts ci-dessus : une variable absente du `.env` de l'hôte ne change rien au démarrage en mode `supabase`. Le `r-worker` ne reçoit que `AUTH_MODE` et `AUTH_LOCAL_ENABLED` : il charge les mêmes réglages que les autres, ne signe personne et ne détient aucune clé.

### Démarrage

- `AUTH_MODE=supabase` et `AUTH_LOCAL_ENABLED=false` : aucune clé n'est lue, rien d'autre n'est exigé qu'avant.
- `AUTH_MODE=local` ou `AUTH_LOCAL_ENABLED=true` : l'API charge les deux jeux de clés au démarrage. Hors production, une clé absente est remplacée par une clé éphémère (tout ce qu'elle a signé ou chiffré disparaît au redémarrage : acceptable pour du développement, jamais ailleurs). **En production (`ENVIRONMENT=production`), une clé absente est une erreur** : mieux vaut un service qui ne démarre pas qu'un service qui déconnecte tout le monde, ou rend illisibles les secrets 2FA, à chaque redémarrage.
- Les messages d'erreur sur les clés nomment la position de l'entrée fautive, jamais sa valeur.

## Instance on-premise (déploiement depuis l'administration)

Une instance on-premise n'a pas de fournisseur externe : la tâche de déploiement (`app/worker/tasks/deployment_task.py`) écrit un `.env` en `AUTH_MODE=local`, sans aucune variable `SUPABASE_*`.

| Réglage écrit | Valeur |
|---|---|
| `AUTH_MODE` | `local` |
| `AUTH_LOCAL_ENABLED` | `true` |
| `AUTH_ISSUER` | `https://<BACKEND_DOMAIN>` : le secret `BACKEND_DOMAIN` du serveur est donc **obligatoire** |
| `AUTH_COOKIE_DOMAIN` | le secret du même nom s'il existe, sinon vide |
| `AUTH_JWT_PRIVATE_KEYS`, `AUTH_ENCRYPTION_KEYS` | le secret du même nom s'il existe, sinon une clé générée pour cette instance |

Les clés générées sont **conservées** : au premier déploiement du backend d'un serveur, la tâche les génère, les enregistre **chiffrées** avec les autres secrets du serveur (table `deployment_secrets`, les mêmes lignes que celles de l'administration), puis écrit le `.env`. Aux déploiements suivants elle les relit avec le reste des secrets. Régénérer à chaque déploiement déconnecterait tout le monde et perdrait tous les secrets 2FA, ce que la tâche ne fait jamais. Si l'enregistrement échoue, le déploiement échoue avant d'écrire le `.env` : un `.env` ne contient jamais une clé que le déploiement suivant ne retrouverait pas. Le journal du déploiement nomme les clés créées, jamais leur valeur. Un déploiement qui ne porte pas sur le backend n'en crée aucune.

Pour imposer ses propres clés, ou faire une rotation, déposer les secrets `AUTH_JWT_PRIVATE_KEYS` et `AUTH_ENCRYPTION_KEYS` du serveur depuis l'administration (l'API des secrets remplace la valeur existante) avant le déploiement.

Un serveur déjà déployé avec l'ancien modèle bascule en `AUTH_MODE=local` à son prochain déploiement du backend : ses secrets `SUPABASE_*` ne sont plus écrits, et ses comptes ne sont pas migrés (le script de migration ne lit que la base du fournisseur externe).

## Clés : génération et rotation

Générer, dans l'image de l'API (rien n'est écrit sur le disque, rien n'est journalisé, aucune base n'est ouverte : les clés n'existent que dans le terminal) :

```bash
docker compose -f docker-compose.prod.yml run --rm --no-deps api python -m app.auth.cli generate-keys
```

La commande écrit deux lignes, dans cet ordre, prêtes à coller dans le coffre de secrets ou dans le `.env` de l'hôte :

```
AUTH_JWT_PRIVATE_KEYS=<base64 d'un PEM PKCS#8, P-256>
AUTH_ENCRYPTION_KEYS=<clé Fernet>
```

Les secrets sont générés hors dépôt et ne vont jamais dans git ni dans une image. Une clé perdue ne se retrouve pas.

Chaque clé de signature porte un `kid` dérivé de sa clé publique (16 premiers caractères hexadécimaux du SHA-256 de la clé publique encodée en DER) : il n'a pas à être configuré, et deux déploiements qui détiennent la même clé s'accordent dessus. Les clés publiques de toutes les clés listées sont publiées à `GET /api/v1/auth/jwks.json`.

**Rotation de la clé de signature.** Mettre la nouvelle clé **en premier** dans `AUTH_JWT_PRIVATE_KEYS` et garder l'ancienne après elle, séparées par une virgule ; redémarrer `api`, `worker` et `beat` (les clés sont lues une fois par processus). La première signe, toutes vérifient : les jetons déjà émis restent valides. Retirer l'ancienne une fois ses jetons expirés (durée de vie d'un jeton d'accès : 15 minutes ; les sessions se renouvellent seules avec la nouvelle clé). Par prudence, si les applications web gardent le JWKS en cache longtemps, publier d'abord la nouvelle clé en seconde position, laisser les caches se rafraîchir, puis la passer en première.

**Rotation de la clé de chiffrement.** Même principe : la nouvelle clé en premier dans `AUTH_ENCRYPTION_KEYS`, l'ancienne après. Les secrets déjà chiffrés restent lisibles. **Retirer une clé rend illisibles les secrets et les codes de secours faits avec elle** : leurs propriétaires doivent refaire leur enrôlement 2FA (`reset-mfa` ci-dessous). Une clé de chiffrement se retire donc en connaissance de cause, jamais par ménage.

Sauvegarder `AUTH_ENCRYPTION_KEYS` avec la même rigueur que la base : sans elle, la copie de la base ne rend pas les secrets 2FA.

## Commandes d'exploitation

Dans le conteneur de l'API (variables d'environnement de l'application chargées). Elles fonctionnent quel que soit `AUTH_MODE` : le dépannage doit marcher aussi pendant la transition.

```bash
# Générer des clés (voir plus haut).
docker compose -f docker-compose.prod.yml run --rm --no-deps api python -m app.auth.cli generate-keys

# Dépannage « accès perdu » : retirer le second facteur d'un compte et terminer ses sessions.
# Le compte en est prévenu par e-mail (au mieux : si l'e-mail ne part pas, un avertissement est écrit
# sur la sortie d'erreur et la réinitialisation tient quand même). Journalisé comme « break_glass ».
docker compose -f docker-compose.prod.yml exec api python -m app.auth.cli reset-mfa --email <adresse>

# Lever le verrouillage anti-force-brute d'un compte (dix échecs consécutifs verrouillent 15 minutes).
docker compose -f docker-compose.prod.yml exec api python -m app.auth.cli unlock --email <adresse>
```

`reset-mfa` et `unlock` cherchent le compte dans la table locale `users`, sans tenir compte de la casse ni des espaces. Une adresse inconnue, ou partagée par plusieurs comptes qui ne diffèrent que par la casse, est refusée avec le code de sortie 2 et rien n'est modifié. Aucune des commandes n'écrit un secret, un hachage, un jeton ou une clé, ni dans ses messages ni dans ses erreurs. Codes de sortie : 0 fait ; 2 usage, ou adresse inconnue ou partagée ; 1 (trace Python) échec ailleurs, rien n'a été validé.

Les actions équivalentes existent dans l'administration pour un administrateur qui peut encore se connecter (réinitialiser la 2FA, révoquer les sessions, lever un verrouillage) ; la commande est la voie de secours quand plus aucun administrateur ne le peut.

## Migration des comptes depuis Supabase

`python -m app.auth.migrate_from_supabase` copie les identifiants des comptes de la table locale `users` depuis la base du fournisseur externe (lecture seule de `auth.users` et de `public.profiles`). Elle ne concerne que **le temps d'avant la bascule**.

- **Connexion.** `SUPABASE_DB_URL` est la chaîne de connexion du **pooler en mode session** du projet (la connexion directe n'est joignable qu'en IPv6). La transmettre à la seule commande, sans l'écrire dans un `.env` ni dans l'environnement d'un service. Ne pas réinitialiser le mot de passe de cette base : une autre application peut s'en servir.
- **Simulation par défaut.** Sans option, la commande n'écrit rien : elle calcule ce que chaque compte recevrait et en donne les comptages. `--apply` écrit le plan en **une seule transaction** ; le script est rejouable sans effet de bord.
- **Règles.** Seuls les comptes présents dans `users` sont concernés ; le hachage bcrypt de Supabase (`$2…`) est repris, un compte sans hachage reçoit un lien de réinitialisation ; `email_verified_at` et `last_login_at` sont repris ; l'e-mail de Supabase fait foi, sauf conflit d'unicité (signalé) ; un rôle `ADMIN` ou `SCILICIUM_ADMIN` local est conservé ; un compte banni ou supprimé côté Supabase est signalé sans rien appliquer ; **un mot de passe déjà choisi dans le nouveau système n'est jamais écrasé**.
- **`--apply` est refusé en `AUTH_MODE=local`** (code de sortie 2). Une fois la bascule faite, les données locales font foi : un nouveau `--apply` remettrait par-dessus des rôles, des e-mails et des dates de connexion figés à leur valeur du jour J, y compris celui d'un administrateur rétrogradé depuis. La simulation et `--report` restent disponibles.
- **`--send-resets`** envoie un lien de réinitialisation aux comptes sans mot de passe. C'est une étape à part, après `--apply`, et seulement en `AUTH_MODE=local`.
- **Rapport.** `--report <chemin>` écrit les anomalies en JSON (identifiant, nature, ligne de détail). **Il contient des adresses e-mail** : le fichier est créé lisible par son seul propriétaire, et il reste **hors du dépôt**. Aucun hachage ne figure dans le rapport, dans la sortie ni dans les journaux, et la chaîne de connexion n'est jamais répétée, pas même dans une erreur.

```bash
read -rs SUPABASE_DB_URL && export SUPABASE_DB_URL      # saisie sans écho, jamais dans l'historique
docker compose -f docker-compose.prod.yml run --rm \
  -e SUPABASE_DB_URL -v <dossier hors dépôt>:/ops \
  api python -m app.auth.migrate_from_supabase --report /ops/rapport-migration.json
# Après lecture du rapport : même commande avec --apply.
unset SUPABASE_DB_URL
```

Codes de sortie : 0 fait ; 1 échec à l'exécution (fournisseur injoignable, erreur de base, e-mail de réinitialisation non parti) ; 2 usage ou configuration (`SUPABASE_DB_URL` absent, réinitialisations hors `AUTH_MODE=local`, rapport impossible à écrire, `--apply` en mode `local`).

## Purge de rétention

La tâche `purge_auth_records` tourne chaque jour à 03:15 UTC dans le service `beat` (après la sauvegarde de 02:00, avant les avertissements d'expiration de 06:30). Sans elle, rien ne supprime jamais ces lignes. Elle supprime, en une transaction, ce qui est **strictement** plus vieux que la limite :

| Table | Supprimé quand |
|---|---|
| `auth_events` (journal d'audit) | créé il y a plus de 365 jours |
| `auth_one_time_tokens` (liens et étapes à usage unique) | expiré ou consommé il y a plus de 7 jours |
| `auth_sessions` | fin absolue, révocation ou expiration d'inactivité il y a plus de 30 jours |

Une session encore vivante n'est jamais touchée. La purge ne touche ni un compte, ni un code de secours, ni un jeton de service. Elle ne dépend que de l'âge : un échec ou un doublon est sans conséquence, la prochaine exécution rattrape.

Pour mémoire, les durées de vie : jeton d'accès 15 minutes ; session 7 jours d'inactivité et 30 jours au plus ; étape de 2FA 5 minutes (5 essais), enrôlement 15 minutes.

## Jetons de service : l'exception à la 2FA

Un accès machine ne peut pas saisir de code TOTP. Les jetons de service (`glst_…`, voir `docs/SERVICE_TOKENS.md`) sont donc **une exception assumée à la 2FA** :

- ils fonctionnent **que leur propriétaire ait activé sa 2FA ou non**, pour que le pipeline survive à la bascule ;
- ce qui compense : ils ne s'émettent que **par commande sur le serveur** (un accès SSH, plus fort qu'une 2FA), leur portée est limitée aux routes qui s'y ouvrent explicitement, et leur propriétaire doit être un compte **actif** et non administrateur ;
- réinitialiser le mot de passe, réinitialiser la 2FA et « déconnecter partout » **ne les révoquent pas** : un pipeline ne doit pas casser parce que son propriétaire change de mot de passe ;
- supprimer le compte rend le jeton inutilisable (`401`), suspendre le compte le fait refuser ;
- en cas de compromission, l'action d'administration « révoquer les sessions » propose de **révoquer aussi les jetons de service** du compte (`revoke_service_tokens`) ; l'administration peut les lister en lecture seule (`GET /api/v1/admin/users/<id>/service-tokens` : nom, droits, dernier usage, jamais le jeton ni son empreinte), l'émission restant en ligne de commande.
