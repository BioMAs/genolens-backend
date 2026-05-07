# Sécurité - GenoLens Backend

## Vue d'ensemble

La sécurité du backend est implémentée à plusieurs niveaux : authentification, autorisation, rate limiting, validation des données et headers de sécurité.

---

## 1. Authentification (Supabase JWT)

### Mécanisme

L'authentification repose sur **Supabase Auth**. Les tokens JWT sont validés côté backend à chaque requête protégée.

```python
# app/core/supabase_auth.py
from supabase import create_client, Client
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """Valide le JWT Supabase et retourne les claims utilisateur."""
    token = credentials.credentials
    
    # Validation du token via Supabase
    supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    
    # Vérification de la signature JWT
    user = supabase.auth.get_user(token)
    
    if not user or not user.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    return {
        "id": str(user.user.id),
        "email": user.user.email,
        "role": user.user.user_metadata.get("role", "user"),
    }
```

### Flux d'authentification

```
Client → POST /auth/login (via Supabase SDK)
         │
         ├─► Supabase vérifie les credentials
         ├─► Retourne un JWT (access_token + refresh_token)
         │
         └─► Client stocke le token (localStorage/cookie)
         
Client → GET /api/v1/projects
         Headers: Authorization: Bearer <JWT>
         │
         └─► Backend valide le JWT via Supabase
```

### Rôles utilisateurs

| Rôle | Accès |
|---|---|
| `user` (default) | Ses propres projets, fonctionnalités standard |
| `admin` | Tous les projets, panel admin, gestion utilisateurs |
| `service_role` | Accès complet (backend interne uniquement) |

### Dépendances FastAPI

```python
# app/api/deps/auth.py
from fastapi import Depends
from .supabase_deps import get_supabase_client

async def get_current_user(
    supabase: Client = Depends(get_supabase_client),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Dépendance pour obtenir l'utilisateur courant."""
    ...

async def require_admin(
    current_user: dict = Depends(get_current_user)
):
    """Vérifie que l'utilisateur a le rôle admin."""
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user
```

---

## 2. Autorisation (Access Control)

### Par projet

Chaque requête sur un projet vérifie l'accès :

```python
async def check_project_access(
    db: AsyncSession,
    project_id: UUID,
    user_id: str
) -> Project:
    """Vérifie que l'utilisateur a accès au projet."""
    
    # L'owner a toujours accès
    project = await get_project(db, project_id)
    if project and project.owner_id == user_id:
        return project
    
    # Vérifier les membres partagés
    member_check = await db.execute(
        select(ProjectMember).where(
            and_(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id
            )
        )
    )
    
    if not member_check.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Project access denied")
    
    return project
```

### Par dataset

Les datasets héritent de l'accès du projet parent :

```python
async def check_dataset_access(
    db: AsyncSession,
    dataset_id: UUID,
    user_id: str
):
    """Vérifie l'accès à un dataset via le projet parent."""
    dataset = await get_dataset(db, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return await check_project_access(db, dataset.project_id, user_id)
```

### Par bookmark/commentaire

Chaque utilisateur ne peut modifier/supprimer que ses propres ressources :

```python
# Dans l'endpoint DELETE /bookmarks/{id}
if bookmark.user_id != current_user["id"]:
    raise HTTPException(status_code=403, detail="Only the owner can delete this bookmark")
```

---

## 3. Rate Limiting

### Configuration (slowapi)

```python
# app/middleware/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
```

### Limits par endpoint

| Endpoint | Limit | Période |
|---|---|---|
| `/auth/*` | 10 req | 1 minute |
| `/datasets/upload` | 5 req | 5 minutes |
| `/analyses/*` | 20 req | 1 minute |
| `/ai/chat` | 30 req | 1 minute |
| Default | 60 req | 1 minute |

### Application dans les endpoints

```python
from app.middleware import limiter

@router.post("/datasets/upload")
@limiter.limit("5/5 minutes")
async def upload_dataset(request: Request, ...):
    ...

@router.post("/ai/chat")
@limiter.limit("30/1 minute")
async def ai_chat(request: Request, ...):
    ...
```

### Réponse rate limit exceeded

```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "retry_after": "60"
}
```

Header HTTP : `429 Too Many Requests` + `Retry-After: 60`

---

## 4. Middlewares de sécurité

### SecurityHeadersMiddleware

```python
# app/middleware/security.py
class SecurityHeadersMiddleware:
    """Ajoute les headers de sécurité à toutes les réponses."""
    
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:;",
    }

    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value
        return response
```

### CORS Configuration

```python
# app/core/config.py
CORS_ORIGINS: list[str] = [
    "http://localhost:3000",   # Dev frontend
    "http://localhost:5173",   # Vite dev (alternative)
    "https://app.genolens.com",  # Production
]

# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Login Tracking Middleware

```python
# app/middleware/login_tracking.py
class LoginTrackingMiddleware:
    """
    Track successful logins in a non-blocking way.
    Deduplicates rapid successive login events from the same IP/user.
    """
    
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        
        # Only track successful auth responses
        if "/auth" in request.url.path and response.status_code == 200:
            asyncio.create_task(self._log_login(request))
        
        return response
    
    async def _log_login(self, request: Request):
        """Log the login event (non-blocking)."""
        ...
```

---

## 5. Validation des données

### Pydantic Schemas stricts

Toutes les entrées sont validées via Pydantic :

```python
class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=5000)
    species: Optional[str] = Field(None, pattern=r"^[a-z]{2,}_[a-z]+$")

class DifferentialExpressionRequest(BaseModel):
    dataset_ids: List[UUID] = Field(..., min_length=2, max_length=100)
    comparison: ComparisonGroup
    species: str = Field(pattern=r"^[a-z]{2,}_[a-z]+$")
    multiple_testing_correction: Literal["fdr", "bonferroni", "benjamini_hochberg"]
```

### Validation des fichiers uploadés

```python
async def validate_upload(file: UploadFile, max_size=500 * 1024 * 1024):
    """Valide un fichier uploadé."""
    
    # Vérifier l'extension
    allowed_extensions = {".csv", ".tsv", ".xlsx", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Extension '{ext}' not allowed")
    
    # Vérifier la taille
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(400, f"File too large (max {max_size / 1024**2} MB)")
    
    return content
```

### Validation des UUIDs et IDs

Tous les IDs sont validés comme des UUID v4 :

```python
from pydantic import UUID4, field_validator

class DatasetQuery(BaseModel):
    dataset_id: UUID4
    
    @field_validator("dataset_id")
    @classmethod
    def validate_uuid(cls, v):
        if v.version != 4:
            raise ValueError("Must be a valid UUID v4")
        return v
```

---

## 6. Protection des endpoints sensibles

### Admin endpoints

Tous les endpoints admin nécessitent le rôle `admin` :

```python
@router.get("/admin/users")
async def list_users(
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    ...
```

### Stripe Webhooks

Les webhooks Stripe sont protégés par signature :

```python
@router.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.stripe_webhook_secret
        )
    except ValueError:
        raise HTTPException(400, "Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(401, "Invalid signature")
    
    # Traitement de l'événement...
```

### Endpoint /metrics (Prometheus)

En production, `/metrics` doit être protégé au niveau du reverse proxy :

```nginx
# Exemple Nginx
location /metrics {
    allow 127.0.0.1;
    deny all;
}
```

---

## 7. Supabase RLS (Row Level Security)

Les politiques RLS de Supabase complètent la sécurité backend :

### Projets

```sql
-- Les utilisateurs ne voient que leurs projets
CREATE POLICY "Users can view own projects"
ON projects FOR SELECT
USING (auth.uid() = owner_id OR 
       id IN (SELECT project_id FROM project_members WHERE user_id = auth.uid()));

-- Seuls l'owner peut modifier
CREATE POLICY "Owner can update projects"
ON projects FOR UPDATE
USING (auth.uid() = owner_id);
```

### Bookmarks

```sql
-- Chaque utilisateur ne voit que ses bookmarks
CREATE POLICY "Users can view own bookmarks"
ON bookmark FOR SELECT
USING (auth.uid() = user_id);

-- Seuls l'owner peut supprimer
CREATE POLICY "Owner can delete bookmarks"
ON bookmark FOR DELETE
USING (auth.uid() = user_id);
```

---

## 8. Quotas & Limites par plan

### Vérification des quotas AI

```python
async def check_ai_quota(user_id: str) -> bool:
    """Vérifie que l'utilisateur a du quota AI disponible."""
    
    subscription = await stripe_service.get_subscription(user_id)
    
    if subscription.plan == "free":
        max_requests = 10
    elif subscription.plan == "premium":
        max_requests = 100
    else:  # advanced
        max_requests = float("inf")
    
    used = await get_ai_usage_count(user_id, period="24h")
    return used < max_requests
```

### Limites par plan

| Fonctionnalité | Free | Premium | Advanced |
|---|---|---|---|
| Projets max | 3 | 20 | ∞ |
| Datasets/projet | 5 | 50 | ∞ |
| Taille upload max | 100 MB | 500 MB | 500 MB |
| Quota AI/jour | 10 | 100 | ∞ |
| Analyses GSEA/jour | 5 | 50 | ∞ |

---

## 9. Logging & Audit

### Logs de sécurité

```python
logger = logging.getLogger("security")

# Exemples d'événements loggés
logger.warning(f"Failed login attempt from IP {ip}")
logger.info(f"Admin action: {action} by user {user_id}")
logger.error(f"Access denied: user {user_id} to project {project_id}")
```

### Journal d'activité des projets

Chaque action importante est enregistrée dans `project_activity_log` :

| Action | Détail enregistré |
|---|---|
| Project created | `{ "action": "create", "user_id": "..." }` |
| Dataset uploaded | `{ "action": "upload", "dataset_id": "...", "file_size": 123456 }` |
| Analysis run | `{ "action": "analysis_run", "type": "deg", "analysis_id": "..." }` |
| Member added | `{ "action": "member_add", "user_id": "..." }` |

---

## 10. Checklist de sécurité pour les nouveaux endpoints

Lors de l'ajout d'un nouvel endpoint, vérifier :

- [ ] Authentification requise (`Depends(get_current_user)`)
- [ ] Autorisation appropriée (owner/admin/membre)
- [ ] Validation des entrées (Pydantic schema)
- [ ] Rate limiting si nécessaire (`@limiter.limit()`)
- [ ] Headers de sécurité applicables
- [ ] Logging des actions sensibles
- [ ] Protection contre les injections (SQL, XSS)
- [ ] Taille maximale des fichiers uploadés
- [ ] Quota vérifié pour les ressources coûteuses