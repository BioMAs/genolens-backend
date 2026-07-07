"""
AI Interpreter Service for GenoLens.
Uses local AI models (Ollama) to interpret RNA-seq comparison results.
No data is exported to external services - fully private and offline.
"""
import asyncio
import httpx
import os
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging

logger = logging.getLogger(__name__)


class LocalAIInterpreter:
    """
    Local AI interpreter using Ollama with retry logic and fallback support.
    
    Prerequisites:
    1. Install Ollama: brew install ollama (macOS) or see https://ollama.ai
    2. Start Ollama: ollama serve
    3. Download BioMistral: ollama pull biomistral
    
    Alternative models (used as fallbacks):
    - ollama pull llama3.2:3b (2GB, lightweight)
    - ollama pull llama3.1:8b (8GB, general purpose)
    """
    
    # Model fallback chain (from most capable to most lightweight)
    MODEL_FALLBACK_CHAIN = [
        "llama3.2:3b",      # 2GB - Default, lightweight
        "llama3.1:8b",      # 8GB - Better quality if available
        "biomistral",       # 4.1GB - Specialized biomedical (may cause OOM)
    ]
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3
    ):
        """
        Initialize the AI interpreter.
        
        Args:
            base_url: Ollama API base URL (default: env OLLAMA_BASE_URL or http://localhost:11434)
            model: Model to use (default: env OLLAMA_MODEL or llama3.1:8b)
            timeout: Request timeout in seconds (default: 600s = 10 minutes)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.timeout = timeout
        self.max_retries = max_retries
    
    async def _call_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: Dict[str, Any],
        attempt: int = 1
    ) -> httpx.Response:
        """
        Call Ollama API with exponential backoff retry logic.
        
        Args:
            client: HTTP client
            url: API endpoint URL
            payload: Request payload
            attempt: Current attempt number
            
        Returns:
            HTTP response
            
        Raises:
            Exception: If all retries fail
        """
        try:
            response = await client.post(url, json=payload)
            
            # Handle OOM errors by falling back to lighter model
            if response.status_code == 500:
                logger.warning(f"Model {self.model} failed with OOM on attempt {attempt}")
                
                # Try next lighter model in fallback chain
                current_idx = self.MODEL_FALLBACK_CHAIN.index(self.model) if self.model in self.MODEL_FALLBACK_CHAIN else 0
                if current_idx < len(self.MODEL_FALLBACK_CHAIN) - 1:
                    fallback_model = self.MODEL_FALLBACK_CHAIN[current_idx + 1]
                    logger.info(f"Falling back to lighter model: {fallback_model}")
                    payload["model"] = fallback_model
                    return await self._call_with_retry(client, url, payload, attempt)
                else:
                    raise Exception(
                        "AI model out of memory. Please increase Docker memory allocation "
                        "(Settings > Resources > Memory to 8+ GB) or use a lighter model."
                    )
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                
                if attempt < self.max_retries:
                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds... (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    return await self._call_with_retry(client, url, payload, attempt + 1)
                else:
                    raise Exception(f"Ollama API returned status {response.status_code} after {self.max_retries} attempts")
            
            return response
            
        except httpx.TimeoutException:
            if attempt < self.max_retries:
                wait_time = 2 ** attempt
                logger.warning(f"Request timeout, retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                await asyncio.sleep(wait_time)
                return await self._call_with_retry(client, url, payload, attempt + 1)
            else:
                logger.error(f"Request timeout after {self.max_retries} attempts")
                raise Exception(
                    f"AI model timeout after {self.timeout}s and {self.max_retries} attempts. "
                    "Try using a smaller model or increase timeout."
                )
        self.max_retries = max_retries
        
    async def interpret_comparison(
        self,
        comparison_name: str,
        deg_summary: Dict[str, Any],
        top_pathways: List[Dict[str, Any]],
        top_genes: List[Dict[str, Any]],
        language: str = "en"
    ) -> str:
        """
        Generate a biological interpretation of a comparison.
        
        Args:
            comparison_name: Name of the comparison (e.g., "Treated_vs_Control")
            deg_summary: Summary of DEGs {"up_count": 520, "down_count": 312}
            top_pathways: Top enriched pathways (max 15)
            top_genes: Top differentially expressed genes (max 20)
            language: Output language ("fr" or "en")
        
        Returns:
            AI-generated interpretation text
        """
        context = {
            "comparison": comparison_name,
            "deg_up": deg_summary.get("up_count", 0),
            "deg_down": deg_summary.get("down_count", 0),
            "top_pathways": top_pathways[:15],
            "top_genes": top_genes[:20]
        }
        
        prompt = self._build_prompt(context, language)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower = more deterministic
                        "top_p": 0.9,
                        "num_predict": 600,  # Increased for better quality
                        "num_ctx": 1024,     # Context window size
                    }
                }
                
                # Use retry logic
                response = await self._call_with_retry(
                    client,
                    f"{self.base_url}/api/generate",
                    payload
                )
                
                result = response.json()
                interpretation = result.get("response", "")
                
                if not interpretation:
                    raise Exception("Empty response from AI model")
                
                return interpretation
                
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama. Is it running? (ollama serve)")
            raise Exception(
                "Impossible de se connecter à Ollama. "
                "Vérifiez qu'Ollama est démarré (ollama serve) "
                "et que le modèle est téléchargé (ollama pull llama3.2)"
            )
        except Exception as e:
            logger.error(f"AI interpretation error: {str(e)}")
            raise
    
    async def interpret_cosmetics(
        self,
        cosmetics_data: Dict[str, Any],
        comparison_name: Optional[str] = None,
    ) -> str:
        """
        Generate a cosmetic-focused interpretation aid from scored claim data.

        Output is always in ENGLISH and structured as:
          1. Per-claim narrative (mechanism in skin-care terms)
          2. Executive marketing summary
          3. Regulatory guardrail (cosmetic, not medical claims)

        Args:
            cosmetics_data: output of cosmetics_service.score_claims()
            comparison_name: optional comparison label

        Returns:
            AI-generated, English markdown interpretation text.
        """
        prompt = self._build_cosmetics_prompt(cosmetics_data, comparison_name)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "top_p": 0.9,
                        "num_predict": 900,
                        "num_ctx": 2048,
                    },
                }
                response = await self._call_with_retry(
                    client, f"{self.base_url}/api/generate", payload
                )
                result = response.json()
                interpretation = result.get("response", "")
                if not interpretation:
                    raise Exception("Empty response from AI model")
                return interpretation
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama for cosmetics interpretation")
            raise Exception(
                "Cannot connect to Ollama. Ensure the AI server is running and "
                "the model is available."
            )
        except Exception as e:
            logger.error(f"Cosmetics AI interpretation error: {str(e)}")
            raise

    def _build_cosmetics_prompt(
        self, data: Dict[str, Any], comparison_name: Optional[str] = None
    ) -> str:
        """Build the English cosmetic-persona prompt from scored claim data."""
        claims = [c for c in data.get("claims", []) if c.get("n_supporting", 0) > 0]
        claims = claims[:8]
        zones = data.get("skin_zones", [])
        caveats = data.get("caveats", [])

        if claims:
            claims_text = "\n".join(
                f"- {c['label']}: activation score {c['score']}/100, "
                f"direction {c['direction']}, confidence {c['confidence']}, "
                f"{c['n_supporting']} supporting pathways"
                + (f" (e.g. {', '.join(p['pathway_name'] for p in c['evidence_pathways'][:3])})"
                   if c.get('evidence_pathways') else "")
                + (f"; key genes: {', '.join(c['top_genes'][:6])}" if c.get("top_genes") else "")
                for c in claims
            )
        else:
            claims_text = "- No cosmetic claim was significantly supported in this comparison."

        zones_text = "\n".join(
            f"- {z['label']}: activity {z['activity']}/100 ({z['dominant_direction']})"
            for z in zones
            if z.get("activity", 0) > 0
        ) or "- No clear skin-compartment engagement."

        caveats_text = "\n".join(
            f"- {c.get('pathway_name', c.get('term_id'))} [{c.get('flag')}]: {c.get('note') or ''}"
            for c in caveats[:6]
        ) or "- None flagged."

        comp = f" for the comparison '{comparison_name}'" if comparison_name else ""

        return f"""You are a senior cosmetic-science expert helping a R&D team turn \
transcriptomic results into clear, marketing-ready skin-benefit insights{comp}.

The data below was computed by mapping the modulated biological pathways of this \
comparison onto a curated catalogue of skin "claims". Each claim has an activation \
score (0-100), a direction (favorable/unfavorable), and a confidence level.

COSMETIC CLAIM SCORES:
{claims_text}

SKIN COMPARTMENT ENGAGEMENT:
{zones_text}

CAVEATS / FLAGS (pathways to handle with care):
{caveats_text}

Write your answer in ENGLISH, in markdown, with EXACTLY these three sections:

## Claim-by-claim narrative
For each well-supported claim (highest scores first), explain in 2-3 sentences the \
biological mechanism in accessible skin-care language (what it means for the skin), \
and how strongly the data supports it. Skip claims with no support.

## Executive summary
One short, punchy marketing-oriented paragraph (4-6 sentences) describing the overall \
effect of the tested product/ingredient on the skin.

## Regulatory note
A brief disclaimer. These are COSMETIC effects only, not medical or therapeutic claims. \
Mention any caveats above. Use cautious, compliant wording (e.g. "helps support", \
"appears to", "may contribute to") and avoid disease/treatment language.

Be specific, rely only on the data provided, and never invent gene names or pathways."""

    async def interpret_comparison_stream(
        self,
        comparison_name: str,
        deg_summary: Dict[str, Any],
        top_pathways: List[Dict[str, Any]],
        top_genes: List[Dict[str, Any]],
        language: str = "en"
    ) -> AsyncGenerator[str, None]:
        """
        Generate a biological interpretation of a comparison with streaming support.
        Yields text chunks as they are generated by the AI model.
        
        Args:
            comparison_name: Name of the comparison
            deg_summary: Summary of DEGs
            top_pathways: Top enriched pathways
            top_genes: Top differentially expressed genes
            language: Output language ("fr" or "en")
        
        Yields:
            str: Text chunks as generated
        """
        context = {
            "comparison": comparison_name,
            "deg_up": deg_summary.get("up_count", 0),
            "deg_down": deg_summary.get("down_count", 0),
            "top_pathways": top_pathways[:15],
            "top_genes": top_genes[:20]
        }
        
        prompt = self._build_prompt(context, language)
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 600,
                        "num_ctx": 1024,
                    }
                }
                
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"Ollama streaming error: {response.status_code}")
                        raise Exception(f"Ollama API returned status {response.status_code}")
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    yield chunk["response"]
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama for streaming")
            raise Exception("Impossible de se connecter à Ollama pour le streaming")
        except Exception as e:
            logger.error(f"AI streaming error: {str(e)}")
            raise
    
    async def _call_ollama_raw(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Raw call to Ollama API for custom prompts.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            
        Returns:
            Raw AI response text
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": 0.3,  # Lower temperature for more focused responses
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API returned status {response.status_code}")
                
                result = response.json()
                return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Ollama raw call error: {str(e)}")
            raise
    
    def _build_prompt(self, context: Dict[str, Any], language: str = "fr") -> str:
        """
        Build the prompt for the AI model.
        
        Args:
            context: Comparison context with DEG and pathway data
            language: Output language
        
        Returns:
            Formatted prompt string
        """
        # Format pathways list
        pathways_text = "\n".join([
            f"- {p.get('pathway_name', p.get('term', 'Unknown'))} "
            f"(adj.p={p.get('padj', 0):.2e}, "
            f"{p.get('gene_count', p.get('count', 0))} gènes)"
            for p in context["top_pathways"][:10]  # Only top 10
        ])
        
        # Format genes list
        genes_text = "\n".join([
            f"- {g.get('gene_name', g.get('gene_id', 'Unknown'))}: "
            f"logFC={g.get('log_fc', 0):.2f}, "
            f"adj.p={g.get('padj', 1):.2e}"
            for g in context["top_genes"][:15]  # Only top 15
        ])
        
        # Build the interpretation prompt (simplified for faster response)
        if language == "fr":
            return f"""Tu es un expert en bioinformatique et biologie moléculaire spécialisé en analyse transcriptomique RNA-seq.

Analyse cette comparaison d'expression génique :

Comparaison : {context['comparison']}

Résumé des gènes différentiellement exprimés (DEG) :
- {context['deg_up']} gènes SUREXPRIMÉS (UP-regulated)
- {context['deg_down']} gènes SOUS-EXPRIMÉS (DOWN-regulated)
- Total : {context['deg_up'] + context['deg_down']} DEG significatifs

Top 10 voies biologiques enrichies (Gene Ontology, KEGG, Reactome) :
{pathways_text}

Top 15 gènes clés :
{genes_text}

TÂCHE : Fournis une interprétation biologique concise en 3 paragraphes :

1. Vue d'ensemble : Quel est le phénomène biologique principal ? Quels processus sont activés ou inhibés ?

2. Analyse des voies : Explique les top 3-4 voies biologiques les plus significatives et leur lien.

3. Gènes clés : Identifie 3-4 gènes importants et leur rôle.

STYLE : Scientifique mais accessible. Cite les voies et gènes spécifiques. N'utilise PAS de formatage Markdown (pas de # pour les titres, pas de ** pour le gras, pas de * pour l'italique). Écris en texte simple avec des sauts de ligne entre les paragraphes.

LONGUEUR : 250-350 mots maximum.

Réponds DIRECTEMENT sans répéter les données brutes."""
        else:  # English (default)
            return f"""You are an expert in bioinformatics and molecular biology specialized in RNA-seq transcriptomic analysis.

Analyze this gene expression comparison:

Comparison: {context['comparison']}

Differentially Expressed Genes (DEG) Summary:
- {context['deg_up']} UP-regulated genes
- {context['deg_down']} DOWN-regulated genes
- Total: {context['deg_up'] + context['deg_down']} significant DEGs

Top 10 Enriched Biological Pathways (Gene Ontology, KEGG, Reactome):
{pathways_text}

Top 15 Key Genes:
{genes_text}

TASK: Provide a concise biological interpretation in 3 paragraphs:

1. Overview: What is the main biological phenomenon? Which processes are activated or inhibited?

2. Pathway Analysis: Explain the top 3-4 most significant pathways and their relationship.

3. Key Genes: Identify 3-4 important genes and their role.

STYLE: Scientific but accessible. Cite specific pathways and genes. Do NOT use Markdown formatting (no # for headings, no ** for bold, no * for italic). Write in plain text with line breaks between paragraphs.

LENGTH: 250-350 words maximum.

Answer DIRECTLY without repeating raw data."""

    def _format_pathways_text(self, pathways: List[Any]) -> str:
        """
        Format pathways for display.
        
        Args:
            pathways: List of pathway dictionaries
            
        Returns:
            Formatted string
        """
        return "\n".join([
            f"- {p.get('pathway_name', p.get('term', 'Unknown'))}: "
            f"p={p.get('padj', 1):.2e}, {p.get('gene_count', 0)} genes"
            for p in pathways[:10]
        ])
    
    async def check_availability(self) -> Dict[str, Any]:
        """
        Check if Ollama is available and which models are installed.
        
        Returns:
            {
                "available": bool,
                "models": List[str],
                "current_model": str,
                "version": str
            }
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check version
                version_response = await client.get(f"{self.base_url}/api/version")
                version = version_response.json().get("version", "unknown")
                
                # List models
                models_response = await client.get(f"{self.base_url}/api/tags")
                models_data = models_response.json()
                models = [m["name"] for m in models_data.get("models", [])]
                
                return {
                    "available": True,
                    "models": models,
                    "current_model": self.model,
                    "model_available": self.model in models,
                    "version": version,
                    "base_url": self.base_url
                }
        except Exception as e:
            logger.warning(f"Ollama not available: {str(e)}")
            return {
                "available": False,
                "models": [],
                "current_model": self.model,
                "model_available": False,
                "error": str(e)
            }
    
    async def generate_simple_answer(self, prompt: str) -> str:
        """
        Generate a simple answer to a user question.

        Args:
            prompt: The question prompt with context

        Returns:
            str: The AI's answer
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 300,  # Shorter answers for Q&A
                            "num_ctx": 1024,     # Reduced to prevent OOM
                        }
                    }
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code}")

                result = response.json()
                return result.get("response", "").strip()

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    # ── Tool-calling chat (agentic chat mode) ────────────────────────────────
    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        num_predict: int = 1024,
    ) -> Dict[str, Any]:
        """
        One non-streamed turn against Ollama's /api/chat with optional tool binding.

        Used by the agentic chat orchestrator to let the model decide whether to
        call a tool. Returns the raw assistant message dict:
            {"role": "assistant", "content": str, "tool_calls": [ ... ] | None}

        Ollama returns tool-call arguments already parsed as dicts (not JSON
        strings). llama3.1:8b sometimes emits a tool call inside `content` instead
        of `tool_calls`; the orchestrator handles that fallback, this method only
        transports the response faithfully.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
        if tools:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await self._call_with_retry(
                    client, f"{self.base_url}/api/chat", payload
                )
                result = response.json()
                message = result.get("message") or {}
                # Normalise: always expose a content string and a tool_calls list/None
                return {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", "") or "",
                    "tool_calls": message.get("tool_calls") or None,
                }
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama for tool-calling chat")
            raise Exception(
                "Impossible de se connecter à Ollama. Vérifiez qu'Ollama est démarré."
            )
        except Exception as e:
            logger.error(f"chat_with_tools error: {str(e)}")
            raise

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        num_ctx: int = 8192,
        num_predict: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the FINAL narrative turn from /api/chat (no tools bound).

        The 8B model is more reliable when tool-selection and prose generation are
        split: the orchestrator resolves tool calls with `chat_with_tools`, then
        streams the answer here with `tools=None` to force prose. Yields content
        tokens as they arrive.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"Ollama chat stream error: {response.status_code}")
                        raise Exception(f"Ollama API returned status {response.status_code}")

                    import json as _json
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        token = (chunk.get("message") or {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama for chat streaming")
            raise Exception("Impossible de se connecter à Ollama pour le streaming")
        except Exception as e:
            logger.error(f"chat_stream error: {str(e)}")
            raise



async def generate_and_store(db, deg_dataset_id, enrichment_dataset_id, comparison_name, language: str = "fr"):
    """
    Build the AI context from the DB and generate + persist an AIInterpretation.

    Genes come from the DEG dataset; pathways from the (separate) annoDB ENRICHMENT
    dataset when provided — the legacy interpret endpoint read pathways from the DEG
    dataset, which no longer holds enrichment rows. Returns the AIInterpretation, or
    None if Ollama is unavailable / there is no DEG data (graceful for report use).
    """
    from sqlalchemy import select, func
    from app.models.models import DegGene, EnrichmentPathway, AIInterpretation

    up = (await db.execute(select(func.count()).where(
        DegGene.dataset_id == deg_dataset_id,
        DegGene.comparison_name == comparison_name,
        DegGene.regulation == "UP"))).scalar() or 0
    down = (await db.execute(select(func.count()).where(
        DegGene.dataset_id == deg_dataset_id,
        DegGene.comparison_name == comparison_name,
        DegGene.regulation == "DOWN"))).scalar() or 0
    if up + down == 0:
        logger.info("generate_and_store: no DEG data for %s", comparison_name)
        return None

    deg_summary = {"up_count": up, "down_count": down, "total": up + down}

    pathways_ds = enrichment_dataset_id or deg_dataset_id
    pathways_rows = (await db.execute(
        select(EnrichmentPathway)
        .where(EnrichmentPathway.dataset_id == pathways_ds,
               EnrichmentPathway.comparison_name == comparison_name)
        .order_by(EnrichmentPathway.padj.asc()).limit(15)
    )).scalars().all()
    top_pathways = [{"pathway_name": p.pathway_name, "category": p.category,
                     "padj": p.padj, "gene_count": p.gene_count, "genes": p.genes or []}
                    for p in pathways_rows]

    top_genes_rows = (await db.execute(
        select(DegGene).where(DegGene.dataset_id == deg_dataset_id,
                              DegGene.comparison_name == comparison_name)
        .order_by(func.abs(DegGene.log_fc).desc()).limit(20)
    )).scalars().all()
    top_genes = [{"gene_id": g.gene_id, "gene_name": g.gene_name or g.gene_id,
                  "log_fc": g.log_fc, "padj": g.padj, "regulation": g.regulation}
                 for g in top_genes_rows]

    interpreter = LocalAIInterpreter()
    availability = await interpreter.check_availability()
    if not availability.get("available") or not availability.get("model_available"):
        logger.info("generate_and_store: Ollama unavailable for %s", comparison_name)
        return None

    interpretation = await interpreter.interpret_comparison(
        comparison_name=comparison_name, deg_summary=deg_summary,
        top_pathways=top_pathways, top_genes=top_genes, language=language)

    ai = AIInterpretation(
        dataset_id=deg_dataset_id, comparison_name=comparison_name,
        interpretation=interpretation, model=interpreter.model,
        deg_up=up, deg_down=down, pathways_count=len(top_pathways), genes_count=len(top_genes))
    db.add(ai)
    try:
        await db.commit()
        await db.refresh(ai)
    except Exception as exc:
        await db.rollback()
        existing = await db.scalar(select(AIInterpretation).where(
            AIInterpretation.dataset_id == deg_dataset_id,
            AIInterpretation.comparison_name == comparison_name))
        if existing:
            return existing
        logger.warning("generate_and_store persist failed for %s: %s", comparison_name, exc)
        return None
    return ai
