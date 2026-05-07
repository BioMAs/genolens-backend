"""
AI Interpreter Service for GenoLens.
Uses local AI models (Ollama) to interpret RNA-seq comparison results.
No data is exported to external services - fully private and offline.
"""
import asyncio
import time
import json
import httpx
import os
from typing import Dict, Any, Optional, List, AsyncGenerator
import logging

# Module-level availability cache (TTL = 30 s) to avoid 2 HTTP calls per request
_availability_cache: Optional[Dict[str, Any]] = None
_availability_cache_ts: float = 0.0
_AVAILABILITY_TTL: float = 30.0

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
        model: str = "llama3.2:3b",
        timeout: float = 600.0,
        max_retries: int = 3
    ):
        """
        Initialize the AI interpreter.
        
        Args:
            base_url: Ollama API base URL (default: env OLLAMA_BASE_URL or http://localhost:11434)
            model: Model to use (default: llama3.2:3b - lightweight, 2GB)
            timeout: Request timeout in seconds (default: 600s = 10 minutes)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
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
                            "num_ctx": 4096,
                            "temperature": 0.3,
                            "top_p": 0.9,
                        },
                    },
                )

                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API returned status {response.status_code}")

                result = response.json()
                return result.get("response", "")

        except Exception as e:
            logger.error(f"Ollama raw call error: {str(e)}")
            raise

    async def _call_ollama_raw_stream(
        self, prompt: str, max_tokens: int = 2000
    ) -> AsyncGenerator[str, None]:
        """
        Like _call_ollama_raw but streams tokens as they are generated.
        Yields text chunks. Raises on connection or HTTP errors.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "num_predict": max_tokens,
                            "num_ctx": 4096,
                            "temperature": 0.3,
                            "top_p": 0.9,
                        },
                    },
                ) as response:
                    if response.status_code != 200:
                        raise Exception(f"Ollama API returned status {response.status_code}")
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("response", "")
                            if token:
                                yield token
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            raise Exception("Impossible de se connecter à Ollama pour le streaming")
        except Exception as e:
            logger.error(f"Ollama stream error: {str(e)}")
            raise

    async def interpret_chart_stream(
        self, chart_type: str, context: dict
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version of interpret_chart.
        Yields token chunks as they arrive from Ollama.
        """
        builders = {
            "volcano": self._build_volcano_prompt,
            "pca": self._build_pca_prompt,
            "umap": self._build_umap_prompt,
            "heatmap": self._build_heatmap_prompt,
            "enrichment": self._build_enrichment_prompt,
        }
        builder = builders.get(chart_type)
        if not builder:
            raise ValueError(f"Unknown chart_type: {chart_type!r}. Must be one of {list(builders)}")
        prompt = f"{self._CHART_SYSTEM}\n\n{builder(context)}"
        async for chunk in self._call_ollama_raw_stream(prompt, max_tokens=400):
            yield chunk
    
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
        Result is cached for 30 s to avoid repeated HTTP roundtrips.
        """
        global _availability_cache, _availability_cache_ts
        now = time.monotonic()
        if _availability_cache is not None and (now - _availability_cache_ts) < _AVAILABILITY_TTL:
            return _availability_cache

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                version_response = await client.get(f"{self.base_url}/api/version")
                version = version_response.json().get("version", "unknown")

                models_response = await client.get(f"{self.base_url}/api/tags")
                models_data = models_response.json()
                models = [m["name"] for m in models_data.get("models", [])]

                result: Dict[str, Any] = {
                    "available": True,
                    "models": models,
                    "current_model": self.model,
                    "model_available": self.model in models,
                    "version": version,
                    "base_url": self.base_url,
                }
        except Exception as e:
            logger.warning(f"Ollama not available: {str(e)}")
            result = {
                "available": False,
                "models": [],
                "current_model": self.model,
                "model_available": False,
                "error": str(e),
            }

        _availability_cache = result
        _availability_cache_ts = now
        return result
    
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

    # ── Chart-type system instruction ───────────────────────────────────────

    _CHART_SYSTEM = (
        "You are a scientific assistant helping non-scientists understand genomics results. "
        "Use plain, everyday language. Avoid technical jargon — if you must use a term, "
        "explain it in parentheses with a simple analogy. "
        "Keep responses to 3–5 sentences. Be warm and direct."
    )

    # ── Public dispatch method ───────────────────────────────────────────────

    async def interpret_chart(self, chart_type: str, context: dict) -> str:
        """
        Generate a plain-English interpretation of a chart.

        Args:
            chart_type: One of "volcano", "pca", "umap", "heatmap", "enrichment"
            context: Chart-specific data dict

        Returns:
            AI-generated plain-English text (3-5 sentences)
        """
        builders = {
            "volcano": self._build_volcano_prompt,
            "pca": self._build_pca_prompt,
            "umap": self._build_umap_prompt,
            "heatmap": self._build_heatmap_prompt,
            "enrichment": self._build_enrichment_prompt,
        }
        builder = builders.get(chart_type)
        if not builder:
            raise ValueError(f"Unknown chart_type: {chart_type!r}. Must be one of {list(builders)}")
        prompt = f"{self._CHART_SYSTEM}\n\n{builder(context)}"
        return await self._call_ollama_raw(prompt, max_tokens=400)

    # ── Per-chart prompt builders ────────────────────────────────────────────

    def _build_volcano_prompt(self, ctx: dict) -> str:
        up = ctx.get("up_count", 0)
        down = ctx.get("down_count", 0)
        name = ctx.get("comparison_name", "the experiment")
        top_up = ctx.get("top_up_genes", [])[:5]
        top_down = ctx.get("top_down_genes", [])[:5]
        up_names = ", ".join(g["gene_id"] for g in top_up) or "none"
        down_names = ", ".join(g["gene_id"] for g in top_down) or "none"
        return (
            f"I am looking at a volcano plot for the comparison '{name}'. "
            f"In this experiment, {up} genes became more active and {down} became less active "
            f"in the treated group compared to controls. "
            f"The most increased genes include: {up_names}. "
            f"The most decreased genes include: {down_names}. "
            f"Explain in plain language what this pattern means for someone without a biology background."
        )

    def _build_pca_prompt(self, ctx: dict) -> str:
        pc1 = ctx.get("variance_pc1", 0)
        pc2 = ctx.get("variance_pc2", 0)
        n_samples = ctx.get("n_samples", 0)
        groups = ", ".join(ctx.get("sample_groups", [])) or "unknown groups"
        separation = ctx.get("group_separation", False)
        sep_text = "The sample groups separate clearly on the plot." if separation else "The sample groups overlap on the plot."
        return (
            f"I am looking at a PCA (Principal Component Analysis) plot with {n_samples} samples. "
            f"The first axis explains {pc1:.1f}% of the variation, and the second explains {pc2:.1f}%. "
            f"The sample groups are: {groups}. {sep_text} "
            f"Explain in plain language what this means for someone without a biology background."
        )

    def _build_umap_prompt(self, ctx: dict) -> str:
        n_clusters = ctx.get("n_clusters", 0)
        sizes = ctx.get("cluster_sizes", [])
        n_samples = ctx.get("n_samples", 0)
        groups = ", ".join(ctx.get("sample_groups", [])) or "unknown groups"
        sizes_text = ", ".join(str(s) for s in sizes[:5]) if sizes else "unknown"
        return (
            f"I am looking at a UMAP plot that groups {n_samples} samples into {n_clusters} clusters. "
            f"Cluster sizes are: {sizes_text} samples each. "
            f"The sample groups present are: {groups}. "
            f"Explain in plain language what these groupings might tell us biologically."
        )

    def _build_heatmap_prompt(self, ctx: dict) -> str:
        n_genes = ctx.get("n_genes", 0)
        n_samples = ctx.get("n_samples", 0)
        n_clusters = ctx.get("n_clusters")
        top_genes = ctx.get("top_varying_genes", [])[:5]
        groups = ", ".join(ctx.get("sample_groups", [])) or "unknown groups"
        genes_text = ", ".join(top_genes) if top_genes else "various genes"
        cluster_text = f" The samples are grouped into {n_clusters} clusters." if n_clusters else ""
        return (
            f"I am looking at a heatmap showing expression levels of {n_genes} genes "
            f"across {n_samples} samples from groups: {groups}.{cluster_text} "
            f"The most variable genes include: {genes_text}. "
            f"Explain in plain language what this pattern of gene activity tells us."
        )

    def _build_enrichment_prompt(self, ctx: dict) -> str:
        category = ctx.get("category", "GO")
        name = ctx.get("comparison_name", "the experiment")
        regulation = ctx.get("regulation", "ALL")
        top_terms = ctx.get("top_terms", [])[:8]
        direction = {"UP": "more active", "DOWN": "less active", "ALL": "changed"}.get(regulation, "changed")
        terms_text = "; ".join(
            f"{t['name']} ({t.get('gene_count', '?')} genes)"
            for t in top_terms
        ) or "none found"
        return (
            f"I am looking at a {category} enrichment plot for the comparison '{name}'. "
            f"It shows biological processes that are {direction} in the treated group. "
            f"The top enriched processes are: {terms_text}. "
            f"Explain in plain language what these biological processes tell us about the experiment."
        )

