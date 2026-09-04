"""
AI Interpreter Service for GenoLens.

Talks to a remote OpenAI-compatible LLM endpoint (Modal + vLLM serving Gemma 4)
to interpret RNA-seq comparison results. Configured via LLM_BASE_URL /
LLM_API_KEY / LLM_MODEL (see app/core/config.py and infra/modal/gemma_vllm.py).
"""
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from openai import APIConnectionError, AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

# Gemma 4 is a thinking model. Our tasks all want a direct answer within a small
# token budget, so we disable chain-of-thought per request: it would otherwise
# consume the max_tokens budget and (in streaming) leak into the prose. Templates
# that don't support the flag simply ignore it.
_NO_THINKING: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}


class LocalAIInterpreter:
    """
    Interpreter backed by an OpenAI-compatible LLM server (Modal/vLLM, Gemma 4).

    The class name is kept for backward compatibility with existing callers; the
    inference itself is now remote and stateless.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 600.0,
        max_retries: int = 3,
    ):
        """
        Args:
            base_url: OpenAI-compatible base URL incl. /v1 (default: settings.LLM_BASE_URL)
            model: Model id to use (default: settings.LLM_MODEL)
            timeout: Request timeout in seconds (default: 600s = 10 minutes)
            max_retries: Retry attempts on transient errors (handled by the SDK)
        """
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        # Modal returns a 303 redirect (to the same URL + __modal_function_call_id)
        # whenever a request outlives ~150s — which happens on every scale-to-zero
        # cold start while Gemma loads. The openai SDK's httpx client does NOT follow
        # redirects by default and would raise; we must opt in explicitly.
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=settings.LLM_API_KEY,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(follow_redirects=True, timeout=timeout),
        )
        # Token usage of the most recent completion (set by _record_usage). Read by
        # callers to log real token counts for per-user cost accounting.
        self.last_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _record_usage(self, response: Any) -> None:
        """Capture token usage from a completion response into self.last_usage."""
        u = getattr(response, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }

    async def interpret_comparison(
        self,
        comparison_name: str,
        deg_summary: Dict[str, Any],
        top_pathways: List[Dict[str, Any]],
        top_genes: List[Dict[str, Any]],
        language: str = "en",
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
            "top_genes": top_genes[:20],
        }

        prompt = self._build_prompt(context, language)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower = more deterministic
                top_p=0.9,
                max_tokens=600,
                extra_body=_NO_THINKING,
            )
            self._record_usage(response)
            interpretation = (response.choices[0].message.content or "").strip()
            if not interpretation:
                raise Exception("Empty response from AI model")
            return interpretation
        except APIConnectionError:
            logger.error("Cannot reach the LLM endpoint (%s)", self.base_url)
            raise Exception(
                "Service LLM indisponible. Vérifiez que le point d'accès Modal "
                "est déployé et que LLM_BASE_URL / LLM_API_KEY sont configurés."
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                top_p=0.9,
                max_tokens=900,
                extra_body=_NO_THINKING,
            )
            self._record_usage(response)
            interpretation = (response.choices[0].message.content or "").strip()
            if not interpretation:
                raise Exception("Empty response from AI model")
            return interpretation
        except APIConnectionError:
            logger.error("Cannot reach the LLM endpoint for cosmetics interpretation")
            raise Exception(
                "LLM endpoint unavailable. Ensure the Modal service is deployed "
                "and LLM_BASE_URL / LLM_API_KEY are configured."
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
        language: str = "en",
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant of interpret_comparison. Yields text chunks as generated.
        """
        context = {
            "comparison": comparison_name,
            "deg_up": deg_summary.get("up_count", 0),
            "deg_down": deg_summary.get("down_count", 0),
            "top_pathways": top_pathways[:15],
            "top_genes": top_genes[:20],
        }

        prompt = self._build_prompt(context, language)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=600,
                stream=True,
                extra_body=_NO_THINKING,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield token
        except APIConnectionError:
            logger.error("Cannot reach the LLM endpoint for streaming")
            raise Exception("Service LLM indisponible pour le streaming.")
        except Exception as e:
            logger.error(f"AI streaming error: {str(e)}")
            raise

    async def _call_llm_raw(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Raw single-shot generation for custom prompts (e.g. pathway selection).

        Returns the assistant text (empty string if the model returns nothing).
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=max_tokens,
                extra_body=_NO_THINKING,
            )
            self._record_usage(response)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"LLM raw call error: {str(e)}")
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
        Report LLM availability from CONFIG, without probing the endpoint.

        The LLM is a serverless (scale-to-zero) Modal endpoint. Actively probing it
        (e.g. on every page mount) would wake the container, hammer it with 429s
        during cold start, waste Modal credit, and yield false negatives that block
        the whole feature. So availability is derived from configuration: if a real
        remote endpoint is configured, the feature is available. Real failures
        surface at generation time (with retries/backoff).
        """
        url = self.base_url or ""
        configured = bool(url) and "localhost" not in url and "127.0.0.1" not in url
        return {
            "available": configured,
            "models": [self.model] if configured else [],
            "current_model": self.model,
            "model_available": configured,
            "base_url": self.base_url,
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,  # Shorter answers for Q&A
                extra_body=_NO_THINKING,
            )
            self._record_usage(response)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    # ── Tool-calling chat (agentic chat mode) ────────────────────────────────
    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        num_predict: int = 1024,
    ) -> Dict[str, Any]:
        """
        One non-streamed turn against /v1/chat/completions with optional tools.

        Returns the normalised assistant message dict:
            {"role": "assistant", "content": str, "tool_calls": [ ... ] | None}

        OpenAI/vLLM returns tool-call arguments as JSON *strings*; the orchestrator
        (chat_agent._extract_tool_calls) parses them.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": num_predict,
            "extra_body": _NO_THINKING,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = await self.client.chat.completions.create(**kwargs)
            self._record_usage(response)
            message = response.choices[0].message
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,  # JSON string
                        },
                    }
                    for tc in message.tool_calls
                ]
            return {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": tool_calls,
            }
        except APIConnectionError:
            logger.error("Cannot reach the LLM endpoint for tool-calling chat")
            raise Exception("Service LLM indisponible pour le chat.")
        except Exception as e:
            logger.error(f"chat_with_tools error: {str(e)}")
            raise

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        num_predict: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """
        Stream the FINAL narrative turn from /v1/chat/completions (no tools bound).

        Thinking is disabled so the stream carries prose only. Yields content
        tokens as they arrive.
        """
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=num_predict,
                stream=True,
                extra_body=_NO_THINKING,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices else None
                if token:
                    yield token
        except APIConnectionError:
            logger.error("Cannot reach the LLM endpoint for chat streaming")
            raise Exception("Service LLM indisponible pour le streaming.")
        except Exception as e:
            logger.error(f"chat_stream error: {str(e)}")
            raise


async def generate_and_store(db, deg_dataset_id, enrichment_dataset_id, comparison_name, language: str = "fr"):
    """
    Build the AI context from the DB and generate + persist an AIInterpretation.

    Genes come from the DEG dataset; pathways from the (separate) annoDB ENRICHMENT
    dataset. Pass `enrichment_dataset_id` when the caller already knows it, otherwise it
    is resolved here — see `app.services.enrichment_source`. Returns the AIInterpretation,
    or None if the LLM is unavailable / there is no DEG data (graceful for report use).
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

    # Resolve rather than assume: `enrichment_dataset_id` is optional, and falling straight back
    # to the DEG dataset found nothing on a self-service analysis, whose enrichment is a separate
    # annoDB ENRICHMENT dataset.
    pathways_ds = enrichment_dataset_id
    if pathways_ds is None:
        from app.models.models import Dataset
        from app.services.enrichment_source import resolve_pathway_dataset_id
        deg_dataset = await db.get(Dataset, deg_dataset_id)
        pathways_ds = (
            await resolve_pathway_dataset_id(db, deg_dataset, comparison_name)
            if deg_dataset is not None
            else deg_dataset_id
        )

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
        logger.info("generate_and_store: LLM endpoint unavailable for %s", comparison_name)
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
