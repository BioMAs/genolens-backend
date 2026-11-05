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
from app.services.cosmomics_nova import subgroup_side

logger = logging.getLogger(__name__)

# Gemma 4 is a thinking model. Our tasks all want a direct answer within a small
# token budget, so we disable chain-of-thought per request: it would otherwise
# consume the max_tokens budget and (in streaming) leak into the prose. Templates
# that don't support the flag simply ignore it.
_NO_THINKING: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

# What the nova prompt carries at most: claims, sub-groups per side of a claim, sub-groups not
# linked to a claim. What is cut is said in the prompt ("N of M claims").
PROMPT_CLAIMS_LIMIT = 9
PROMPT_SUBGROUPS_PER_SIDE = 3
PROMPT_UNLINKED_LIMIT = 6

CLAIM_STATE_TEXT = {
    "for_only": "evidence for only, none against",
    "against_only": "evidence against only, none for",
    "both_ways": "evidence both for and against",
}


def prompt_claims_sent(claims_total: int) -> int:
    """How many of a nova's claims the AI prompt carries."""
    return min(max(claims_total, 0), PROMPT_CLAIMS_LIMIT)


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

    async def interpret_cosmomics(
        self,
        nova: Dict[str, Any],
        comparison_name: Optional[str] = None,
    ) -> str:
        """
        Reading of a CosmOmics Pathway Nova (output of cosmomics_nova.build_nova), in two English
        markdown sections — a claim-by-claim reading and a summary — that say nothing beyond the
        sub-groups for / against each claim that the prompt carries.
        """
        prompt = self._build_cosmomics_prompt(nova, comparison_name)
        return await self._complete_cosmetic_prompt(prompt)

    async def _complete_cosmetic_prompt(self, prompt: str) -> str:
        """Run one cosmetic-persona prompt and return the non-empty answer."""
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

    def _build_cosmomics_prompt(
        self, nova: Dict[str, Any], comparison_name: Optional[str] = None
    ) -> str:
        """Build the English prompt that has a CosmOmics Pathway Nova read, claim by claim.

        It carries the data and asks for a reading of them and nothing else: no marketing
        paragraph, no regulatory wording written by the model. What is cut to fit (claims beyond
        `PROMPT_CLAIMS_LIMIT`, sub-groups beyond `PROMPT_SUBGROUPS_PER_SIDE` per side) is said in
        the prompt, as "N of M".

        What leaves for the hosted model is what the user agrees to before the first call: claim
        names, sub-group names, directions, counts and p-values, and nothing else. So the prompt
        carries no comparison name (it often holds the user's own condition or product names;
        `comparison_name` is accepted so callers need not change, and is never used), no dataset
        or analysis identifier, no column name of an uploaded table, no score and no setting.
        """

        def p_text(nl: float) -> str:
            return f"p = {10 ** -nl:.1e}" if nl >= 3 else f"p = {10 ** -nl:.3f}"

        def sub_text(s: Dict[str, Any]) -> str:
            arrow = "up" if s.get("obs") == "up" else "down"
            return f"{s['name']} ({arrow}, {p_text(s['nl'])})"

        def count(n: int, noun: str) -> str:
            return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

        all_claims = nova.get("claims", [])
        shown_claims = all_claims[:PROMPT_CLAIMS_LIMIT]
        claim_lines = []
        for c in shown_claims:
            subs = c.get("subgroups", [])
            pro = [s for s in subs if subgroup_side(s) == "for"][:PROMPT_SUBGROUPS_PER_SIDE]
            con = [s for s in subs if subgroup_side(s) == "against"][:PROMPT_SUBGROUPS_PER_SIDE]
            tied = [s for s in subs if subgroup_side(s) == "balanced"][:PROMPT_SUBGROUPS_PER_SIDE]
            counts = f"{count(c.get('n_for', 0), 'sub-group')} for, {c.get('n_against', 0)} against"
            if c.get("n_balanced"):
                counts += f", {c['n_balanced']} balanced"
            state = CLAIM_STATE_TEXT.get(c.get("state"))
            if state:
                counts += f"; {state}"
            line = f"- {c['name']} ({counts})"
            if pro:
                line += "; supporting: " + ", ".join(sub_text(s) for s in pro)
            if con:
                line += "; contradicting: " + ", ".join(sub_text(s) for s in con)
            if tied:
                line += "; balanced: " + ", ".join(sub_text(s) for s in tied)
            claim_lines.append(line)
        claims_text = "\n".join(claim_lines) or (
            "- No cosmetic claim has a significant pathway in this comparison."
        )
        cut = (
            f"; {len(shown_claims)} of {len(all_claims)} claims shown"
            if len(all_claims) > len(shown_claims)
            else ""
        )

        unlinked = nova.get("unlinked", [])
        unlinked_total = max(int(nova.get("unlinked_total") or 0), len(unlinked))
        unlinked_shown = unlinked[:PROMPT_UNLINKED_LIMIT]
        unlinked_text = ", ".join(sub_text(s) for s in unlinked_shown) or "none"
        if unlinked_total > len(unlinked_shown) and unlinked_shown:
            unlinked_text += f" ({len(unlinked_shown)} of {unlinked_total} sub-groups)"

        # What the p-values are, in a fixed vocabulary: the payload's own wording may carry the
        # column name of an uploaded table, which is the user's text and does not leave.
        p_source = str(nova.get("p_source") or "")
        if p_source.startswith("adjusted p from uploaded"):
            p_kind = "an adjusted p-value read from uploaded tables"
        elif p_source:
            p_kind = "a Benjamini–Hochberg adjusted p-value"
        else:
            p_kind = "an adjusted p-value"

        return f"""You are a scientist helping an R&D team read transcriptomic results.

The data below comes from the CosmOmics Pathway Nova: the enriched terms of this comparison are \
mapped onto a curated catalogue of cosmetic claims and grouped into pathway sub-groups. A \
sub-group counts FOR a claim when the direction in which its terms are observed (up / down) \
matches the one the claim expects, and AGAINST it otherwise; a sub-group whose evidence for and \
against cancels out is BALANCED. The p-value of a term is {p_kind}; the p shown for a \
sub-group is a combined p-value of its terms (unsigned Stouffer), not an adjusted p-value.

CLAIMS (strongest supporting evidence first; at most {PROMPT_SUBGROUPS_PER_SIDE} sub-groups per \
side are listed{cut}):
{claims_text}

SIGNIFICANT SUB-GROUPS NOT LINKED TO ANY CLAIM: {unlinked_text}

Write your answer in ENGLISH, in markdown, with EXACTLY these two sections:

## Claim-by-claim reading
For each claim listed above, in the order given, write 2-3 sentences: the sub-groups that \
support it, and those that go against it or are balanced, with their direction and p-value as \
given. When a claim has no pathway sub-group against it, say "no pathway sub-group against it \
among the significant terms stored" and do not present that as a confirmation of the claim.

## Summary
One short paragraph (3-5 sentences) that restates, claim by claim, how many sub-groups go for \
and against it and which sub-groups lead.

Say nothing beyond the data provided above: no gene, pathway, mechanism or product that is not \
named in it, and no wording of efficacy, benefit, proof, treatment or disease. Describe what \
the data show ("is enriched", "goes in the direction of the claim"), never what a product does."""

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


async def build_comparison_ai_context(db, deg_dataset_id, enrichment_dataset_id, comparison_name):
    """
    Inputs of `interpret_comparison` read from the DB: DEG counts, the top pathways and the
    genes with the largest fold changes. Returns None when the comparison has no DEG.

    Genes come from the DEG dataset; pathways from the (separate) annoDB ENRICHMENT
    dataset. Pass `enrichment_dataset_id` when the caller already knows it, otherwise it
    is resolved here — see `app.services.enrichment_source`.
    """
    from sqlalchemy import select, func
    from app.models.models import DegGene, EnrichmentPathway

    up = (await db.execute(select(func.count()).where(
        DegGene.dataset_id == deg_dataset_id,
        DegGene.comparison_name == comparison_name,
        DegGene.regulation == "UP"))).scalar() or 0
    down = (await db.execute(select(func.count()).where(
        DegGene.dataset_id == deg_dataset_id,
        DegGene.comparison_name == comparison_name,
        DegGene.regulation == "DOWN"))).scalar() or 0
    if up + down == 0:
        logger.info("build_comparison_ai_context: no DEG data for %s", comparison_name)
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
    return {"deg_summary": deg_summary, "top_pathways": top_pathways, "top_genes": top_genes}


async def generate_comparison_text(db, deg_dataset_id, enrichment_dataset_id, comparison_name,
                                   language: str = "en"):
    """
    Generate a comparison interpretation WITHOUT persisting it — for a deliverable that needs
    another language than the cached `AIInterpretation` row, which stays untouched.
    Returns (text, model) or None when there is no DEG data or the LLM is unavailable.
    """
    context = await build_comparison_ai_context(
        db, deg_dataset_id, enrichment_dataset_id, comparison_name)
    if context is None:
        return None
    interpreter = LocalAIInterpreter()
    availability = await interpreter.check_availability()
    if not availability.get("available") or not availability.get("model_available"):
        logger.info("generate_comparison_text: LLM endpoint unavailable for %s", comparison_name)
        return None
    text = await interpreter.interpret_comparison(
        comparison_name=comparison_name, language=language, **context)
    return text, interpreter.model


async def generate_and_store(db, deg_dataset_id, enrichment_dataset_id, comparison_name, language: str = "fr"):
    """
    Build the AI context from the DB and generate + persist an AIInterpretation.

    See `build_comparison_ai_context` for where genes and pathways come from. Returns the
    AIInterpretation, or None if the LLM is unavailable / there is no DEG data (graceful for
    report use).
    """
    from sqlalchemy import select
    from app.models.models import AIInterpretation

    context = await build_comparison_ai_context(
        db, deg_dataset_id, enrichment_dataset_id, comparison_name)
    if context is None:
        logger.info("generate_and_store: no DEG data for %s", comparison_name)
        return None
    deg_summary = context["deg_summary"]
    top_pathways = context["top_pathways"]
    top_genes = context["top_genes"]
    up, down = deg_summary["up_count"], deg_summary["down_count"]

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
