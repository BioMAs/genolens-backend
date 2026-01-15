"""
AI Interpreter Service for GenoLens.
Uses local AI models (Ollama) to interpret RNA-seq comparison results.
No data is exported to external services - fully private and offline.
"""
import httpx
import os
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class LocalAIInterpreter:
    """
    Local AI interpreter using Ollama.
    
    Prerequisites:
    1. Install Ollama: brew install ollama (macOS) or see https://ollama.ai
    2. Start Ollama: ollama serve
    3. Download BioMistral: ollama pull biomistral
    
    Alternative models:
    - ollama pull llama3.1:8b (8GB, general purpose)
    - ollama pull mixtral:8x7b (26GB, high quality)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llama3.2:3b",  # Switched to lighter model (2GB) to prevent OOM
        timeout: float = 600.0
    ):
        """
        Initialize the AI interpreter.
        
        Args:
            base_url: Ollama API base URL (default: env OLLAMA_BASE_URL or http://localhost:11434)
            model: Model to use (default: llama3.2:3b - lightweight, 2GB, good for basic interpretation)
            timeout: Request timeout in seconds (default: 600s = 10 minutes)
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        self.timeout = timeout
        
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
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,  # Lower = more deterministic
                            "top_p": 0.9,
                            "num_predict": 600,  # Increased for better quality with 8b model
                            "num_ctx": 1024,     # Reduced to prevent OOM
                        }
                    }
                )
                
                if response.status_code == 500:
                    logger.error(f"Ollama API error 500 - likely out of memory")
                    raise Exception(
                        "Le modèle AI manque de mémoire. "
                        "Augmentez la RAM allouée à Docker (Settings > Resources > Memory à 8+ GB) "
                        "ou utilisez un modèle plus petit."
                    )
                
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API returned status {response.status_code}")
                
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
                "et que le modèle est téléchargé (ollama pull biomistral)"
            )
        except httpx.TimeoutException:
            logger.error(f"Ollama request timeout after {self.timeout}s")
            raise Exception(
                f"Le modèle AI a mis trop de temps à répondre (>{self.timeout}s). "
                "Essayez avec un modèle plus petit (llama3.1:8b) ou augmentez le timeout."
            )
        except Exception as e:
            logger.error(f"AI interpretation error: {str(e)}")
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

