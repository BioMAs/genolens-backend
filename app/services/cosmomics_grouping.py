"""
CosmOmics nova — grouping of ontology terms into pathway sub-groups.

The claim referential is a flat list of terms (GO, Reactome, STRING clusters). The nova
reasons on *sub-groups*: one pathway = several terms. This module derives a sub-group name
from a term label, deterministically and without external data:

  1. a theme keyword found in the label ("autophag" -> "Autophagy", "collagen" ->
     "Collagen & ECM synthesis", ...), most specific theme first;
  2. otherwise the label stem — the label stripped of "positive/negative regulation of",
     "biosynthetic/metabolic/catabolic process", ... — so "Regulation of keratinocyte
     proliferation" and "Keratinocyte proliferation" land together;
  3. otherwise the readable name of the referential category.

It serves two callers: `scripts/build_claim_subgroups.py`, which pre-fills the versioned
`app/data/claims/claim_subgroups.yaml` (then reviewed by a curator, who has the final say),
and the nova service, which groups at runtime the significant terms that no claim covers
("Not linked to a claim").

Also home to `label_match_key`, the label normalization used to join MSigDB-style names
("REACTOME_COLLAGEN_BIOSYNTHESIS_AND_MODIFYING_ENZYMES") against referential descriptions.
"""

from __future__ import annotations

import re
from typing import Optional

# (keyword regex, sub-group name). First match wins, so specific themes come before broad ones.
_THEMES: list[tuple[str, str]] = [
    (r"autophag|aggrephag|mitophag|lysosom|vacuol", "Autophagy"),
    (r"proteasom|ubiquitin|deubiquitin|sumoylation|neddylation", "Proteasome & ubiquitin"),
    (r"chaperon|protein folding|unfolded protein|heat shock|prefoldin|tric/cct", "Protein folding"),
    (r"collagen", "Collagen & ECM synthesis"),
    (r"elastic fib|elastin|microfibril|fibrillin", "Elastic fibres"),
    (
        r"matrix metallo|extracellular matrix|\becm\b|matrisome|basement membrane",
        "Extracellular matrix",
    ),
    (r"integrin|focal adhesion|cell-substrate|cell–matrix|cell-matrix", "Cell–matrix adhesion"),
    (
        r"tight junction|adherens junction|cell-cell junction|cell–cell junction|desmosom|"
        r"gap junction",
        "Cell–cell junctions",
    ),
    (r"keratin|cornif|epiderm|skin barrier|peptide cross-linking", "Epidermal differentiation"),
    (r"ceramide|sphingo", "Ceramide & sphingolipids"),
    (r"cholesterol|sterol|steroid|isoprenoid|mevalonate", "Sterol biosynthesis"),
    (
        r"fatty acid|acyl-coa|lipid|phospholipid|glycerolipid|triglycerid|lamellar",
        "Lipid metabolism",
    ),
    (
        r"hyaluron|glycosaminoglycan|proteoglycan|amino sugar|chondroitin|heparan",
        "Glycosaminoglycans",
    ),
    (r"glutathione", "Glutathione metabolism"),
    (
        r"nfe2l2|nrf2|antioxidant|anti-oxidant|oxidative stress|oxidant detox|reactive oxygen|"
        r"superoxide|peroxid|\bredox\b|nadph oxidase|nad\(p\)h oxidase|\bros\b|\brns\b|\buv\b",
        "Oxidative stress response",
    ),
    (
        r"xenobiotic|detoxif|phase i\b|phase ii\b|conjugation of|chemical stress|toxic",
        "Detoxification",
    ),
    (
        r"respirat|electron transport|oxidative phosphorylation|atp synth|complex i\b|"
        r"tricarboxylic|tca cycle|citric acid|nadh dehydrogenase|ubiquinone|cytochrome c oxidase",
        "Mitochondrial respiration",
    ),
    (r"mitochondri", "Mitochondrial function"),
    (r"glycoly|gluconeogen|pentose phosphate|pyruvate|glucose", "Glucose metabolism"),
    (r"purine|pyrimidine|nucleotide|nucleoside", "Nucleotide metabolism"),
    (r"amino acid|amino-acid|branched-chain|glutamine|serine|methionine", "Amino acid metabolism"),
    (r"ribosom|rrna", "Ribosome biogenesis"),
    (
        r"trna|aminoacylation|translation|translational|peptide chain elongation|\b[468]0s\b",
        "Translation",
    ),
    (r"splic|spliceosom|snrnp", "RNA splicing"),
    (
        r"mrna|rna export|rna processing|rna catabolic|rna stability|nonsense-mediated",
        "mRNA processing",
    ),
    (r"histone|chromatin|methylation|acetylation", "Chromatin regulation"),
    (
        r"dna repair|mismatch|excision repair|double-strand|homologous recombination|dna damage|"
        r"\bner\b|incision",
        "DNA repair",
    ),
    (r"telomer", "Telomere maintenance"),
    (r"senescen|p53|tp53", "Senescence & p53"),
    (r"apopto|necrop|necrosis|pyropto|cell death|caspase|bcl-2|\bbad\b", "Cell death"),
    (
        r"cell cycle|mitotic|mitosis|cyclin|apc/c|g1/s|g2/m|chromosome segregation|spindle|"
        r"dna replication",
        "Cell cycle",
    ),
    (r"wound|re-epithelial|epithelial cell migration", "Wound healing"),
    (r"angiogen|vascul|endothel|vegf", "Angiogenesis"),
    (r"\bwnt\b|beta-catenin|β-catenin", "Wnt signalling"),
    (r"tgf|smad|bmp", "TGF-β signalling"),
    (
        r"\begf|\bfgf|\bpdgf|\bigf|growth factor|receptor tyrosine kinase",
        "Growth-factor signalling",
    ),
    (r"\bmtor|\bampk|\bakt\b|pi3k|pip3", "PI3K–AKT–mTOR signalling"),
    (r"\bmapk|\berk\b|\bras\b|\braf\b|\bjnk|\bp38", "MAPK signalling"),
    (r"rho[a-z]? gtpase|gtpase cycle|rac1|cdc42|small gtpase", "Rho GTPase signalling"),
    (
        r"actin|actomyosin|myosin|cytoskelet|microtubule|stress fiber|contract",
        "Cytoskeleton & contraction",
    ),
    (
        r"interleukin|cytokine|chemokine|\btnf|nf-kb|nf-κb|nfkb|kappab|kappa b|inflamm|eicosanoid|"
        r"prostaglandin|leukotriene|interferon",
        "Inflammatory signalling",
    ),
    (
        r"complement|antigen|\bmhc\b|t cell|tcr|b cell|immune|toll-like|neutrophil|cd28|"
        r"fc epsilon|fcer|lectin receptor|infection|symbiont|phagocyt",
        "Immune response",
    ),
    (r"\btrp\b|neuropeptide|sensory|nocicept", "Neurosensory signalling"),
    (r"melan|pigment|tyrosinase", "Pigmentation"),
    (r"circadian|clock", "Circadian rhythm"),
    (r"hypoxia|\bhif\b|hypoxia-inducible", "Hypoxia response"),
    (
        r"golgi|endoplasmic reticulum|\ber\b|copi|copii|vesicle|endocyt|exocyt|secretion|"
        r"trafficking|transport to|retrograde|cargo|endosom",
        "Vesicle trafficking",
    ),
    (r"glycan|glycosyl", "Glycosylation"),
    (r"semaphorin|ephrin|\beph\b|motility|migration|axon guidance", "Cell migration & guidance"),
    (
        r"protein localization|nucleocytoplasmic|import into nucleus|"
        r"nuclear (?:import|export|pore|envelope)",
        "Protein localisation",
    ),
    (r"peptidase|proteolysis|protease", "Proteolysis"),
    (r"calcium", "Calcium signalling"),
    (r"insulin", "Insulin signalling"),
]
_THEME_RES = [(re.compile(pattern, re.IGNORECASE), name) for pattern, name in _THEMES]

# Modifiers stripped from a label before falling back to its stem.
_STEM_PREFIXES = re.compile(
    r"^(?:(?:positive|negative|pos\.|neg\.)\s+)?(?:regulation of\s+)?"
    r"(?:establishment of\s+|maintenance of\s+|cellular\s+)?",
    re.IGNORECASE,
)
_STEM_SUFFIXES = re.compile(
    r"\s+(?:biosynthetic|metabolic|catabolic)?\s*process$|\s+pathway$|\s+activity$",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(
    r"\s*\((?:GO:\d+|R-HSA-\d+|WP\d+|GO|Reactome|KEGG)\)\s*$", re.IGNORECASE
)
_MSIGDB_PREFIX = re.compile(
    r"^(?:REACTOME|KEGG|WP|HALLMARK|GOBP|GOCC|GOMF|BIOCARTA|PID|KEGG_MEDICUS)_", re.IGNORECASE
)
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

_CATEGORY_NAMES: dict[str, str] = {
    "BARRIER_LIPID": "Barrier lipids",
    "BARRIER_ECM": "Barrier ECM",
    "ECM_CYTO": "ECM & cytoskeleton",
    "IMMUNE_INFLAM": "Immunity & inflammation",
    "EICOSANOID": "Eicosanoids",
    "REDOX_ANTIOX": "Redox",
    "ENERGY_METAB": "Energy metabolism",
    "PROTEOSTASIS": "Proteostasis",
    "GENOME_STABILITY": "Genome stability",
    "AGING": "Ageing",
    "CELL_DEATH": "Cell death",
    "TRAFFICKING": "Trafficking",
    "GENE_EXPRESSION": "Gene expression",
    "CELL_CYCLE": "Cell cycle",
    "GROWTH_SIGNAL": "Growth signalling",
    "MTOR_SIGNAL": "mTOR signalling",
}


def clean_label(label: Optional[str]) -> str:
    """Strip a trailing "(GO:…)" / "(WP123)" accession and surrounding whitespace."""
    if not label:
        return ""
    return _PARENTHETICAL.sub("", str(label)).strip()


def label_theme(label: Optional[str]) -> Optional[str]:
    """The theme sub-group a label belongs to, or None when no theme keyword matches."""
    text = clean_label(label)
    if not text:
        return None
    for pattern, name in _THEME_RES:
        if pattern.search(text):
            return name
    return None


def label_stem(label: Optional[str]) -> str:
    """The label without its regulation prefix and process suffix, capitalised.

    "Positive regulation of keratinocyte proliferation" -> "Keratinocyte proliferation",
    "Ceramide biosynthetic process" -> "Ceramide".
    """
    text = clean_label(label)
    stem = _STEM_SUFFIXES.sub("", _STEM_PREFIXES.sub("", text)).strip(" ,;:-")
    stem = stem or text
    return stem[:1].upper() + stem[1:] if stem else ""


def subgroup_name(label: Optional[str], category: Optional[str] = None) -> str:
    """Sub-group name for one term: its theme if any, else its stem, else its category."""
    theme = label_theme(label)
    if theme:
        return theme
    stem = label_stem(label)
    if stem:
        return stem
    return _CATEGORY_NAMES.get((category or "").strip().upper(), "Other pathways")


def label_match_key(label: Optional[str]) -> str:
    """Lower-case alphanumeric key used to join enrichment names to referential labels.

    "REACTOME_COLLAGEN_BIOSYNTHESIS_AND_MODIFYING_ENZYMES" and
    "Collagen biosynthesis and modifying enzymes" both give
    "collagen biosynthesis and modifying enzymes".
    """
    text = clean_label(label)
    if not text:
        return ""
    text = _MSIGDB_PREFIX.sub("", text)
    return _NON_ALNUM.sub(" ", text.lower()).strip()
