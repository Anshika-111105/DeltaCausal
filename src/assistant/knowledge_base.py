from dataclasses import dataclass
from typing import Dict, List


@dataclass
class KnowledgeDocument:
    doc_id: str
    title: str
    category: str
    content: str


KNOWLEDGE_CORPUS: List[KnowledgeDocument] = [
    KnowledgeDocument(
        doc_id="causal_assumptions_01",
        title="Fundamental Causal Identification Assumptions",
        category="causal_theory",
        content=(
            "To interpret an estimated treatment effect tau(x) = E[Y(1) - Y(0) | X=x] causally, "
            "three core identification assumptions must hold:\n"
            "1. SUTVA (Stable Unit Treatment Value Assumption): No interference between units (one customer receiving an offer "
            "does not affect another customer's outcome) and no hidden variations of treatment.\n"
            "2. Positivity (Overlap): 0 < P(T=1 | X=x) < 1 for all x. Every customer profile must have a non-zero probability "
            "of both receiving and not receiving the intervention.\n"
            "3. Unconfoundedness (Ignorability / Conditional Independence): (Y(0), Y(1)) _||_ T | X. All common causes of "
            "treatment assignment and conversion outcome are observed in pre-treatment covariates X. "
            "In randomized controlled trials (RCTs), unconfoundedness holds unconditionally."
        ),
    ),
    KnowledgeDocument(
        doc_id="model_architectures_02",
        title="Uplift Meta-Learners Architecture and Trade-Offs",
        category="models",
        content=(
            "Comparative overview of uplift meta-learners:\n"
            "- S-Learner: Single model on [X, T]. Susceptible to regularization bias where treatment indicator T is shrunk or ignored if features X dominate.\n"
            "- T-Learner: Fits separate models mu_1(X) on treated and mu_0(X) on control. Prevents treatment shrinkage but can suffer high variance if sample sizes differ.\n"
            "- X-Learner (Künzel et al. 2019): Optimized for imbalanced treatment/control allocations (e.g. 85% treated / 15% control). "
            "Imputes counterfactuals D_1 = Y_1 - mu_0(X_1) and D_0 = mu_1(X_0) - Y_0, fits effect models, and combines them via propensity weighting.\n"
            "- Doubly Robust Learner (AIPW): Regresses the augmented inverse probability weighted pseudo-outcome on X. "
            "Yields unbiased CATE estimates if either the outcome models or the propensity model is correctly specified."
        ),
    ),
    KnowledgeDocument(
        doc_id="evaluation_metrics_03",
        title="Uplift Evaluation Metrics: Qini and AUUC",
        category="evaluation",
        content=(
            "Evaluating uplift models cannot use standard AUC or accuracy because counterfactual potential outcomes are unobservable.\n"
            "- Qini Curve: Plots cumulative incremental conversions against fraction of population targeted, scaling control conversions "
            "by (N_treated_total / N_control_total) to account for sample ratio disparities. The Qini Score measures the area between the model "
            "curve and the diagonal random baseline.\n"
            "- Area Under the Uplift Curve (AUUC): Integrated area of empirical treatment effects across cumulative targeting deciles.\n"
            "- PEHE (Precision in Estimation of Heterogeneous Effects): Root mean squared error between predicted CATE and true CATE. "
            "Computed exclusively on synthetic benchmark datasets where true potential outcomes Y(0) and Y(1) are mathematically known."
        ),
    ),
    KnowledgeDocument(
        doc_id="customer_segmentation_04",
        title="Causal Customer Archetypes and Targeting Rules",
        category="policy",
        content=(
            "Customers are segmented into four analytical quadrants based on baseline conversion P(Y(0)=1|X) and treatment effect tau(x):\n"
            "1. Persuadables (tau > 0, baseline low/moderate): Customers who only purchase if they receive an intervention. Primary target.\n"
            "2. Sure Things (baseline high, tau ~ 0): Customers who convert organically without promotional incentives. Targeting them generates zero incremental lift and wastes margin.\n"
            "3. Lost Causes (baseline low, tau ~ 0): Customers unresponsive to marketing touchpoints. Suppress from expensive campaigns.\n"
            "4. Sleeping Dogs (tau < 0): Customers where outreach induces negative reactions, fatigue, or unsubscriptions. Strictly exclude."
        ),
    ),
    KnowledgeDocument(
        doc_id="policy_optimization_05",
        title="Budget-Constrained Policy Optimization & Unit Economics",
        category="policy",
        content=(
            "Marketing campaign budget allocation formula:\n"
            "Expected Incremental Net Value per customer = tau_hat(x) * Margin - Cost.\n"
            "Optimal policy under fixed per-customer cost: Sort eligible customers by estimated CATE tau_hat(x) and select individuals "
            "where Expected Net Value > 0 until budget B is reached.\n"
            "Conventional Response-Propensity models fail because they rank by P(Y=1|X), selecting Sure Things who purchase organically, "
            "yielding poor incremental ROI compared to uplift-driven allocation."
        ),
    ),
    KnowledgeDocument(
        doc_id="safeguards_experiments_06",
        title="Causal Safeguards, Uncertainty, and Next Recommended Experiments",
        category="experimentation",
        content=(
            "Operational Safeguards and Next Steps:\n"
            "1. Individual Effect Caveat: Uplift models estimate conditional average treatment effects E[Y(1)-Y(0)|X=x], not deterministic individual fates.\n"
            "2. Selection Bias Check: If observational confounding is detected (e.g. SMD > 0.10 or propensity overlap violations), results must be interpreted as adjusted associations rather than experimental truths.\n"
            "3. Recommended Follow-Up Experiment: If validation Qini score is low or uncertainty intervals are wide, run a randomized holdout test (A/B test with 10-15% pure control group) focused on the highest-uncertainty deciles to refine CATE estimates."
        ),
    ),
]


class CausalKnowledgeBase:
    """Provides structured retrieval over the causal inference corpus."""

    def __init__(self):
        self.documents = KNOWLEDGE_CORPUS

    def get_all_documents(self) -> List[KnowledgeDocument]:
        return list(self.documents)

    def search_keyword(self, query: str, top_k: int = 3) -> List[KnowledgeDocument]:
        """Simple keyword scoring for fast retrieval."""
        tokens = query.lower().split()
        scores = []
        for doc in self.documents:
            text = (doc.title + " " + doc.content).lower()
            score = sum(text.count(t) for t in tokens)
            scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scores[:top_k] if score > 0] or self.documents[:top_k]

