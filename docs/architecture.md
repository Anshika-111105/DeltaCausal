# System Architecture: DeltaCausal Engine

<p align="center">
  <img src="images/system_architecture.jpg" alt="System Architecture: Counterfactual Business Decision & Uplift Modeling Engine" width="100%" />
</p>

```mermaid
flowchart TD
    subgraph Ingestion["1. Data Ingestion & Validation"]
        Raw["Raw Data / Criteo / Synthetic"] --> Loader["UpliftDataLoader (Streaming / Chunked)"]
        Loader --> Val["DataValidator (Schema, Leakage, Balance)"]
        Val --> ValRep["Validation Report (JSON/MD)"]
    end

    subgraph CausalDiag["2. Causal Diagnostics & Splitting"]
        Val --> Diag["CausalDiagnostics (SMD, SRM Chi2, Overlap)"]
        Diag --> Split["CausalDataSplitter (Stratified Train/Val/Test 60/20/20)"]
    end

    subgraph Modeling["3. Preprocessing & Meta-Learners"]
        Split --> Prep["CausalPreprocessor (Fit on Train Only)"]
        Prep --> Models["Meta-Learners Suite"]
        Models --> SLearner["S-Learner (Single Model)"]
        Models --> TLearner["T-Learner (Two Models)"]
        Models --> XLearner["X-Learner (Künzel et al. 2019)"]
        Models --> DRLearner["Doubly Robust Learner (AIPW)"]
        Models --> Baselines["Baselines (Propensity, Naive, Random, ATE)"]
    end

    subgraph Evaluation["4. Uplift Evaluation & Selection"]
        Models --> Eval["UpliftEvaluator (Qini Curve, AUUC, PEHE, Bootstrap CI)"]
        Eval --> BestModel["Model Selection on Val Set"]
    end

    subgraph Policy["5. Segmentation & Policy Optimization"]
        BestModel --> Seg["CustomerSegmenter (Persuadables, Sure Things, Lost Causes, Sleeping Dogs)"]
        BestModel --> Opt["PolicyOptimizer (Knapsack Selection, Budget B, Net Value)"]
        Opt --> Sens["Multidimensional Sensitivity Grid"]
    end

    subgraph Serving["6. Delivery & Decision Support"]
        Opt --> API["FastAPI REST Service (/uplift/predict, /policy/simulate)"]
        Opt --> Dash["Streamlit Analytics Dashboard (6 Pages)"]
        Eval --> Assist["Grounded Decision Assistant (RAG + Metric Tool)"]
    end
```

## Architectural Design Principles

1. **Strict Causal Framing**:
   The engine models individual potential outcomes $Y(0)$ and $Y(1)$ through Conditional Average Treatment Effects $\tau(x) = E[Y(1) - Y(0) | X=x]$. It estimates incremental effects, contrasting against traditional predictive propensity models that target organic buyers.

2. **Leakage-Free Transformation**:
   The `CausalPreprocessor` is fit exclusively on the training partition. Covariates, treatments, and outcomes are separated to prevent target leakage and post-treatment conditioning bias.

3. **Multi-Model Meta-Learner Portfolio**:
   Supports `SLearner`, `TLearner`, `XLearner` (specialized for imbalanced treatment/control allocations), and `DoublyRobustLearner` (AIPW pseudo-outcome regression).

4. **Mathematical Evaluation Suite**:
   Metrics account for treatment/control ratio scaling (`Qini Score`), ranking efficiency (`AUUC`), top-decile lift, and ground-truth CATE recovery (`PEHE`).

5. **Decision-Support RAG Layer**:
   Combines local semantic retrieval over verified causal methodology documentation with deterministic factual metric injection from actual experiment runs.

