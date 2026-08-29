#!/usr/bin/env python3
"""Build the 26-page APA-style methods and preregistration report."""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
PDF_PATH = REPORT_DIR / "OpenSky_Streaming_DPMM_APA_Report.pdf"
W, H = letter
MARGIN = 0.82 * inch
CONTENT_W = W - 2 * MARGIN
NAVY = colors.HexColor("#102435")
CYAN = colors.HexColor("#0A8FB3")
MINT = colors.HexColor("#14845F")
AMBER = colors.HexColor("#A86400")
GRAY = colors.HexColor("#526675")
PALE = colors.HexColor("#EAF4F8")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
    ]
    bold_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
    ]
    sans_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    if candidates[0].exists():
        pdfmetrics.registerFont(TTFont("BodySerif", str(candidates[0])))
        pdfmetrics.registerFont(TTFont("BodySerifBold", str(bold_candidates[0])))
        pdfmetrics.registerFont(TTFont("BodySans", str(sans_candidates[0])))
        return "BodySerif", "BodySerifBold", "BodySans"
    return "Times-Roman", "Times-Bold", "Helvetica"


SERIF, SERIF_BOLD, SANS = register_fonts()
BODY = ParagraphStyle(
    "Body", fontName=SERIF, fontSize=10.2, leading=15.2, textColor=NAVY,
    alignment=TA_LEFT, firstLineIndent=0.25 * inch, spaceAfter=8,
)
NO_INDENT = ParagraphStyle(
    "NoIndent", parent=BODY, firstLineIndent=0, spaceAfter=8,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY, firstLineIndent=0, leftIndent=0.23 * inch,
    bulletIndent=0.03 * inch, spaceAfter=4,
)
EQUATION = ParagraphStyle(
    "Equation", parent=BODY, fontName=SERIF, firstLineIndent=0,
    leftIndent=0.30 * inch, rightIndent=0.30 * inch, alignment=TA_CENTER,
    backColor=colors.HexColor("#F2F7FA"), borderColor=colors.HexColor("#B9D4E0"),
    borderWidth=0.6, borderPadding=9, spaceBefore=5, spaceAfter=11,
)
REFERENCE = ParagraphStyle(
    "Reference", parent=BODY, firstLineIndent=-0.25 * inch,
    leftIndent=0.25 * inch, fontSize=9.6, leading=14.2, spaceAfter=7,
)


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_figures() -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="#0A8FB3", lw=3)
    for deg, color in [(2, "#14845F"), (358, "#A86400")]:
        a = np.deg2rad(deg)
        ax.arrow(0, 0, .88*np.cos(a), .88*np.sin(a), width=.018,
                 color=color, length_includes_head=True)
        ax.text(1.04*np.cos(a), 1.04*np.sin(a), f"{deg} degrees", color=color,
                ha="left" if np.cos(a) > 0 else "right", va="center", weight="bold")
    ax.scatter([1], [0], s=65, color="#D3524C", zorder=5)
    ax.text(.92, .18, "coordinate cut", color="#D3524C", ha="right")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Geodesically adjacent headings remain adjacent on S1", weight="bold")
    paths["manifold"] = save_figure(fig, "observation_manifold.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    k = np.arange(2, 41)
    for alpha, color in [(0.5, "#14845F"), (2.0, "#0A8FB3"), (5.0, "#8B63C7")]:
        ax.semilogy(k, (alpha/(1+alpha))**(k-1), lw=2.4, color=color,
                    label=f"alpha = {alpha:g}")
    ax.set_xlabel("Truncation K"); ax.set_ylabel("Prior expected residual mass")
    ax.grid(alpha=.22); ax.legend(frameon=False); ax.set_title("Residual-stick approximation control", weight="bold")
    paths["truncation"] = save_figure(fig, "truncation_residual.png")

    fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.axis("off")
    labels = ["Freeze block", "Nested masks", "Blind fit", "Lock outputs", "Evaluate"]
    colors_ = ["#0A8FB3", "#14845F", "#8B63C7", "#A86400", "#D3524C"]
    xs = np.linspace(.08, .92, len(labels))
    for i, (x, label, col) in enumerate(zip(xs, labels, colors_)):
        ax.add_patch(plt.Rectangle((x-.075,.39),.15,.22,transform=ax.transAxes,
                                   facecolor=col,alpha=.14,edgecolor=col,lw=2))
        ax.text(x,.50,label,transform=ax.transAxes,ha="center",va="center",weight="bold",fontsize=8)
        if i < len(labels)-1:
            ax.annotate("",xy=(xs[i+1]-.08,.50),xytext=(x+.08,.50),xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="->",lw=2,color="#526675"))
    ax.text(.5,.80,"Evaluation labels cross the boundary only after outputs are frozen",
            transform=ax.transAxes,ha="center",color="#D3524C",weight="bold")
    ax.plot([.59,.99],[.70,.70],transform=ax.transAxes,color="#D3524C",ls="--")
    paths["pipeline"] = save_figure(fig, "blind_pipeline.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    matrix = np.full((2,3),100)
    im=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=130,aspect="auto")
    for i in range(2):
        for j in range(3): ax.text(j,i,"100 blocks",ha="center",va="center",weight="bold",color="#102435")
    ax.set_xticks(range(3),["0%","15%","30%"]); ax.set_yticks(range(2),["Streaming VI","Collapsed DP-SMC"])
    ax.set_xlabel("Controlled observation thinning"); ax.set_title("Six paired cells; 600 planned executions",weight="bold")
    fig.colorbar(im,ax=ax,label="Recorded blocks")
    paths["doe"] = save_figure(fig, "factorial_design.png")

    rng=np.random.default_rng(2026); fig,axs=plt.subplots(1,2,figsize=(7.4,3.6),sharey=True)
    for scenario,ax in zip(["stationary","shifted"],axs):
        for c,col in enumerate(["#0A8FB3","#14845F","#A86400","#8B63C7"]):
            x=np.zeros(240)
            for t in range(1,len(x)): x[t]=.55*x[t-1]+rng.normal(scale=.84)
            if scenario=="shifted" and c==3:x+=2.8
            ax.plot(x,lw=.9,color=col,label=f"Chain {c+1}")
        ax.set_title(scenario.capitalize());ax.grid(alpha=.2);ax.set_xlabel("Iteration")
    axs[0].set_ylabel("Scalar functional");axs[1].legend(frameon=False,fontsize=7)
    fig.suptitle("Independent chains are required for rank diagnostics",weight="bold")
    paths["diagnostics"] = save_figure(fig, "diagnostic_chains.png")

    fig, ax = plt.subplots(figsize=(7.3, 3.3)); ax.axis("off")
    steps=[("Mathematics","complete","#14845F"),("Software","complete","#14845F"),("Frozen benchmark","pending","#A86400"),("Operational RF","out of scope","#D3524C")]
    for i,(name,state,col) in enumerate(steps):
        x=.04+i*.24
        ax.add_patch(plt.Rectangle((x,.3),.20,.40,transform=ax.transAxes,facecolor=col,alpha=.12,edgecolor=col,lw=2))
        ax.text(x+.10,.56,name,transform=ax.transAxes,ha="center",weight="bold",fontsize=9)
        ax.text(x+.10,.40,state,transform=ax.transAxes,ha="center",color=col,fontsize=8)
        if i<3:ax.annotate("",xy=(x+.235,.5),xytext=(x+.205,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",color="#526675"))
    paths["ladder"] = save_figure(fig, "validation_ladder.png")
    return paths


PAGES = [
    {"title": "Abstract", "paragraphs": [
        "This methods and preregistration report develops a topology-aware, open-world Bayesian clustering framework for aircraft kinematic state vectors obtained from the OpenSky Network. The observation is represented on the product space R2 x S1: transformed ground speed and vertical rate occupy Euclidean coordinates, whereas true track is modeled as a circular variable. This construction removes the artificial discontinuity at north and gives the likelihood a coherent density relative to a product of Lebesgue and arc-length Haar measures.",
        "Two streaming inference strategies are specified. The first is a truncated stick-breaking variational Dirichlet-process mixture with K-1 beta variables and an explicit residual component. The second is a collapsed Dirichlet-process sequential Monte Carlo method with fully adapted allocation, systematic resampling, and windowed Gibbs rejuvenation. Both use Normal-Gamma structure for Euclidean coordinates and a von Mises factor for direction. Cumulative learning and power-forgetting tracking are treated as distinct inferential targets.",
        "The planned validation is a 2 x 3 crossed repeated-measures randomized-block experiment: two architectures, three deterministic observation-thinning levels, and 100 frozen OpenSky blocks, yielding 600 planned executions. Primary endpoints include prequential log score, external clustering agreement when evaluator-only category coverage is adequate, synchronized computation time, and cluster-count stability. The release contains executable code, invariant tests, continuous integration, an interactive laboratory, and a locked analysis protocol. It does not contain executed OpenSky benchmark results and makes no radar-pulse, emitter-identification, ELINT, SIGINT, identity, intent, or operational-performance claim.",
    ], "bullets": ["Keywords: Bayesian nonparametrics; circular statistics; Dirichlet process; online variational inference; sequential Monte Carlo; OpenSky Network."]},
    {"title": "Problem Definition and Scientific Scope", "paragraphs": [
        "The scientific problem is online discovery of latent behavioral structure in a stream whose number of clusters is unknown and potentially time varying. Conventional finite mixtures prespecify a fixed number of components and ordinary Euclidean models ignore the topology of heading. The proposed framework instead lets posterior mass distribute across a countably infinite mixture, represented computationally by either a controlled truncation or an allocation-based particle approximation.",
        "OpenSky state vectors are surveillance-derived kinematic records, not raw receiver samples. Velocity, true track, and vertical rate can support a methodological study of open-world kinematic clustering, but they cannot establish performance for radar pulse deinterleaving or emitter classification. This distinction is not semantic: the observation sigma-field does not contain pulse width, carrier frequency, angle of arrival, modulation, or receiver-calibration information needed for such claims.",
        "The central estimand is therefore comparative algorithmic robustness under controlled loss of otherwise identical public kinematic observations. The framework asks whether a fast variational approximation and a higher-cost particle comparator respond differently as observations are thinned. Identity variables are excluded from model fitting. Optional category fields are isolated for post-hoc evaluation after predictions are frozen. This separation converts an initially confounded demonstration into a falsifiable and reproducible computational-statistics experiment.",
    ]},
    {"title": "OpenSky Data Source and Acquisition Contract", "paragraphs": [
        "The ingestion layer calls the documented GET /api/states/all operation at the current OpenSky API root. The bounding box is represented by four distinct WGS84 parameters: lamin, lomin, lamax, and lomax. Earlier code used the website root, duplicated lamin, and omitted lomin; those defects could not produce the intended regional state query. The corrected client validates latitude and longitude order before any request.",
        "Anonymous access is supported for descriptive development. When credentials are available, the client uses the OAuth2 client-credentials flow and renews bearer tokens before expiry. It does not use obsolete HTTP basic authentication. API network latency is recorded as acquisition metadata and excluded from synchronized model-compute latency, because server delay is not an inferential property of either architecture.",
        "Each accepted state row must contain finite velocity, true track, and vertical rate. The feature view contains only transformed kinematics. A salted hash of the transponder identifier may be retained as a session-pseudonymous association key outside the feature matrix; it enables frozen-block alignment without making identity predictive. Category codes are stored in a separate evaluation array. The public repository does not redistribute recorded rows and ignores raw block directories by default.",
    ], "bullets": ["Source operation: https://opensky-network.org/api/states/all", "Model features: log(1 + velocity), asinh(vertical rate / 5), and heading in radians.", "Evaluation-only metadata: category code when available; missing category is coded separately."]},
    {"title": "Observation Space and Reference Measure", "paragraphs": [
        "Let Y=(X,Theta), where X belongs to R2 and Theta belongs to S1. The measurable space is equipped with the Borel sigma-field and reference measure mu=lambda_2 tensor lambda_S1, where lambda_2 is two-dimensional Lebesgue measure and lambda_S1 is arc-length Haar measure on [0,2pi). A component density is consequently defined only after both its Euclidean and circular normalizers are specified with respect to this product measure.",
        "This formulation prevents a common singularity. Mapping heading to (cos Theta, sin Theta) is useful for visualization and moment calculation, but the image lies on a one-dimensional subset of R2. A nonsingular Gaussian density on the ambient cosine-sine plane places probability off that circle. Multiplying it by an indicator that selects the circle does not produce a normalized density relative to two-dimensional Lebesgue measure, because the circle has Lebesgue measure zero.",
        "The corrected kernel is a Euclidean normal or Student predictive term multiplied by a von Mises directional term. The likelihood integrates to one under mu. Posterior component parameters live in a parameter space, not in the observation manifold itself. Accordingly, the Dirichlet-process base measure is placed on component locations, precisions, directions, and concentrations rather than directly on aircraft observations.",
    ], "equation": "p(x, theta | eta) = p_E(x | eta_E) p_C(theta | eta_C); integral p d(lambda_2 tensor lambda_S1) = 1."},
    {"title": "Circular Geometry at the Coordinate Cut", "paragraphs": [
        "Headings just below 360 degrees and just above 0 degrees are physically adjacent. Their Euclidean difference in degrees is nearly 360, but their geodesic separation is small. The circular difference is wrap(theta_1-theta_2)=atan2(sin(theta_1-theta_2),cos(theta_1-theta_2)), and the geodesic distance is its absolute value. This distance is invariant to integer multiples of 2pi.",
        "The von Mises distribution supplies a smooth circular density p(theta|psi,kappa)=exp{kappa cos(theta-psi)}/[2pi I0(kappa)]. Its sufficient statistics are cos theta and sin theta. These statistics are not treated as an unconstrained bivariate observation; instead, their resultant vector updates the posterior directional mean and concentration. The distinction preserves topology and the correct base measure.",
        "Figure 1 shows headings of 2 and 358 degrees. On the circle, the two arrows are separated by four degrees and lie on opposite sides of the arbitrary coordinate label. A topology-aware model therefore assigns them similar directional likelihood when its concentration permits. This behavior is essential for any streaming process that crosses north frequently.",
    ], "figure": "manifold", "caption": "Figure 1. Circular representation of headings adjacent across the 0/360-degree coordinate cut."},
    {"title": "Product-Space Component Model", "paragraphs": [
        "For component k, the Euclidean coordinates have conditionally independent Gaussian factors with unknown means and precisions. Each coordinate receives a Normal-Gamma prior. Integrating those parameters yields a Student predictive density for collapsed SMC. In the variational approximation, expected log precision and expected quadratic error enter the responsibility update. Using two scalar Normal-Gamma factors is a deliberate diagonal-covariance baseline; a Normal-Wishart extension is identified as a sensitivity analysis rather than silently implied.",
        "The directional coordinate uses a von Mises factor with component direction psi_k and concentration kappa_k. Directional sufficient statistics are the weighted cosine and sine sums. The posterior direction is the argument of their resultant vector. Concentration is estimated through the standard mean-resultant map A1(kappa)=I1(kappa)/I0(kappa), using a stable approximation and bounded numerical domain.",
        "The product kernel assumes conditional independence between transformed speed, transformed vertical rate, and heading given the component. This is testable and may be inadequate for coordinated turns or climb regimes. The preregistration therefore includes posterior predictive checks stratified by inferred cluster and comparison with an optional coupled directional-linear kernel. The baseline remains valuable because every approximation and normalization is explicit.",
    ], "equation": "p_k(y) = NormalGammaPredictive(x_1) NormalGammaPredictive(x_2) vonMises(theta | psi_k, kappa_k)."},
    {"title": "Dirichlet-Process Prior and Open-World Semantics", "paragraphs": [
        "A Dirichlet process G distributed as DP(alpha,G0) is a random probability measure on the component-parameter space. Sethuraman's construction writes G as a countable sum of atoms with stick-breaking weights. Conditional observations select atoms, and repeated selections induce a random partition. The concentration alpha governs prior partition growth but does not directly equal the expected number of occupied clusters for a finite sample.",
        "Open-world terminology is used narrowly. New posterior components may become occupied as novel kinematic patterns appear; no semantic aircraft class is automatically discovered. Component labels are exchangeable and subject to label switching. External category agreement is therefore computed with permutation-invariant measures. Claims about physical meaning require a separate, blinded interpretation stage and are not created by the nonparametric prior.",
        "The base measure combines Normal-Gamma terms for Euclidean component parameters and a circular prior for direction. Hyperparameters are fixed before the primary benchmark or estimated in a training-only calibration stage. Sensitivity analyses vary alpha and the Euclidean/circular scales on the same frozen blocks. Hyperparameter selection may not use the held-out test categories.",
    ]},
    {"title": "Finite Truncation and Residual Mass", "paragraphs": [
        "The variational method approximates the infinite construction with K components. A valid residual-stick truncation uses K-1 independent beta variables. For k<K, pi_k=V_k product_{j<k}(1-V_j); the final component receives pi_K=product_{j<K}(1-V_j). The weights then sum to one for every draw. Drawing K beta variables while retaining only K weights discards probability mass and changes the stated approximation.",
        "Under the prior V_k~Beta(1,alpha), the expected residual mass is [alpha/(1+alpha)]^(K-1). This quantity is a prior approximation diagnostic. It does not establish that posterior truncation error is small when data push mass toward late components. The implementation therefore also reports posterior residual expected weight, occupied-component count, and boundary occupancy.",
        "Figure 2 shows the prior expected remainder as a function of K at three concentration levels. Higher alpha demands a larger truncation for the same residual target. A preregistered rule should choose K so the worst-case planned alpha has a sufficiently small prior remainder, then verify that posterior mass does not accumulate at component K. Increasing K after observing test outcomes would compromise the confirmatory contrast.",
    ], "figure": "truncation", "caption": "Figure 2. Prior expected residual mass under valid K-component stick-breaking truncation."},
    {"title": "Streaming Variational Family", "paragraphs": [
        "The variational family factorizes over allocation variables, K-1 stick variables, Euclidean component factors, and circular component factors. Coordinate ascent maximizes an evidence lower bound within this restricted family. The allocation responsibility is proportional to the exponentiated sum of expected log mixture weight, expected Euclidean log likelihood, and circular log likelihood. Log-sum-exp normalization prevents numerical underflow.",
        "The implementation exposes two update modes. In cumulative mode, batch sufficient statistics are added to all earlier sufficient statistics, matching sequential processing of a fixed-data posterior approximation. In power-forgetting mode, stored sufficient statistics are multiplied by rho in (0,1) before new evidence is added. The second target intentionally discounts the past and is suitable for tracking drift, but it is not ordinary Bayesian conditioning and should not be described with static posterior-consistency claims.",
        "An initialization-only symmetry break is permitted because identical component factors otherwise yield identical responsibilities. Bootstrap centers are used only to begin coordinate optimization; they are not retained as pseudo-observations in the posterior sufficient statistics. This correction prevents initial data from being counted twice. Iteration stops at a fixed cap or when the maximum responsibility change is below a prespecified tolerance.",
    ]},
    {"title": "Variational Coordinate Updates", "paragraphs": [
        "Let N_k be the sum of responsibilities for component k. For k<K, the beta parameters update to gamma_{k1}=1+N_k and gamma_{k2}=alpha+sum_{j>k}N_j. Expected log weights combine digamma expectations for log V_k and preceding log(1-V_j). The final residual component uses only the preceding log-complements; it has no independent beta factor.",
        "For Euclidean coordinate d, weighted sums and sums of squares update Normal-Gamma parameters. Careful central-moment algebra is required when cumulative or discounted sufficient statistics are merged. Directly averaging successive means without preserving effective counts and second moments gives order-dependent variances. The implementation stores count, first moment, and second moment sufficient statistics before converting them into posterior parameters.",
        "Circular updates accumulate C_k=sum_n r_nk cos theta_n and S_k=sum_n r_nk sin theta_n. The posterior resultant combines these data statistics with the prior resultant. The updated direction is atan2(S,C) modulo 2pi. The concentration is obtained from the resultant length through an approximation to the inverse of A1. Every operation is continuous across the coordinate cut except at the unidentifiable zero-resultant point, where the prior supplies a defined direction.",
    ], "equation": "log r_nk = const + E[log pi_k] + E[log p_E(x_n | eta_k)] + kappa_k cos(theta_n - psi_k)."},
    {"title": "Collapsed Dirichlet-Process SMC", "paragraphs": [
        "The comparator represents a posterior over partitions with a weighted particle cloud. For each incoming observation, particle-specific existing-cluster probabilities are proportional to occupied counts times collapsed predictive densities; the new-cluster probability is proportional to alpha times the base predictive density. Because the allocation proposal is normalized from these same predictive factors, it is fully adapted for the local assignment and the particle weight receives the corresponding predictive normalizer.",
        "Euclidean predictive factors are Student distributions obtained by integrating Normal-Gamma component parameters. The circular predictive approximation integrates the directional location under its conjugate resultant representation while treating concentration according to the specified base model. This is a better-defined comparator than applying a Euclidean Gaussian to a sine-cosine embedding, although it remains an approximation when concentration uncertainty is simplified.",
        "Particles contain only current allocation and sufficient-statistic state. True categories and readable identifiers are absent. The number of occupied clusters varies by particle. Summaries of cluster count therefore integrate particle weights and must include uncertainty rather than report one maximum-weight partition as if it were the unique answer.",
    ]},
    {"title": "Resampling and Rejuvenation", "paragraphs": [
        "Sequential importance weights degenerate as streams lengthen. Particle effective sample size is computed as one divided by the sum of squared normalized weights. When it falls below a prespecified fraction of particle count, systematic resampling replaces the weighted cloud with equally weighted ancestral copies. The threshold is an algorithmic tuning parameter fixed before the primary experiment.",
        "Resampling alone duplicates particles and can reduce partition diversity. A Gibbs rejuvenation sweep revisits assignments in a recent time window. Each selected observation is removed from its cluster sufficient statistics, reassigned using the collapsed conditional, and restored. Empty clusters are deleted. Restricting rejuvenation to a window bounds streaming cost but changes the approximation; window width is therefore logged and subjected to sensitivity analysis.",
        "SMC particle ESS is not MCMC ESS. It measures concentration of normalized importance weights at one sequential step. MCMC ESS estimates information loss from serial correlation in a Markov trace. The software and dashboard label these separately. Conflating them could make a degenerate particle cloud appear well mixed or make a short rejuvenation trace appear to have particle diversity it does not possess.",
    ], "figure": "pipeline", "caption": "Figure 3. Frozen-block processing and the boundary between blind fitting and unlocked evaluation."},
    {"title": "Streaming, Drift, and Inferential Targets", "paragraphs": [
        "A stationary exchangeable mixture is not automatically a state-space model. The cumulative variational estimator targets a posterior approximation for the accumulated observations under exchangeability. The power-forgetting estimator targets a time-local pseudo-posterior in which older likelihood contributions are geometrically discounted. The collapsed particle method in this release evolves a partition sequentially but does not posit continuous latent aircraft dynamics.",
        "This distinction matters for asymptotic language. Under cumulative sampling and suitable identifiability and prior-support conditions, one may study posterior concentration or predictive consistency. With fixed forgetting rho<1, the effective historical sample size remains bounded in order, and classical static posterior concentration is not the appropriate limit. Evaluation should instead study tracking error, adaptation lag, and stability under controlled drift.",
        "A future dependent Dirichlet process or distance-dependent Chinese restaurant process could introduce explicit temporal dependence. Such an extension would require a transition law for component parameters or links, a precise filtration, and an observation process indexed longitudinally by pseudonymous track keys. It should not be implied by the present exchangeable-within-block model.",
    ]},
    {"title": "Blindness, Association, and Leakage Control", "paragraphs": [
        "Strict blind fitting means that the model consumes only transformed kinematics. It does not mean that all association information must be destroyed before experimental blocking. A session-pseudonymous key can define which records belong to a longitudinal unit while remaining outside the predictor sigma-field. Deleting association before defining the sampling unit would turn the stream into unrelated cross-sections and invalidate repeated-measures statements.",
        "The software returns an OpenSkySnapshot with separate model_view, track_keys, and categories fields. The engine receives model_view only. The evaluator is a separate stage that reads locked allocations or predictive distributions, joins evaluator-only categories through a manifest, and computes permutation-invariant metrics. This API-level separation is auditable and more reliable than a prose promise not to use labels.",
        "Leakage can also occur through preprocessing or hyperparameter tuning. Feature transformations are fixed analytically, so they do not use held-out categories. If data-dependent scaling is later introduced, scale parameters must be estimated within training blocks and applied to validation or test blocks without refitting. All design decisions triggered by test outcomes must be labeled exploratory.",
    ]},
    {"title": "Observation Thinning and Missingness", "paragraphs": [
        "The experimental dropout factor is controlled observation thinning, not a claim about adversarial network behavior. Within block b, one deterministic uniform variate U_bi is assigned to observation i. The record is retained at dropout d exactly when U_bi>=d. Consequently, the 30% retained set is contained in the 15% retained set, which is contained in the complete set. Every architecture receives the identical retained indices.",
        "Nested masks create strong pairing and reduce random variation in thinning contrasts. Their reproducibility follows from a block-specific seed and observation order. Because masks are imposed after acquisition, they do not alter API network latency. Compute latency is synchronized around inference only; acquisition time is reported separately.",
        "Natural missingness requires different handling. Rows without required numeric fields are ineligible before random thinning. Complete API outages can remove an entire planned block and may depend on time or service conditions. Exclusion counts, timestamps, and reasons are preserved in a provenance ledger. The primary analysis conditions on eligible recorded blocks; sensitivity analysis uses inverse-probability or pattern stratification only if missingness diagnostics justify it.",
    ]},
    {"title": "Factorial Randomized-Block Design", "paragraphs": [
        "The design has six treatment cells rather than eighteen. Factor A has two levels: streaming variational DPMM and collapsed DP-SMC. Factor B has three levels: 0%, 15%, and 30% controlled thinning. One hundred independently scheduled acquisition blocks form the blocking factor. All six cells are evaluated within every block, yielding 600 planned method executions.",
        "Cell execution order is randomized within block after data and masks are frozen. This prevents systematic thermal, caching, or background-load order from aligning with one method. Warm-up runs are conducted before timed repetitions and discarded according to a fixed rule. GPU synchronization brackets each timed method call when CUDA is active. Peak memory, hardware, package versions, and thread settings are recorded.",
        "Figure 4 depicts the complete cell structure. The unit of statistical independence for population inference is the recorded block, not each aircraft row and not each algorithmic particle. Treating rows or particles as independent replicates would create pseudoreplication and artificially narrow uncertainty intervals.",
    ], "figure": "doe", "caption": "Figure 4. Crossed 2 x 3 repeated-measures randomized-block design."},
    {"title": "Primary and Secondary Endpoints", "paragraphs": [
        "The primary predictive endpoint is average prequential log score on the held-out portion of each frozen block. It uses each method's predictive distribution and remains defined when category labels are absent. When category coverage passes the preregistered threshold, adjusted mutual information or normalized mutual information provides a secondary external partition-agreement endpoint. Both are invariant to cluster-label permutations.",
        "Compute endpoints include synchronized model latency per eligible observation and peak memory. Network time is excluded from model latency. Structural endpoints include posterior expected occupied-cluster count, between-replay cluster-count variation, posterior residual-stick mass for VI, and particle ESS trajectories for SMC. Diagnostic failure is reported as an outcome, not silently filtered.",
        "No single metric proves superior open-world inference. High category agreement can reward taxonomy alignment even when prequential calibration is poor; good predictive score can coexist with unstable semantic partitions. The analysis therefore reports an endpoint family, confidence intervals, and trade-off plots. A composite winner is not created after seeing the data.",
    ], "bullets": ["Primary: prequential log score and synchronized compute latency.", "Conditional secondary: label agreement when blinded category coverage is adequate.", "Algorithmic health: VI residual mass, SMC particle ESS, numerical failures, and cluster-count stability."]},
    {"title": "Mixed-Effects Contrast Model", "paragraphs": [
        "For each endpoint, the primary model includes an architecture contrast A, orthogonal linear and quadratic contrasts L and Q for thinning, and architecture-by-trend interactions. A block random intercept accounts for shared traffic composition. A random architecture slope is included when supported by the number of blocks and a nonsingular covariance estimate. The maximal stable random-effects structure is recorded without using p-value-driven stepwise selection.",
        "Latency and strictly positive error-like outcomes are analyzed on the log scale when residual diagnostics support that transformation. Proportions may use a link-scale generalized mixed model or block-level transformation. Degrees of freedom use a small-sample correction; a block bootstrap supplies a robustness analysis. Cell-specific residual variances are allowed if the three thinning levels exhibit materially different dispersion.",
        "The architecture main effect estimates an average over the coded thinning levels. The architecture-by-linear-thinning interaction tests whether method separation changes monotonically as records are removed; the quadratic interaction tests curvature. These contrasts are more interpretable than a collection of unstructured pairwise tests. Prespecified cell contrasts are nevertheless reported with simultaneous uncertainty intervals.",
    ], "equation": "Y_mdb = beta_0 + beta_A A_m + beta_L L_d + beta_Q Q_d + beta_AL A_m L_d + beta_AQ A_m Q_d + b_0b + b_1b A_m + epsilon_mdb."},
    {"title": "Multiplicity, Uncertainty, and Decision Rules", "paragraphs": [
        "The analysis emphasizes effect estimates and uncertainty rather than binary declarations. For each endpoint family, primary fixed-effect tests are adjusted by the Holm procedure. Confidence intervals for prespecified cell contrasts are reported on both model and interpretable response scales. Standardized effects use block-level variation rather than row-level variation.",
        "A method is not declared globally superior unless it improves the primary predictive endpoint without violating a preregistered latency or failure-rate constraint. If one method is more accurate and slower, the result is a Pareto trade-off rather than a win. Equivalence or noninferiority margins, if used, must be set before the frozen test execution and justified in operational units relevant to the software setting, not reverse-engineered from observed effects.",
        "Sensitivity analyses vary DP concentration, truncation K, SMC particle count, rejuvenation window, and forgetting factor. They are organized around the same blocks and masks. Confirmatory conclusions come from the locked configuration. Any alternative discovered after inspection is clearly labeled exploratory and accompanied by the complete multiverse or a documented selection path.",
    ]},
    {"title": "Convergence and Monte Carlo Diagnostics", "paragraphs": [
        "Rank-normalized split R-hat and bulk effective sample size are reserved for genuine Markov chains. Each sensitivity chain must be independently initialized and simulated with its own random stream. The diagnostic pipeline splits chains, pools ranks, maps ranks to normal scores, and computes both bulk and folded scale checks. The reported R-hat is the maximum of the two. Bulk ESS uses an autocorrelation sequence with an initial-positive, monotone paired truncation.",
        "Synthetic Figure 5 demonstrates why independent chains matter. Four stationary traces overlap, whereas a shifted fourth chain produces clear between-chain dispersion. Adding noise to one saved chain would not probe overdispersed initial states and cannot support a convergence conclusion. This report does not present the synthetic diagnostic values as empirical evidence.",
        "For SMC, particle ESS is tracked at every update and summarized by minimum, median, and resampling frequency. Genealogical diversity and occupied-partition diversity supplement weight ESS because resampled duplicates can have equal weights while representing few ancestors. For VI, the evidence lower bound or maximum responsibility change, iteration count, residual-stick weight, and nonfinite-update count are retained.",
    ], "figure": "diagnostics", "caption": "Figure 5. Synthetic diagnostic illustration; these traces are not OpenSky outcomes."},
    {"title": "Latency and Streaming-Systems Measurement", "paragraphs": [
        "Wall-clock timing is meaningful only when its boundaries are explicit. The benchmark separates network acquisition, parsing, host-to-device transfer, model update, and evaluation. The primary compute endpoint brackets only the inference call. CUDA synchronization occurs immediately before the start time and immediately after the method returns; otherwise asynchronous kernels would bias measured durations downward.",
        "Every block-method-thinning cell receives warm-up executions outside the timed sample. Execution order is randomized within block. The environment manifest records processor, accelerator, operating system, Python and package versions, precision, number of particles, truncation, batch size, and active threads. Latency distributions are expected to be skewed, so geometric means, medians, high quantiles, and log-scale mixed models are more informative than an arithmetic mean alone.",
        "Real-time feasibility is not inferred from average latency alone. A deadline-miss probability must be evaluated relative to a declared update budget. The public OpenSky API cadence and network policy may differ from a production feed, so this study can establish computational throughput in its documented environment but cannot establish operational surveillance deadlines.",
    ]},
    {"title": "Software Verification Strategy", "paragraphs": [
        "Verification targets mathematical invariants rather than only successful execution. Unit tests assert that expected truncated weights include the residual and sum to one, responsibilities normalize for every observation, circular wrapping is periodic, nested masks are deterministic and ordered by thinning, particle ESS lies between one and particle count, and rank diagnostics react to a shifted chain. The API client test verifies the current endpoint and four distinct bounding-box keys.",
        "A deterministic smoke benchmark simulates three circular-linear clusters, fits both inference engines, and checks finite outputs. It is a software test rather than a performance result. Continuous integration installs a CPU PyTorch build, executes the tests, runs the smoke benchmark, rebuilds the website and report, checks the machine-readable empirical-status flag, scans required claim-boundary phrases, and verifies the report has exactly 26 pages.",
        "Reproducibility also requires negative controls. The code rejects invalid geographic boxes and incomplete authentication pairs. Raw recorded blocks, credentials, model checkpoints, and result directories are ignored by Git. Publication artifacts are generated from committed scripts. The report and site repeat the empirical-status boundary so that a later partial execution cannot silently appear as a complete benchmark.",
    ]},
    {"title": "Preregistered Execution Procedure", "paragraphs": [
        "Before collection, freeze the geographic region, acquisition schedule, minimum eligible row count, number and duration of blocks, random seeds, hyperparameters, outcome definitions, and exclusion rules. Record software and hardware manifests. Acquire each OpenSky block once, verify its checksum, and avoid rerunning a treatment on a newer traffic snapshot. A failed acquisition is logged before method assignment.",
        "For each accepted block, generate one nested mask vector and materialize the three retained index sets. Randomize six cell execution orders. Run VI and SMC with their locked seeds, capture predictions and health metrics, and write append-only cell records. Freeze all allocations and predictive distributions before joining category labels. The evaluator then computes block-level endpoints and a coverage report.",
        "After all 600 executions or the preregistered stopping condition, verify completeness, hash the analysis dataset, and run a single primary-analysis command. Export aggregate results, design matrix, model specification, diagnostics, and a disclosure of deviations. Do not commit readable identifiers or raw state rows. Update the machine-readable status from pending only when the complete locked analysis and provenance files exist.",
    ], "bullets": ["Freeze -> record -> checksum -> mask -> randomize -> fit -> lock -> evaluate -> analyze.", "The same frozen block is replayed in all six treatment cells.", "Raw data and credentials remain outside the public repository."]},
    {"title": "Interpretation Framework", "paragraphs": [
        "A favorable architecture contrast would mean that, over the sampled OpenSky blocks and documented hardware, one approximation achieved a better prespecified endpoint at the tested thinning levels. It would not prove recovery of true behavioral classes, because category codes are imperfect proxies and the mixture components are statistical partitions. External agreement should be interpreted alongside predictive score and cluster stability.",
        "A thinning interaction would quantify robustness to controlled record removal. It would not identify the effect of natural API outages unless the missingness mechanism were exchangeable with the imposed mask. Similarly, synchronized compute time supports a software-performance conclusion for the benchmark environment, not a guarantee for live systems with different hardware, batching, network policy, or update deadlines.",
        "Null or imprecise contrasts remain informative. They may show that the more expensive particle approximation does not produce detectable predictive improvement at the planned block count, that between-block heterogeneity dominates algorithm choice, or that evaluator-label coverage is insufficient. Conclusions must follow the uncertainty intervals and diagnostics rather than the desired narrative.",
    ]},
    {"title": "Validation Ladder and Claim Governance", "paragraphs": [
        "The validation ladder prevents an engineering artifact from being narrated as an operational finding. Mathematics and software verification form the first two gates and are complete in this release. The third gate is the frozen public-data benchmark and remains pending. A fourth gate involving raw radio-frequency receiver data, synchronized ground truth, hardware timing, and independent replication is outside the scope of OpenSky state vectors.",
        "Figure 6 encodes that boundary visually. Completion of a lower gate does not imply completion of a higher gate. Correct normalization does not prove predictive usefulness; predictive usefulness on public kinematics does not prove radar-emitter performance; and laboratory receiver results would not alone establish field effectiveness. Each claim must name its data-generating process and experimental unit.",
        "Repository governance implements this logic through data/project_status.json, visible dashboard language, a claim-boundary document, and CI assertions. Future maintainers should update status only through a reviewable commit containing aggregate results, a provenance manifest, and executed diagnostic records. Marketing language such as operationally validated is prohibited unless a distinct authorized campaign directly supports it.",
    ], "figure": "ladder", "caption": "Figure 6. Validation gates and the explicit boundary of this OpenSky methodology project."},
    {"title": "Limitations", "paragraphs": [
        "OpenSky coverage is heterogeneous across geography, altitude, receiver density, and time. State vectors may reflect filtering and multilateration rather than raw sensor measurements. The dataset is therefore not a random sample of all aircraft activity. A 100-block study estimates behavior over its declared sampling frame; generalization beyond that frame requires replication across regions and seasons.",
        "The baseline conditional-independence kernel can miss dependence between speed, climb rate, and heading. The DP mixture clusters observations rather than fitting a full aircraft dynamic model. Category codes, when present, are coarse, incomplete, and not a definitive behavioral truth. External agreement can be low because the categories and latent kinematic regimes answer different questions.",
        "Truncated VI has approximation bias and can understate posterior uncertainty. Collapsed DP-SMC can suffer particle impoverishment and order dependence, especially with a limited rejuvenation window. Both methods require sensitivity analysis. The current repository provides no complete empirical OpenSky result, so it cannot yet estimate method differences or thinning degradation.",
    ]},
    {"title": "Future Extensions", "paragraphs": [
        "A longitudinal extension could use pseudonymous association keys to model within-aircraft evolution through a dependent Dirichlet process, recurrent transition kernel, or distance-dependent Chinese restaurant process. Such a model should define a filtration, transition dependence, birth and death processes, and a likelihood for irregular observation times. The comparison with the exchangeable baseline would be preregistered.",
        "A richer emission kernel could use a full Normal-Wishart Euclidean block, projected-normal directional-linear dependence, or normalizing flows defined on product manifolds. Flexible kernels increase computation and identifiability risk; calibration and posterior predictive checking must accompany them. Approximation quality may be evaluated against longer MCMC or SMC runs on small frozen blocks.",
        "A separate receiver-data campaign would require lawful raw I/Q or pulse data, synchronized transmitter truth, channel and receiver calibration, train-test separation by hardware/session, and hardware-in-the-loop timing. Those data would constitute a new project with a new claim boundary. They cannot be inferred from OpenSky kinematics or created by renaming state-vector clusters as emitters.",
    ]},
    {"title": "Reproducibility and Data Availability", "paragraphs": [
        "The repository contains the inference engine, tests, machine-readable design, statistical model, protocol, report builder, site builder, continuous-integration workflows, license, and citation metadata. The report is regenerated with python scripts/build_report.py; the standalone site is regenerated with python scripts/build_site.py. The deterministic smoke benchmark runs with python -m opensky_streaming_dpmm.engine.",
        "OpenSky data are obtained from the source API and are not redistributed here. Users must comply with current OpenSky terms, rate limits, attribution requirements, and applicable privacy law. Credentials are provided only through environment variables. Raw and recorded-block paths are ignored. Aggregate, non-identifying results may be released after the complete locked benchmark is executed.",
        "The public artifact is designed for audit. Every dashboard number is either a structural design quantity, a software version fact, or clearly labeled synthetic illustration. The empirical status flag is false at release 1.0. No inferential estimate is backfilled with simulated or invented numbers.",
    ], "bullets": ["Code repository: https://github.com/salemmorelli1/OpenSky-Streaming-DPMM", "Interactive site: https://salemmorelli1.github.io/OpenSky-Streaming-DPMM/", "OpenSky documentation: https://openskynetwork.github.io/opensky-api/rest.html"]},
    {"title": "References", "references": [
        "Antoniak, C. E. (1974). Mixtures of Dirichlet processes with applications to Bayesian nonparametric problems. Annals of Statistics, 2(6), 1152-1174.",
        "Blei, D. M., & Jordan, M. I. (2006). Variational inference for Dirichlet process mixtures. Bayesian Analysis, 1(1), 121-144. https://doi.org/10.1214/06-BA104",
        "Blackwell, D., & MacQueen, J. B. (1973). Ferguson distributions via Polya urn schemes. Annals of Statistics, 1(2), 353-355.",
        "Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems. Annals of Statistics, 1(2), 209-230.",
        "Fisher, N. I. (1993). Statistical analysis of circular data. Cambridge University Press.",
        "Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis (3rd ed.). CRC Press.",
        "Mardia, K. V., & Jupp, P. E. (2000). Directional statistics. Wiley.",
        "OpenSky Network. (2026). REST API documentation. https://openskynetwork.github.io/opensky-api/rest.html",
        "Sethuraman, J. (1994). A constructive definition of Dirichlet priors. Statistica Sinica, 4, 639-650.",
        "Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Buerkner, P. C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. Bayesian Analysis, 16(2), 667-718. https://doi.org/10.1214/20-BA1221",
        "West, M. (1992). Hyperparameter estimation in Dirichlet process mixture models. Duke University technical report.",
    ], "paragraphs": ["References are limited to sources directly used to define the probability model, computational approximations, diagnostics, and data interface. API behavior should be verified against the current OpenSky documentation at execution time because authentication and rate-limit policies can change."]},
    {"title": "Appendix A: Algorithmic Pseudocode", "paragraphs": [
        "Streaming variational DPMM: initialize K component factors without retaining bootstrap pseudo-counts. For each frozen batch, compute expected log residual-stick weights. Alternate responsibility and component-factor updates until tolerance or iteration cap. Merge sufficient statistics cumulatively or through the prespecified forgetting operator. Return normalized responsibilities, predictive score, active-component count, residual weight, iteration count, and synchronized latency.",
        "Collapsed DP-SMC: initialize P partition particles. For each observation and each particle, compute collapsed predictive probabilities for occupied clusters and a new cluster. Sample one fully adapted allocation and update the incremental log weight by the predictive normalizer. Normalize weights; if particle ESS crosses the threshold, perform systematic resampling. Apply a fixed number of Gibbs reassignment sweeps to the recent window. Return weighted partition summaries and health traces.",
        "Frozen experiment: record each eligible block once; checksum and seal it. Construct nested deterministic masks. Randomize six cell orders. Execute the two inference methods under three masks with fixed configurations. Freeze outputs. Join evaluator-only labels. Compute block-level endpoints. Fit the prespecified mixed-effects contrasts and publish aggregate results with deviations and diagnostics.",
    ]},
    {"title": "Appendix B: Reporting Checklist", "paragraphs": [
        "A complete empirical release should report the acquisition interval, geographic box, eligibility criteria, number of attempted and accepted blocks, natural missingness, category coverage, transformation constants, DP concentration, truncation, posterior residual mass, SMC particle count, resampling threshold, rejuvenation window, VI convergence tolerance, execution randomization, warm-up policy, precision, hardware, package lock, and every analysis deviation.",
        "For each endpoint, report all six cell summaries, paired architecture contrasts at each thinning level, architecture main effect, linear and quadratic thinning effects, interactions, uncertainty intervals, multiplicity adjustment, residual diagnostics, random-effects structure, and number of contributing blocks. Do not report row-level standard errors as though rows were independent blocks.",
        "The release conclusion must restate the observation domain. If the data remain OpenSky state vectors, the conclusion is limited to public kinematic clustering. The exact phrases radar pulse, emitter identification, ELINT, SIGINT, threat, and operational validation should appear only in explicit exclusions unless separately authorized evidence is added. This checklist is part of the statistical design, not a cosmetic disclaimer.",
    ]},
]

# The full drafting bank above deliberately retains modular material for later
# revisions. The publication artifact selects 25 body pages plus the title page.
_OMIT_FROM_RELEASE = {5, 12, 18, 23, 26, 29}
PAGES = [page for index, page in enumerate(PAGES) if index not in _OMIT_FROM_RELEASE]


def draw_header_footer(c: canvas.Canvas, page_num: int) -> None:
    c.setStrokeColor(colors.HexColor("#B9D4E0")); c.setLineWidth(.45)
    c.line(MARGIN, H - .54*inch, W - MARGIN, H - .54*inch)
    c.setFont(SANS, 7.6); c.setFillColor(GRAY)
    c.drawString(MARGIN, H - .42*inch, "TOPOLOGY-AWARE OPENSKY STREAMING BAYESIAN CLUSTERING")
    c.drawRightString(W-MARGIN, H-.42*inch, str(page_num))
    c.line(MARGIN, .55*inch, W-MARGIN, .55*inch)
    c.drawString(MARGIN, .38*inch, "Methods and preregistration report · empirical benchmark pending")


def paragraph(c: canvas.Canvas, text: str, y: float, style=BODY) -> float:
    p = Paragraph(escape(text), style)
    _, h = p.wrap(CONTENT_W, y - .72*inch)
    p.drawOn(c, MARGIN, y-h)
    return y-h-style.spaceAfter


def draw_title_page(c: canvas.Canvas) -> None:
    c.setFillColor(NAVY); c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColor(CYAN); c.rect(MARGIN,H-1.43*inch,1.15*inch,.08*inch,fill=1,stroke=0)
    c.setFont(SANS,9); c.drawString(MARGIN,H-1.12*inch,"COMPUTATIONAL STATISTICS · METHODS AND PREREGISTRATION")
    title="Topology-Aware Streaming Bayesian Nonparametric Clustering of OpenSky Aircraft Kinematics"
    style=ParagraphStyle("Title",fontName=SERIF_BOLD,fontSize=26,leading=31,textColor=colors.white,alignment=TA_LEFT)
    p=Paragraph(title,style);_,h=p.wrap(CONTENT_W,H);p.drawOn(c,MARGIN,H-1.75*inch-h)
    y=H-1.95*inch-h
    subtitle=Paragraph(
        "Measure-Theoretic Model, Computational Algorithms, and Frozen Validation Design",
        ParagraphStyle("Subtitle",fontName=SERIF,fontSize=13,leading=18,
                       textColor=colors.HexColor("#CFE5F0"),alignment=TA_LEFT),
    )
    _,subtitle_h=subtitle.wrap(CONTENT_W,.6*inch);subtitle.drawOn(c,MARGIN,y-subtitle_h)
    y-=subtitle_h+0.72*inch
    c.setFont(SERIF_BOLD,12);c.setFillColor(colors.white);c.drawString(MARGIN,y,"Salem Morelli")
    y-=.30*inch;c.setFont(SERIF,11);c.setFillColor(colors.HexColor("#CFE5F0"));c.drawString(MARGIN,y,"Independent computational-statistics research")
    y-=.30*inch;c.drawString(MARGIN,y,"August 29, 2026")
    c.setFillColor(colors.HexColor("#0B3148"));c.roundRect(MARGIN,1.25*inch,CONTENT_W,1.20*inch,10,fill=1,stroke=0)
    c.setFillColor(colors.HexColor("#D9F5FF"));c.setFont(SANS,9.2)
    claim="Claim boundary: OpenSky supplies aircraft state vectors, not raw radar pulses. This report presents a methodology and preregistration. The 600-run empirical benchmark is pending; no operational SIGINT claim is made."
    p=Paragraph(claim,ParagraphStyle("Claim",fontName=SANS,fontSize=9.4,leading=14,textColor=colors.HexColor("#D9F5FF")))
    _,h=p.wrap(CONTENT_W-.4*inch,.9*inch);p.drawOn(c,MARGIN+.2*inch,1.38*inch)
    c.setFont(SANS,8);c.setFillColor(colors.HexColor("#90AEC2"));c.drawRightString(W-MARGIN,.55*inch,"1")


def draw_standard_page(c: canvas.Canvas, page_num: int, item: dict, figures: dict[str, Path]) -> None:
    draw_header_footer(c,page_num)
    y=H-.82*inch
    c.setFont(SERIF_BOLD,16);c.setFillColor(NAVY);c.drawString(MARGIN,y,item["title"]);y-=.33*inch
    c.setFillColor(CYAN);c.rect(MARGIN,y+.10*inch,.72*inch,.04*inch,fill=1,stroke=0);y-=.08*inch
    for text in item.get("paragraphs",[]): y=paragraph(c,text,y)
    if "equation" in item: y=paragraph(c,item["equation"],y,EQUATION)
    for bullet in item.get("bullets",[]): y=paragraph(c,"• "+bullet,y,BULLET)
    if "references" in item:
        for ref in item["references"]: y=paragraph(c,ref,y,REFERENCE)
    if "figure" in item:
        path=figures[item["figure"]]
        available=max(1.7*inch,min(3.15*inch,y-1.05*inch))
        img_w=min(CONTENT_W,7.2*inch);img_h=available
        c.drawImage(str(path),MARGIN,y-img_h,width=img_w,height=img_h,preserveAspectRatio=True,anchor="c",mask="auto")
        y-=img_h+.06*inch
        cap=Paragraph(item["caption"],ParagraphStyle("Caption",parent=NO_INDENT,fontSize=8.3,leading=11,textColor=GRAY))
        _,h=cap.wrap(CONTENT_W,.5*inch);cap.drawOn(c,MARGIN,y-h)
    if y < .70*inch:
        raise RuntimeError(f"Page {page_num} content overflow: y={y:.1f}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    figures=build_figures()
    c=canvas.Canvas(str(PDF_PATH),pagesize=letter,pageCompression=1)
    c.setTitle("Topology-Aware Streaming Bayesian Nonparametric Clustering of OpenSky Aircraft Kinematics")
    c.setAuthor("Salem Morelli")
    c.setSubject("APA-style methods and preregistration report")
    draw_title_page(c);c.showPage()
    for page_num,item in enumerate(PAGES,start=2):
        draw_standard_page(c,page_num,item,figures)
        c.showPage()
    c.save()
    pages=len(PdfReader(str(PDF_PATH)).pages)
    if pages!=26:raise RuntimeError(f"Expected 26 pages, found {pages}")
    print(f"Wrote {PDF_PATH} ({pages} pages; {PDF_PATH.stat().st_size:,} bytes)")


if __name__=="__main__":
    main()
