# Rigorous Reconstruction of the OpenSky Streaming Bayesian Mixture Project

## Peer-review determination

The submitted draft contains a valuable research direction, but its claim of a
“fully functional,” “measure-theoretically sound” state-space classifier is not
yet supportable. It confounds four objects: a public aircraft-state API, a
static exchangeable mixture, a longitudinal state-space model, and
radar/emitter classification. The code also queries the wrong URL, supplies an
invalid bounding box, uses an incomplete DP truncation, treats a singular
ambient Gaussian as a manifold density, constructs artificial MCMC chains, and
describes a six-cell experiment as an 18-cell split plot.

The corrected scope is:

> A topology-aware, open-world Bayesian clustering experiment for streaming
> aircraft kinematics obtained from OpenSky state vectors. The benchmark
> compares a truncated streaming variational approximation with a collapsed
> Dirichlet-process particle filter under controlled observation thinning. It
> is an aviation-surveillance methodology study, not a direct radar-pulse,
> emitter, ELINT, or operational validation study.

The companion implementation is OpenSky_Streaming_DPMM_Corrected.py.

## 1. Defects corrected

| Submitted element | Defect | Correction |
|---|---|---|
| OpenSky website root | This is not the state-vector operation. | Use https://opensky-network.org/api/states/all. |
| Duplicate lamin key | The second entry overwrites the first and lomin is absent. | Supply (lamin, lomin, lamax, lomax) and validate WGS84 bounds. |
| Username/password authentication | OpenSky no longer accepts HTTP basic authentication for this operation. | Use OAuth2 client credentials and renewable bearer tokens, or the documented anonymous tier. |
| Four-dimensional Gaussian after circle embedding | A Gaussian density relative to four-dimensional Lebesgue measure assigns probability zero to the embedded circle. An indicator does not repair normalization. | Define a product kernel relative to Euclidean Lebesgue measure and Haar measure on the circle. |
| DP base measure on the observation manifold | DP atoms must be component parameters. | Put the base measure on a component-parameter space. |
| K beta sticks for K components | The residual stick is lost. | Use K−1 beta variables and make component K the residual mass. |
| Fixed component variances | This is not the stated conjugate variational model. | Use Normal–Gamma factors for Euclidean coordinates and a von Mises directional factor. |
| Constant gain called posterior consistency | Constant gain creates forgetting. | Separate cumulative inference from power-prior tracking. |
| Identifiers deleted before association | Each response becomes an unordered cross-section. | Keep an ephemeral pseudonymous association key outside the model features. |
| OpenSky described as pulse trains | OpenSky supplies aircraft state vectors, not received radar pulses. | Treat the data as aircraft-kinematic surveillance observations. |
| Noisy copy called a second chain | This is not an independent Markov chain. | Diagnose only genuine independently initialized MCMC chains. |
| Folded R-hat alone | Modern practice uses bulk and folded rank-normalized split statistics. | Report their maximum and bulk ESS. |
| New live pull for each treatment | Treatment is confounded with changing traffic and API conditions. | Record each block once and replay identical data and masks. |
| “18-cell” 2 by 3 design | There are six cells. | Use six within-block cells and 100 blocks, or 600 executions. |
| Split-plot label | No whole-plot randomization is defined. | Call it a crossed repeated-measures randomized-block design. |

## 2. Correct measure-theoretic formulation

### 2.1 Observation space and reference measure

For an aircraft state observed at index \(t\), let

\[
Y_t=(V_t,\Theta_t,R_t)\in
\mathcal Y=(0,\infty)\times S^1\times\mathbb R,
\]

where \(V_t\) is ground speed, \(\Theta_t\) is true track, and \(R_t\)
is vertical rate. Equip this space with

\[
\mu=\lambda_{(0,\infty)}\otimes\lambda_{S^1}\otimes\lambda_{\mathbb R},
\]

where \(\lambda_{S^1}(d\theta)=d\theta\) is the arc-length Haar measure
under the angular parameterization.
For numerical conditioning define

\[
Z_t=
\begin{bmatrix}
\log(1+V_t)\\
\operatorname{asinh}(R_t/s_0)
\end{bmatrix}\in\mathbb R^2,
\qquad s_0=5\ \mathrm{m/s}.
\]

The circle may be displayed as
\(u(\theta)=(\cos\theta,\sin\theta)\), but the likelihood remains a
density on \(S^1\); it is not an unconstrained Gaussian density in
\(\mathbb R^2\).

### 2.2 Product mixture kernel

Let

\[
\vartheta_k=(m_k,\Lambda_k,\mu_k,\kappa_\theta)\in
\Theta=\mathbb R^2\times\mathbb R_+^2\times S^1\times\mathbb R_+.
\]

The corrected kernel is

\[
f(z,\theta\mid\vartheta_k)
=
\prod_{d=1}^{2}
\mathcal N(z_d\mid m_{kd},\lambda_{kd}^{-1})
\frac{\exp\{\kappa_\theta\cos(\theta-\mu_k)\}}
{2\pi I_0(\kappa_\theta)}.
\]

This is normalized relative to
\(\lambda_{\mathbb R^2}\otimes\lambda_{S^1}\) and respects periodic
topology. The baseline fixes and preregisters \(\kappa_\theta\).
Estimating it requires an additional variational factor or Metropolis update.

### 2.3 Dirichlet-process mixture

The random mixing measure is defined on \(\Theta\):

\[
G\mid\alpha,G_0\sim\operatorname{DP}(\alpha,G_0),\qquad
\vartheta_t\mid G\sim G,\qquad
(Z_t,\Theta_t)\mid\vartheta_t\sim f(\cdot\mid\vartheta_t).
\]

Under the Sethuraman representation,

\[
G=\sum_{k=1}^{\infty}\pi_k\delta_{\vartheta_k},\qquad
\pi_k=V_k\prod_{j<k}(1-V_j),\qquad
V_k\stackrel{\mathrm{iid}}{\sim}\operatorname{Beta}(1,\alpha).
\]

For a \(K\)-component truncation, only \(V_1,\ldots,V_{K-1}\) are
random and

\[
\pi_K=\prod_{j=1}^{K-1}(1-V_j).
\]

The expected ordinary GEM residual after \(K-1\) sticks is

\[
\mathbb E(R_K)=
\left(\frac{\alpha}{1+\alpha}\right)^{K-1},
\]

which supplies a transparent truncation diagnostic. Finite \(K\) remains an
approximation and is not literally infinite-dimensional.

### 2.4 Static and nonstationary targets

An ordinary DP mixture is conditionally exchangeable. It is not a state-space
model and does not make clusters drift by itself. Two streaming regimes are
defensible:

1. Cumulative sufficient statistics approximate a stationary-mixture
   posterior.
2. Geometric forgetting defines a power target

\[
\widetilde\Pi_t(d\vartheta)
\propto
\Pi_0(d\vartheta)
\prod_{s=1}^{t}f(Y_s\mid\vartheta)^{\lambda^{t-s}},
\qquad 0<\lambda<1.
\]

The second regime supports adaptation but is not the ordinary stationary
posterior. A genuinely nonstationary Bayesian model requires a dependent
random measure \(G_t\), explicit component transitions, or a dependent
Chinese restaurant process.

## 3. Corrected inference

### 3.1 Truncated streaming variational Bayes

Use

\[
q(V,\vartheta,C)=
\prod_{k=1}^{K-1}q(V_k)
\prod_{k=1}^{K}q(m_k,\Lambda_k)q(\mu_k)
\prod_i q(C_i).
\]

Responsibilities satisfy

\[
\log r_{ik}
\stackrel{c}{=}
\mathbb E_q\log\pi_k+
\mathbb E_q\log f(Z_i,\Theta_i\mid\vartheta_k).
\]

For \(k<K\),

\[
\begin{aligned}
\mathbb E\log\pi_k
&=\psi(a_k)-\psi(a_k+b_k)
+\sum_{j<k}\{\psi(b_j)-\psi(a_j+b_j)\},\\
\mathbb E\log\pi_K
&=\sum_{j<K}\{\psi(b_j)-\psi(a_j+b_j)\}.
\end{aligned}
\]

The beta updates are

\[
a_k=1+N_k,\qquad
b_k=\alpha+\sum_{j>k}N_j,\qquad k=1,\ldots,K-1.
\]

The companion implementation updates complete Normal–Gamma sufficient
statistics and the directional natural vector

\[
\eta_k=\kappa_\theta\sum_i r_{ik}
\begin{bmatrix}\cos\Theta_i\\\sin\Theta_i\end{bmatrix}.
\]

Thus \(q(\mu_k)\) is von Mises with mean direction
\(\operatorname{atan2}(\eta_{k2},\eta_{k1})\) and concentration
\(\lVert\eta_k\rVert\). No arbitrary decimal stabilizer is added to the
posterior; positivity is protected at the machine-representable boundary.

### 3.2 Collapsed DP sequential Monte Carlo

Each particle contains a partition and its sufficient statistics. For a new
observation \(y_t\), use the fully adapted proposal

\[
q(c_t=k\mid c_{1:t-1},y_{1:t})
\propto
\begin{cases}
n_{k,t-1}\,p(y_t\mid y_{c=k}),&k\text{ existing},\\
\alpha\,p(y_t\mid G_0),&k\text{ new}.
\end{cases}
\]

The Euclidean predictive density is Student-\(t\). For directional sufficient
vector \(\eta_k\), the von Mises predictive term is

\[
p(\theta\mid\eta_k)=
\frac{I_0(\lVert\eta_k+\kappa_\theta u(\theta)\rVert)}
{2\pi I_0(\lVert\eta_k\rVert)I_0(\kappa_\theta)}.
\]

When particle ESS falls below a preregistered fraction of \(N\), systematic
resampling is followed by collapsed Gibbs sweeps over a fixed recent window.
This is a valid SMC resample–move construction. It is not created by adding
noise to responsibilities.

### 3.3 Diagnostics

Particle ESS, MCMC ESS, and R-hat are different objects:

- Particle ESS diagnoses degeneracy of normalized SMC weights.
- Bulk and tail MCMC ESS measure autocorrelation in actual chains.
- Rank-normalized split R-hat compares genuine independent MCMC chains and is
  not defined for deterministic VI iterates.

For MCMC outputs, use at least four independently initialized chains and report

\[
\widehat R=
\max(\widehat R_{\mathrm{bulk}},\widehat R_{\mathrm{folded}}).
\]

The companion code implements average ranks for ties, Blom rank
normalization, split chains, folded and bulk R-hat, and a Geyer
positive-sequence bulk ESS estimator.

## 4. OpenSky data contract and blinding

The current operation is:

    GET https://opensky-network.org/api/states/all

Bounding-box parameters are lamin, lomin, lamax, and lomax. Authenticated
requests use OAuth2 client credentials and a bearer token. The client requests
extended=1 so aircraft category, when present, is retained only for
evaluation.

Blinding is implemented through separation:

- the model receives only transformed velocity, vertical rate, and heading;
- the API identifier is converted to an ephemeral, keyed session pseudonym for
  association and blocking;
- category is sealed from fitting and used only after model lock; and
- neither the salt nor the original identifier belongs in committed artifacts.

OpenSky category is a coarse aircraft-category field. NMI against it measures
alignment with those categories; it does not establish discovery of radar
modes, emitter identities, or adversarial behavior.

## 5. Corrected experimental design

### 5.1 Experimental unit

There are six cells:

- Factor A: streaming variational Bayes versus collapsed DP-SMC;
- Factor B: 0%, 15%, and 30% controlled observation thinning; and
- 100 frozen streaming blocks, each evaluated under all six combinations.

This gives \(2\times3\times100=600\) method-cell executions. Capture each block
once, archive it with a timestamp and integrity hash, and replay it to every
condition. Generate one uniform random vector per block and threshold that
same vector at 0.15 and 0.30 to obtain nested masks. Both algorithms then
receive identical retained observations.

If the treatment is network-update loss rather than observation thinning, the
entire timestamped batch must be removed. These are distinct missing-data
mechanisms.

### 5.2 Mixed-effects model

Because every block receives every treatment, this is crossed repeated
measures. With \(A_a\in\{-1/2,1/2\}\) and orthonormal linear and quadratic
dropout contrasts \(B_{Lj},B_{Qj}\), a working model is

\[
\begin{aligned}
g(Y_{kaj})={}&
\beta_0+\beta_AA_a+\beta_LB_{Lj}+\beta_QB_{Qj}
+\beta_{AL}A_aB_{Lj}+\beta_{AQ}A_aB_{Qj}\\
&+b_{0k}+b_{Ak}A_a+b_{Lk}B_{Lj}+b_{Qk}B_{Qj}
+\varepsilon_{kaj},
\end{aligned}
\]

where \(b_k\sim\mathcal N(0,\Sigma_b)\). A maximal covariance may be unstable
with six observations per block; preregister a parsimonious structure and
justify simplification. Cell-specific residual scales or a heterogeneous
repeated-measures covariance may be used when supported.

Match the analysis to the endpoint:

- log latency: Gaussian mixed model after separating network and compute time;
- NMI in \([0,1]\): bootstrap or an appropriate bounded-response model;
- held-out log predictive density: Gaussian working model if diagnostics allow;
- particle ESS: SMC diagnostic only; and
- R-hat and bulk ESS: MCMC diagnostics only, not cross-method responses.

Category NMI should be secondary. Primary outcomes should include held-out log
predictive density, partition stability under thinning, occupied-cluster
uncertainty, and compute latency per retained observation.

## 6. Implementation

Install on Windows Git Bash:

    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    python -m pip install torch numpy scipy requests

Run the deterministic offline smoke test:

    python OpenSky_Streaming_DPMM_Corrected.py

Run one descriptive anonymous pull:

    python OpenSky_Streaming_DPMM_Corrected.py --live

For authenticated access, set OAuth2 credentials in the environment rather
than source code:

    export OPENSKY_CLIENT_ID=\"your-client-id\"
    export OPENSKY_CLIENT_SECRET=\"your-client-secret\"
    python OpenSky_Streaming_DPMM_Corrected.py --live

One pull is a connectivity demonstration, not the factorial experiment. The
publication experiment still requires a capture-and-replay controller, frozen
block manifests, nested masks, at least four genuine MCMC runs whenever R-hat
is reported, and preregistered endpoints and exclusions.

## 7. Defensible claims

The reconstruction supports these claims:

1. The observation model is defined relative to a valid reference measure on
   \(\mathbb R^2\times S^1\).
2. Truncated variational weights include the residual stick.
3. Euclidean and directional component uncertainty is represented.
4. SMC particles propagate partitions and receive Gibbs rejuvenation.
5. Evaluation labels and pseudonymous keys are excluded from model features.
6. Network and synchronized inference latency are separated.
7. The experiment contains six cells and requires paired replay.

The following remain unsupported until empirical artifacts exist:

- superiority of either inference strategy;
- discovery of genuinely new aircraft classes;
- longitudinal state-space tracking performance;
- radar-pulse or emitter-mode classification;
- operational ELINT or SIGINT validation; and
- posterior consistency under constant forgetting.

## References

Blei, D. M., and Jordan, M. I. (2006). Variational inference for Dirichlet
process mixtures. Bayesian Analysis, 1(1), 121–144.
https://doi.org/10.1214/06-BA104

Hoffman, M. D., Blei, D. M., Wang, C., and Paisley, J. (2013). Stochastic
variational inference. Journal of Machine Learning Research, 14, 1303–1347.

OpenSky Network. (2026). OpenSky REST API documentation.
https://openskynetwork.github.io/opensky-api/rest.html

Sethuraman, J. (1994). A constructive definition of Dirichlet priors.
Statistica Sinica, 4, 639–650.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P.-C. (2021).
Rank-normalization, folding, and localization: An improved R-hat for assessing
convergence of MCMC. Bayesian Analysis, 16(2), 667–718.
https://doi.org/10.1214/20-BA1221
