# Physics-Informed Learning Lab

> A notebook-centric research lab for physics-informed learning. The repo is now organized around a narrow core path, optional research tracks, and frontier note lanes instead of a single oversized linear course.

## Current Maturity

### Current Status Labels

| Label | Meaning |
|---|---|
| `scaffold` | The notebook has been moved into its new lane, but the body is still a starter scaffold. |
| `core-executable` | The notebook is already runnable and narrow enough to serve as part of the main learning backbone. |

### Target Maturity Labels

| Label | Meaning |
|---|---|
| `core-executable` | Intended to become a narrow, runnable core notebook with one main research question. |
| `research-executable` | Intended to become an optional but runnable extension notebook. |
| `note-first` | Intended to become a literature-first notebook that frames the topic before any serious replication work. |

## Repo Layout

```text
.
|-- _quarto.yml              # site navigation and render targets
|-- pyproject.toml           # runtime + dev dependencies
|-- README.md
|-- AGENTS.md                # repo guidelines for contributors
|-- .github/workflows/       # Quarto build + GitHub Pages deployment
|-- robots.txt
|-- docs/
|   |-- index.md             # site landing page (hero + cards)
|   |-- quarto/              # shared SCSS + HTML includes
|   |-- javascripts/         # browser-side helpers
|   |-- 00_Lab_Guide/
|   |-- 01_Core_Path/        # 9 runnable core notebooks
|   |-- 10_Research_Tracks/ # 6 subfolders of scaffold notebooks
|   |   |-- 01_foundations_extensions/
|   |   |-- 02_advanced_pinns/
|   |   |-- 03_operator_extensions/
|   |   |-- 04_structure_preserving_networks/
|   |   |-- 05_inverse_and_discovery/
|   |   `-- 06_scalable_methods/
|   `-- 20_Frontier_Notes/   # 4 note-first scaffolds
`-- site-quarto/             # generated HTML output (do not hand-edit)
```

## Lane Overview

### `00_Lab_Guide`

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_repo_map_and_experiment_protocol.ipynb` | Entry point for repo navigation, maturity labels, and future experiment discipline | `scaffold` | `scaffold` |

### `01_Core_Path`

This is the backbone of the repo. The goal is to keep each notebook narrow: one method family, one canonical problem, one main training loop, one baseline comparison.

Deeper autodiff internals remain in `10_Research_Tracks/01_foundations_extensions` for now. The core path only needs the smallest derivative bridge required to make the PINN story coherent.

| # | Notebook | Future Focus | Current | Target |
|---|---|---|---|---|
| 1 | `01_pde_primer_and_numerical_methods.ipynb` | Minimal PDE framing, a numerical baseline, and the smallest autodiff bridge needed for PINNs | `core-executable` | `core-executable` |
| 2 | `02_vanilla_pinn_forward_problems.ipynb` | A single canonical forward PINN problem | `core-executable` | `core-executable` |
| 3 | `03_boundary_and_initial_conditions.ipynb` | Soft vs hard BC/IC enforcement on the same problem family | `core-executable` | `core-executable` |
| 4 | `04_loss_landscape_and_training_dynamics.ipynb` | Gradient imbalance, optimizer choice, and loss balancing | `core-executable` | `core-executable` |
| 5 | `05_residual_based_adaptive_sampling.ipynb` | Uniform vs adaptive collocation strategies on one benchmark | `core-executable` | `core-executable` |
| 6 | `06_multi_scale_pinns.ipynb` | Spectral bias and one multi-scale PINN thread | `core-executable` | `core-executable` |
| 7 | `07_parameter_identification_with_pinns.ipynb` | A narrow inverse problem with light noise | `core-executable` | `core-executable` |
| 8 | `08_deeponet_from_scratch.ipynb` | Minimal operator-learning baseline | `core-executable` | `core-executable` |
| 9 | `09_fourier_neural_operator.ipynb` | Minimal FNO baseline with one resolution-transfer style task | `core-executable` | `core-executable` |

### `10_Research_Tracks`

These notebooks stay outside the core path because they are broader, more specialized, or better treated as follow-on threads after the reader has a stable PINN baseline.

#### `01_foundations_extensions`

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_automatic_differentiation_for_physics.ipynb` | Deeper autograd and derivative-operator background | `scaffold` | `research-executable` |
| `02_spectral_methods_and_fourier_analysis.ipynb` | Spectral methods as an optional numerical-analysis branch | `scaffold` | `research-executable` |
| `03_neural_network_function_approximation.ipynb` | Approximation-theory background for scientific ML | `scaffold` | `research-executable` |

#### `02_advanced_pinns`

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_adaptive_activation_functions.ipynb` | Activation-design extensions for PINNs | `scaffold` | `research-executable` |
| `02_domain_decomposition_xpinn_cpinn.ipynb` | Domain-decomposition family: XPINN, cPINN, and related decompositions | `scaffold` | `research-executable` |
| `03_transfer_learning_and_meta_pinns.ipynb` | Transfer and meta-learning directions for PDE families | `scaffold` | `research-executable` |

#### `03_operator_extensions`

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_physics_informed_neural_operators.ipynb` | Bridge between operator learning and physics-based regularization | `scaffold` | `research-executable` |
| `02_latent_and_transformer_neural_operators.ipynb` | Operator variants beyond the minimal DeepONet/FNO baseline | `scaffold` | `research-executable` |

#### `04_structure_preserving_networks`

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_hamiltonian_neural_networks.ipynb` | Energy-preserving modeling track | `scaffold` | `research-executable` |
| `02_lagrangian_neural_networks.ipynb` | Lagrangian mechanics track | `scaffold` | `research-executable` |
| `03_neural_ode_and_continuous_dynamics.ipynb` | Continuous-depth modeling and solver-aware learning | `scaffold` | `research-executable` |
| `04_symplectic_and_volume_preserving_nets.ipynb` | Geometric integration and long-horizon stability | `scaffold` | `research-executable` |

#### `05_inverse_and_discovery`

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_sindy_sparse_equation_discovery.ipynb` | Sparse equation discovery track | `scaffold` | `research-executable` |
| `02_symbolic_regression_for_physics.ipynb` | Symbolic-regression track for scientific law discovery | `scaffold` | `research-executable` |
| `03_data_driven_discovery_of_conservation_laws.ipynb` | Conservation-law and symmetry discovery track | `scaffold` | `research-executable` |

#### `06_scalable_methods`

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_causal_training_and_temporal_causality.ipynb` | Causal training ideas for stiff temporal problems | `scaffold` | `research-executable` |
| `02_separable_pinns_and_efficiency.ipynb` | Efficiency-oriented architecture ideas | `scaffold` | `research-executable` |
| `03_kolmogorov_arnold_networks_for_pdes.ipynb` | KAN-style function parameterizations for PDE learning | `scaffold` | `research-executable` |
| `04_large_scale_scientific_computing.ipynb` | Systems and scaling concerns that do not belong in the core path | `scaffold` | `research-executable` |

### `20_Frontier_Notes`

These notebooks are deliberately separated from the core path. They should evolve as note-first research maps before the repo commits to serious implementation or replication.

| Notebook | Role | Current | Target |
|---|---|---|---|
| `01_pinnsformer_and_attention_pde_solvers.ipynb` | Attention-based PDE solvers and PINNsFormer-style directions | `scaffold` | `note-first` |
| `02_diffusion_models_for_scientific_computing.ipynb` | Diffusion and score-based directions for scientific computing | `scaffold` | `note-first` |
| `03_in_context_operator_learning.ipynb` | In-context operator learning and prompt-based PDE generalization | `scaffold` | `note-first` |
| `04_foundation_models_for_pdes.ipynb` | PDE foundation-model and pretraining directions | `scaffold` | `note-first` |

## Design Rules Going Forward

The restructure is meant to enforce a tighter writing and implementation discipline:

- One core notebook should answer one main question, not summarize a whole subfield.
- Core notebooks should stay on one canonical PDE or system whenever possible.
- If a topic needs multiple paper families, multiple geometries, 3D scaling, or multi-GPU systems work, it belongs in `10_Research_Tracks` or `20_Frontier_Notes`, not in `01_Core_Path`.
- Frontier notebooks should start as literature-first notes before the repo promises reproducible code.
- The core path should eventually be something a new researcher can finish end-to-end without being forced into every advanced branch.

## Suggested Reading Order

1. Start with `00_Lab_Guide/01_repo_map_and_experiment_protocol.ipynb`.
2. Complete the nine notebooks in `01_Core_Path` in order.
3. Branch into one `10_Research_Tracks` subfolder based on your current research question.
4. Use `20_Frontier_Notes` as a reading map, not as a promise of immediate replication.

## Getting Started

```bash
git clone https://github.com/duncan-nguyen/physics-informed-learning.git
cd physics-informed-learning
pip install -e .
jupyter lab
```

## Website

The repository renders as a Quarto website with a polished academic UI and can be deployed automatically to GitHub Pages.

### Local preview

Install Quarto first, then start a local preview:

```bash
pip install -e ".[dev]"
quarto preview
```

Quarto can render existing `.ipynb` files without re-executing them by default, which keeps preview fast for notebook-heavy repos.

### Static build

```bash
quarto render --to html
```

The generated site is written to `site-quarto/`.

If you want to regenerate notebook outputs during render, run the notebook separately in JupyterLab first or render an individual notebook with `quarto render path/to/notebook.ipynb --execute`.

### Auto deploy with GitHub Pages

- The workflow lives at `.github/workflows/deploy-pages.yml`.
- Every push to `main` runs `quarto render --to html`, uploads `site-quarto/`, and deploys the result through GitHub Pages.
- The website entry page is `docs/index.md`, and full navigation is defined in `_quarto.yml`.
- For this repository, the published URL is `https://duncan-nguyen.github.io/physics-informed-learning/`.

### One-time GitHub setting

In the GitHub repository settings, open **Pages** and set the source to **GitHub Actions**. After that, pushes to `main` will publish the site automatically.

## Prerequisites

- Multivariable calculus, linear algebra, and basic ODE/PDE background
- Intermediate Python
- Familiarity with NumPy and PyTorch
- Basic optimization and neural-network training concepts

## Tech Stack

| Library | Purpose |
|---|---|
| `PyTorch` | Neural networks, autograd, GPU support |
| `NumPy` | Array operations and numerical utilities |
| `SciPy` | Optimization, sparse tools, and numerical methods |
| `Matplotlib` | Visualization |
| `SymPy` | Symbolic checks and analytical references |

## Representative References

- Raissi, Perdikaris, Karniadakis. *Physics-Informed Neural Networks* (JCP, 2019)
- Lu et al. *Learning Nonlinear Operators via DeepONet* (Nat. Mach. Intell., 2021)
- Li et al. *Fourier Neural Operator for Parametric PDEs* (ICLR, 2021)
- Wang, Yu, Perdikaris. *When and Why PINNs Fail to Train* (JCP, 2022)
- Li et al. *Physics-Informed Neural Operator* (Nat. Mach. Intell., 2024)
- Zhao et al. *PINNsFormer* (ICLR, 2024)

## License

MIT License. See [LICENSE](LICENSE).
