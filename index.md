# Physics-Informed Learning Lab

<div class="landing-hero">
  <p class="eyebrow">Notebook-driven research website</p>
  <h1>From classical PDE baselines to PINNs, operator learning, and frontier research.</h1>
  <p>
    This site renders the repository as a continuously deployed academic lab. It is designed to be honest about notebook maturity while still giving readers a clean path from foundations to current research directions.
  </p>
  <div class="hero-actions">
    <a class="hero-button primary" href="00_Lab_Guide/01_repo_map_and_experiment_protocol.html">Open the Lab Guide</a>
    <a class="hero-button secondary" href="01_Core_Path/01_pde_primer_and_numerical_methods.html">Start the Core Path</a>
  </div>
</div>

<div class="metric-strip">
  <div class="metric-card">
    <span class="metric-number">1</span>
    <span class="metric-label">lab guide</span>
  </div>
  <div class="metric-card">
    <span class="metric-number">9</span>
    <span class="metric-label">core notebooks</span>
  </div>
  <div class="metric-card">
    <span class="metric-number">19</span>
    <span class="metric-label">research-track notebooks</span>
  </div>
  <div class="metric-card">
    <span class="metric-number">4</span>
    <span class="metric-label">frontier notes</span>
  </div>
</div>

## How to Navigate the Site

<div class="lane-grid">
  <a class="lane-card" href="00_Lab_Guide/01_repo_map_and_experiment_protocol.html">
    <span class="lane-tag">00</span>
    <h3>Lab Guide</h3>
    <p>Start here for repo structure, maturity labels, and experiment discipline.</p>
  </a>
  <a class="lane-card" href="01_Core_Path/01_pde_primer_and_numerical_methods.html">
    <span class="lane-tag">01</span>
    <h3>Core Path</h3>
    <p>The narrow backbone of the project: one canonical problem per notebook, one main idea at a time.</p>
  </a>
  <a class="lane-card" href="10_Research_Tracks/01_foundations_extensions/01_automatic_differentiation_for_physics.html">
    <span class="lane-tag">10</span>
    <h3>Research Tracks</h3>
    <p>Optional branches for deeper dives into advanced PINNs, operator learning, structure preservation, and discovery.</p>
  </a>
  <a class="lane-card" href="20_Frontier_Notes/01_pinnsformer_and_attention_pde_solvers.html">
    <span class="lane-tag">20</span>
    <h3>Frontier Notes</h3>
    <p>Literature-first notebooks for emerging directions that are not yet mature enough for full reproduction.</p>
  </a>
</div>

## What This Website Optimizes For

- **Clarity over breadth**: the core path should stay digestible instead of turning into a giant survey.
- **Notebook-native research flow**: the website mirrors the repo structure rather than hiding it behind a separate docs layer.
- **Honest maturity**: some pages are fully runnable, many are still scaffolds, and frontier notes are explicitly exploratory.
- **Continuous deployment**: every push to `main` can rebuild and publish the site on GitHub Pages.

## Recommended Entry Points

- If you are new to the repo, begin with [Repo Map and Experiment Protocol](00_Lab_Guide/01_repo_map_and_experiment_protocol).
- If you want the first runnable technical notebook, open [PDE Primer and Numerical Methods](01_Core_Path/01_pde_primer_and_numerical_methods).
- If you already know the basics and want depth, jump straight into one of the grouped research-track sections in the left navigation.
- If you care about emerging methods, use the frontier notes as a reading map rather than a promise of full benchmark reproduction.

## Deployment Model

This website is built with Jupyter Book and intended for GitHub Pages deployment through GitHub Actions. The repository includes a workflow that:

1. installs the project and site dependencies,
2. builds the static HTML book,
3. uploads the Pages artifact,
4. deploys it to GitHub Pages.

Once GitHub Pages is configured to use **GitHub Actions** as the source, every push to `main` will publish a fresh version of the site.

> The site can render notebooks directly from the repository tree. As more notebooks become executable, the website will gradually become a stronger teaching and research surface instead of only a file browser.
