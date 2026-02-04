## Overview

This repository contains the code used for the paper:

Title: The Shape of Beliefs: Geometry, Dynamics, and Interventions along Representation Manifolds of Language Models' Posteriors
Sarfati et al., Goodfire team
arXiv: `https://arxiv.org/abs/2602.02315`

## Workflow

1. Generate data (sequences, activations, logits):
   ```bash
   chmod +x scripts/generate_all.sh
   ./scripts/generate_all.sh
   ```

2. Generate figures from the notebooks in `figures/`.

3. Visualize different steering schemes from the streamlit app:
   ```bash
   uv run streamlit run steering_explorer_app.py
   ```