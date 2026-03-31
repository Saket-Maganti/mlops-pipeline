#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Push this project to a new GitHub repo
# Usage: bash scripts/init_github.sh YOUR_GITHUB_USERNAME
# ─────────────────────────────────────────────────────────────
set -e

USERNAME=${1:-YOUR_GITHUB_USERNAME}
REPO_NAME="mlops-pipeline"

echo "Creating GitHub repo and pushing..."

# Init git
git init
git add .
git commit -m "feat: initial MLOps pipeline implementation

- FastAPI inference server with drift logging
- MLflow experiment tracking
- Evidently AI drift detection with PSI fallback
- Airflow DAG for automated retraining
- GitHub Actions CI/CD (test -> train -> build -> deploy)
- Docker + docker-compose full stack
- Synthetic drift injection (covariate/label/concept)
- Full test suite (pytest)
- AWS Lambda + ECR deployment config"

# Create GitHub repo via CLI (requires `gh` installed)
if command -v gh &> /dev/null; then
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
  echo "Repo created: https://github.com/$USERNAME/$REPO_NAME"
else
  # Manual fallback
  git branch -M main
  git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"
  echo ""
  echo "gh CLI not found. Do this manually:"
  echo "  1. Go to https://github.com/new"
  echo "  2. Name it: $REPO_NAME"
  echo "  3. Then run:"
  echo "       git push -u origin main"
fi

echo ""
echo "Add these GitHub Secrets for CI/CD:"
echo "  AWS_ACCESS_KEY_ID"
echo "  AWS_SECRET_ACCESS_KEY"
