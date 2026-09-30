# Deployment guide: from zip to live service via GitHub

This guide takes the project from the zip file to a live API on Google Cloud Run, with **GitHub as the
single source of truth**: every change goes through Git, GitHub Actions tests it, and a push to `main`
deploys it. You never deploy by hand.

```
your change ──git push──► GitHub ──Actions──► tests ──► Docker image ──► Artifact Registry ──► Cloud Run
                                     │
                                     └── weekly: drift check ──► GitHub issue "retrain"
```

**Time:** about 45 minutes the first time. **Cost:** stays inside the Cloud Run free tier for a demo;
a budget alert protects you.

Everything below runs in **Google Cloud Shell** (a terminal in the browser with `git`, `gcloud` and
Python preinstalled), so you don't need to install anything locally.

---

## Overview

| Part | What you do | Result |
|---|---|---|
| A | Accounts and names | GitHub account, Google account, fixed names |
| B | Google Cloud project, billing, budget | Project with billing and a budget alert |
| C | Cloud resources for deployment | Artifact Registry, service account, keyless GitHub login |
| D | GitHub repository and variables | Empty repo with 3 variables |
| E | First push | Code on GitHub, Actions deploys automatically |
| F | Verify | Live URL, green pipeline |
| G | Daily workflow | Branch, pull request, merge, release a new model |
| H | Drift check | Weekly check, issue on drift |
| I | Rollback, canary, clean-up | Operations |
| J | Troubleshooting | Common errors and fixes |

---

## Part A: Accounts and names

1. **GitHub account:** sign in at [github.com](https://github.com).
2. **Google account:** sign in at [console.cloud.google.com](https://console.cloud.google.com).
3. **Fix your names now.** They appear in several commands, and some of them are case-sensitive.
   Open Cloud Shell (the `>_` icon at the top right of the Cloud Console) and set:

```bash
export GITHUB_USER=<your-github-username>        # exactly as in your GitHub URL, case matters
export REPO_NAME=xpass-ml-service
export PROJECT_ID=xpass-demo-<yourname>-2026     # globally unique, lowercase, 6-30 characters
export REGION=europe-west3                       # Frankfurt
```

> Cloud Shell forgets variables when the session ends. If you come back later, run these four lines again.

---

## Part B: Google Cloud project, billing, budget

### B1. Create the project

```bash
gcloud projects create $PROJECT_ID --name="xPass demo"
gcloud config set project $PROJECT_ID
```

**Output:** `Create in progress for [https://cloudresourcemanager.googleapis.com/v1/projects/xpass-demo-...]`

### B2. Link billing (in the browser)

Cloud Console → **Billing** → **My projects** → your project → **Change billing** → select your billing
account. If you have no billing account yet, create one (credit card needed; new accounts usually receive
free credit).

Check:
```bash
gcloud billing projects describe $PROJECT_ID --format='value(billingEnabled)'
```
**Output:** `True`

### B3. Budget alert (do not skip)

Cloud Console → **Billing** → **Budgets & alerts** → **Create budget**:
- Scope: your project
- Amount: `1` EUR
- Alerts at 50 %, 90 %, 100 % (email to you)

**Result:** you get an email long before anything costs real money.

---

## Part C: Cloud resources for deployment

### C1. Enable the required services

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com
```

**Output:** `Operation "operations/..." finished successfully.` (takes about 1 minute)

| Service | Why |
|---|---|
| Cloud Run | Runs the container |
| Artifact Registry | Stores the Docker images |
| IAM, IAM Credentials, STS | Keyless login from GitHub Actions (Workload Identity Federation) |

### C2. Artifact Registry repository (image storage)

```bash
gcloud artifacts repositories create xpass \
  --repository-format=docker --location=$REGION \
  --description="xPass service images"
```

**Output:** `Created repository [xpass].`

### C3. Service account for GitHub Actions

This is the technical identity GitHub uses to deploy. It gets only the three roles it needs
(principle of least privilege).

```bash
gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"
export SA=github-deployer@$PROJECT_ID.iam.gserviceaccount.com

for ROLE in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:$SA --role=$ROLE --condition=None --quiet
done
```

| Role | Allows |
|---|---|
| `run.admin` | Create and update the Cloud Run service, make it public |
| `artifactregistry.writer` | Push Docker images |
| `iam.serviceAccountUser` | Run the service under the default runtime identity |

### C4. Workload Identity Federation (keyless login from GitHub)

Instead of storing a JSON key in GitHub (a security risk), Google trusts short-lived tokens that GitHub
issues for **your repository only**.

```bash
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

gcloud iam workload-identity-pools create github \
  --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github \
  --display-name="GitHub OIDC" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='$GITHUB_USER/$REPO_NAME'"

gcloud iam service-accounts add-iam-policy-binding $SA \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$GITHUB_USER/$REPO_NAME"
```

### C5. Print the three values for GitHub

```bash
echo "GCP_PROJECT_ID   = $PROJECT_ID"
echo "GCP_WIF_PROVIDER = projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-provider"
echo "GCP_DEPLOY_SA    = $SA"
```

**Output:** three lines. Keep this tab open, you need them in Part D.

---

## Part D: GitHub repository and variables

### D1. Create an empty repository

GitHub → **New repository**:
- Name: `xpass-ml-service` (must match `$REPO_NAME`)
- Visibility: **Public** (so employers can see it; also gives unlimited free Actions minutes)
- **Do not** add a README, .gitignore or license (the project already has them)

### D2. Add the three repository variables

Repository → **Settings** → **Secrets and variables** → **Actions** → tab **Variables** → **New repository variable**:

| Name | Value (from C5) |
|---|---|
| `GCP_PROJECT_ID` | `xpass-demo-...` |
| `GCP_WIF_PROVIDER` | `projects/<number>/locations/global/workloadIdentityPools/github/providers/github-provider` |
| `GCP_DEPLOY_SA` | `github-deployer@<project>.iam.gserviceaccount.com` |

These are **variables**, not secrets: none of them is a password. Security comes from the Workload
Identity condition (only your repository can use them).

### D3. Personal access token for pushing from Cloud Shell

GitHub → your avatar → **Settings** → **Developer settings** → **Personal access tokens** →
**Fine-grained tokens** → **Generate new token**:
- Repository access: **Only select repositories** → `xpass-ml-service`
- Permissions: **Contents: Read and write**, **Workflows: Read and write**
- Expiration: 30 days

Copy the token (starts with `github_pat_`). You will paste it as the password when Git asks.

> **Workflows: Read and write** is required because the project contains `.github/workflows/`.
> Without it, GitHub rejects the push with *"refusing to allow a Personal Access Token to create or
> update workflow"*.

---

## Part E: First push

### E1. Upload and unpack the project in Cloud Shell

Cloud Shell → **⋮** (More) → **Upload** → select `xpass-ml-service.zip`, then:

```bash
cd ~ && unzip -o xpass-ml-service.zip && cd xpass-ml-service
ls
```

**Output:** `Dockerfile  Makefile  README.md  data  docs  models  pyproject.toml  reports  ...`

### E2. Optional: run the tests before the first commit

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -q -r requirements-dev.txt
python -m pytest -q
```

**Output:** `25 passed`

### E3. Initialise Git and make the first commit

```bash
git init -b main
git config user.name  "Mohsen Parizad Moghaddam"
git config user.email "<email-of-your-github-account>"
git add .
git status          # check: no data/raw, no data/processed, no .venv
git commit -m "xPass ML service: pipeline, API, tests, CI/CD, drift monitoring"
```

**Output:** `[main (root-commit) abc1234] xPass ML service ... 46 files changed`

`.gitignore` keeps downloaded data and caches out of the repository. The model files (about 4 MB) are
included on purpose: they are the versioned artifact that gets deployed.

### E4. Push

```bash
git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git
git push -u origin main
# Username: your GitHub username
# Password: the github_pat_... token from D3
```

**Output:** `* [new branch] main -> main`

**The push starts the pipeline automatically.**

---

## Part F: Verify

### F1. Watch the pipeline

GitHub → repository → **Actions** → run **CI/CD**. Two jobs:

| Job | Steps | Duration |
|---|---|---|
| `test` | install → 25 tests → Docker build | about 3 minutes |
| `deploy` | Google login (keyless) → push image → deploy Cloud Run → smoke test `/health` | about 4 minutes |

Both green = the service is live. If a job is red, open it, read the failing step and check Part J.

### F2. Find the URL

```bash
gcloud run services describe xpass-api --region $REGION --format='value(status.url)'
```

**Output:** `https://xpass-api-xxxxxxxxxx.europe-west3.run.app`

### F3. Test the live service

```bash
URL=$(gcloud run services describe xpass-api --region $REGION --format='value(status.url)')
curl -s $URL/health
curl -s -X POST $URL/predict -H "Content-Type: application/json" -d '{
  "height":"ground","start_x":60,"start_y":40,"end_x":45,"end_y":38,
  "outplayed_players":0,"goal_diff":4,"opponent":"Norway"}'
```

**Output:**
```
{"status":"ok","model_version":"1.0.0"}
{"xpass":0.9977,"risk_category":"non-risky","model_version":"1.0.0"}
```

Open **`<URL>/docs`** in the browser: the interactive Swagger page is your live demo for interviews.

### F4. See the prediction logs

Cloud Console → **Cloud Run** → `xpass-api` → **Logs**. Every prediction appears as one JSON line
(features, xPass, risk, model version, latency). This is the data a production drift monitor would read.

### F5. Add the URL to the README

Replace the first lines under the title with:

```markdown
**Live demo:** https://xpass-api-xxxxxxxxxx.europe-west3.run.app/docs
![CI/CD](https://github.com/<GITHUB_USER>/xpass-ml-service/actions/workflows/ci-cd.yml/badge.svg)
```

Then commit and push (this also tests the whole flow a second time):

```bash
git add README.md && git commit -m "Add live demo link and CI badge" && git push
```

---

## Part G: Daily workflow with Git

### G1. Protect `main` (recommended)

GitHub → **Settings** → **Branches** → **Add branch ruleset** (or *branch protection rule*) for `main`:
- Require a pull request before merging
- Require status checks to pass: `test`

Now nothing reaches production without passing tests.

### G2. Make a change via branch and pull request

```bash
git checkout -b feature/better-power-index
# ... edit code ...
python -m pytest -q
git add -A && git commit -m "Use pre-tournament FIFA ranking for power index"
git push -u origin feature/better-power-index
```

GitHub → **Compare & pull request** → the `test` job runs on the PR (no deployment) →
**Merge** → the merge to `main` triggers `deploy`.

### G3. Release a new model version

Retraining changes the model files, so it is a normal commit, with a version number and a tag:

```bash
make data features                 # fresh data
make train VERSION=1.1.0           # exit code 1 = gate failed -> do not commit
python -m pytest -q
git add models reports
git commit -m "Model v1.1.0: retrained on ..."
git tag v1.1.0
git push && git push --tags
```

**Result:** `/health` returns `"model_version":"1.1.0"` after the deployment, and the Git tag links the
running model to the exact code and data version (`models/metadata.json` contains the data hash).

---

## Part H: Drift check

The workflow `.github/workflows/drift.yml` runs **every Monday at 06:00 UTC** and can be started by hand.

### H1. Run it once by hand

GitHub → **Actions** → **Drift check** → **Run workflow** → tag `wwc2023` → **Run**.

**Output:**
- Job summary: the report is attached as the artifact **drift-report**.
- A new **Issue**: *"Retraining recommended: drift detected on wwc2023"* with the full report
  (66 % of passes against opponents without power index, Brier 0.111 → 0.134).

### H2. Close the loop

Fix the cause on a branch (Part G2), release a new model (Part G3), reference the issue in the commit
message (`Fixes #1`), and GitHub closes the issue automatically on merge.

This is exactly the BPMN process in `docs/xpass_process.png`: timer → drift check → gateway →
issue → retraining → evaluation gate → deployment.

---

## Part I: Rollback, canary, clean-up

### Roll back to the previous version (seconds, no rebuild)

```bash
gcloud run revisions list --service xpass-api --region $REGION
gcloud run services update-traffic xpass-api --region $REGION --to-revisions=<old-revision>=100
```

### Canary: send 10 % of traffic to the new version

```bash
gcloud run services update-traffic xpass-api --region $REGION \
  --to-revisions=<new-revision>=10,<old-revision>=90
```

Afterwards, back to normal: `--to-latest`.

> The pipeline always deploys to 100 %. Use canary manually when you want to compare a new model live.

### Clean up everything

```bash
gcloud run services delete xpass-api --region $REGION
gcloud artifacts repositories delete xpass --location $REGION
# or delete the whole project (stops all costs):
gcloud projects delete $PROJECT_ID
```

---

## Part J: Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Push rejected: *refusing to allow a Personal Access Token to create or update workflow* | Token lacks the Workflows permission | New token with **Workflows: Read and write** (D3) |
| `deploy` fails at **auth**: *unauthorized_client* / *attribute condition* | Repository name in C4 does not match exactly (case!) | Check `echo $GITHUB_USER/$REPO_NAME` against the GitHub URL; recreate the provider or update its `--attribute-condition` |
| `deploy` fails at **auth**: *permission iam.serviceAccounts.getAccessToken denied* | `workloadIdentityUser` binding missing or wrong repository | Repeat the last command of C4 |
| `deploy` job skipped | Pipeline ran on a pull request, not on `main` | Expected: only pushes to `main` deploy |
| Variables empty (`projects//locations/...`) | Variables created as **secrets** or in another repository | Create them under **Variables**, in this repository (D2) |
| Push to Artifact Registry: *denied* / *403* | Repository missing or wrong region | Run C2; region in the workflow is `europe-west3` |
| Cloud Run: *container failed to start and listen on PORT* | App not listening on `$PORT` | The Dockerfile already uses `${PORT}`; check logs for an import error |
| Cloud Run: *memory limit exceeded* | CatBoost model + pandas need about 500 MB | Keep `--memory=1Gi` (set in the workflow) |
| *Billing account not found* / services cannot be enabled | Billing not linked | B2, then retry |
| `gcloud` commands fail after a pause | Cloud Shell session restarted, variables lost | Re-run the `export` lines from Part A |
| Drift workflow cannot create an issue | Issues disabled | Settings → General → Features → enable **Issues** |

---

## What this demonstrates (for your CV)

- Git-based delivery: branch → pull request → CI → merge → automatic deployment
- Evaluation gate and tests block bad models from production
- Container deployment on Google Cloud Run with keyless authentication (Workload Identity Federation)
- Versioned model artifacts linked to code (Git tag) and data (hash in `metadata.json`)
- Scheduled drift monitoring that opens an issue and triggers retraining
- Rollback and canary release via Cloud Run revisions
