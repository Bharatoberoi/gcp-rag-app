# Deploying on Render

This guide walks you through deploying the RAG app on [Render](https://render.com) using the included `render.yaml` blueprint.

## Prerequisites

1. A **Render** account (free tier works for testing, but Standard plan recommended for production).
2. A **GCP project** with Vertex AI APIs enabled (`aiplatform.googleapis.com`).
3. A **GCP service account** JSON key with the `Vertex AI User` role.
4. A **Qdrant Cloud** cluster (free tier available at [cloud.qdrant.io](https://cloud.qdrant.io)).

## Step 1: Set Up Qdrant Cloud

Render doesn't offer a managed Qdrant service, so use Qdrant Cloud:

1. Go to [cloud.qdrant.io](https://cloud.qdrant.io) and create a free cluster.
2. Note your **cluster URL** (e.g., `https://abc123.us-east4-0.gcp.cloud.qdrant.io:6333`).
3. Create an **API key** from the Qdrant Cloud dashboard.

## Step 2: Create a GCP Service Account Key

1. In the GCP Console, go to **IAM & Admin > Service Accounts**.
2. Create a service account (or use an existing one) with the **Vertex AI User** role.
3. Generate a JSON key and download it.
4. You'll paste the entire JSON contents into Render as a secret file (see Step 4).

## Step 3: Deploy via Render Blueprint

### Option A: One-click Blueprint

1. Push this repo to GitHub/GitLab.
2. Go to [Render Dashboard](https://dashboard.render.com) > **New** > **Blueprint**.
3. Connect your repository — Render detects the `render.yaml` automatically.
4. Fill in the prompted environment variables (see Step 4).
5. Click **Apply** to deploy.

### Option B: Manual Web Service

1. Go to Render Dashboard > **New** > **Web Service**.
2. Connect your repo and select the branch.
3. Set **Environment** to **Docker**.
4. Render will use the `Dockerfile` at the repo root.
5. Configure environment variables manually (see Step 4).
6. Add a **Disk** mounted at `/data` (1 GB is sufficient).
7. Set the **Health Check Path** to `/health`.

## Step 4: Environment Variables

Set these in the Render dashboard (or fill them when the Blueprint prompts):

| Variable | Value | Notes |
|----------|-------|-------|
| `GCP_PROJECT` | `your-gcp-project-id` | Required |
| `GCP_LOCATION` | `us-central1` | Or your preferred region |
| `QDRANT_URL` | `https://your-cluster.cloud.qdrant.io:6333` | From Qdrant Cloud |
| `QDRANT_API_KEY` | Your Qdrant Cloud API key | Mark as **secret** |
| `API_KEYS` | Comma-separated keys for auth | Mark as **secret** |
| `CORS_ORIGINS` | `https://your-app.onrender.com` | Your Render URL |
| `GCP_SA_KEY_JSON` | Paste entire JSON key contents | Secret file — rendered to `/etc/secrets/gcp-sa-key.json` |

The remaining variables have sensible defaults in `render.yaml` and typically don't need changes.

## Step 5: Verify Deployment

Once deployed, Render gives you a URL like `https://gcp-rag-app.onrender.com`.

```bash
# Health check
curl https://gcp-rag-app.onrender.com/health

# Ingest a document
curl -X POST https://gcp-rag-app.onrender.com/v1/ingest \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "file=@document.pdf"

# Query
curl -X POST https://gcp-rag-app.onrender.com/v1/query \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 5}'
```

The web UI is available at `https://gcp-rag-app.onrender.com/ui/`.

## Notes

- **Cold starts**: On the free/starter plan, Render spins down idle services. The first request after idle may take 30-60s. Use the Standard plan to avoid this.
- **Persistent storage**: The `/data` disk persists BM25 state and uploads across deploys and restarts.
- **Scaling**: Render supports autoscaling on paid plans if you need multiple instances. Note that BM25 state on disk is per-instance — for multi-instance, consider a shared store or disable BM25.
- **Logs**: View real-time logs in the Render dashboard under your service > **Logs**.
- **Custom domain**: Add a custom domain in **Settings > Custom Domains** on Render.
