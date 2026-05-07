# Production deploy (GCP + Managed Qdrant)

This stack runs the **same** hybrid Qdrant pipeline locally, but in production you point **`QDRANT_URL`** at **[Qdrant Cloud](https://cloud.qdrant.io/)** (or another managed Qdrant) and run the API on **Cloud Run**.

## 1. Qdrant Cloud

1. Create a cluster (pick a region close to **`GCP_LOCATION`**, e.g. `us-east4` / `us-central1`).
2. Copy the **HTTPS cluster URL** (include port **`6333`** if shown), e.g. `https://xxxxxxxx.aws.cloud.qdrant.io:6333`.
3. Create an **API key** with write access for your app.

## 2. Secret Manager (before Terraform)

```bash
gcloud config set project YOUR_PROJECT_ID

# Qdrant API key (required)
echo -n 'PASTE_QDRANT_API_KEY' | gcloud secrets create rag-qdrant-api-key --replication-policy=automatic --data-file=-

# Groq API key (required for answer generation)
echo -n 'PASTE_GROQ_API_KEY' | gcloud secrets create rag-groq-api-key --replication-policy=automatic --data-file=-

# Optional: app API keys (comma-separated) for X-API-Key on /v1/*
echo -n 'your-long-random-secret' | gcloud secrets create rag-app-api-keys --replication-policy=automatic --data-file=-
```

## 3. Artifact Registry + container image

```bash
gcloud artifacts repositories create rag --repository-format=docker --location=us-central1

cd gcp-rag-app
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/rag/gcp-rag-app:latest .
```

## 4. Terraform

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: project_id, container_image, qdrant_url, qdrant_secret_id, groq_secret_id, cors_origins, etc.

terraform init
terraform apply
```

Outputs:

- **`cloud_run_uri`** - open **`{cloud_run_uri}/ui/`** in the browser.
- If **`allow_unauthenticated = false`**, grant yourself invoker:

```bash
gcloud run services add-iam-policy-binding gcp-rag-app \
  --region=us-central1 \
  --member="user:you@example.com" \
  --role="roles/run.invoker"
```

## 5. App behaviour in production

| Env | Purpose |
|-----|---------|
| `PRODUCTION_MODE=true` | Fails startup if `QDRANT_URL` is `memory` (prevents accidental ephemeral vector DB). |
| `QDRANT_URL` | Managed Qdrant HTTPS URL. |
| `QDRANT_API_KEY` | From Secret Manager in Terraform (or set manually in Cloud Run). |
| `GROQ_API_KEY` | From Secret Manager in Terraform; required for answer generation. |
| `LLM_MODEL` | Groq model name, defaults to `llama-3.1-8b-instant`. |
| `DOCS_ENABLED=false` | Disables `/docs` and OpenAPI JSON on the public service. |
| `API_KEYS` | Optional comma-separated keys; `/v1/*` requires header **`X-API-Key`**. |
| `CORS_ORIGINS` | Set to your Cloud Run URL (and any other front-end origins) so `/ui` in the browser can call the API. |

**BM25 file** (`BM25_STATE_PATH`): stored on the container disk by default. For multi-revision durability, add a **Cloud Storage sync** or a **mounted volume** (Filestore / NFS) in a follow-up change.

## 6. GCP checklist

- Billing enabled.
- **Qdrant** reachable from Cloud Run (Qdrant Cloud is public HTTPS; VPC peering is optional for stricter setups).
- **Groq API key** available in Secret Manager.
