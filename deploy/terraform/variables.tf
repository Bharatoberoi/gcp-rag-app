variable "project_id" {
  type        = string
  description = "GCP project id."
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "Cloud Run region (keep Qdrant cluster in a compatible region)."
}

variable "service_name" {
  type        = string
  default     = "gcp-rag-app"
}

variable "container_image" {
  type        = string
  description = "Full Artifact Registry image URI, e.g. us-central1-docker.pkg.dev/PROJECT/rag/gcp-rag-app:TAG"
}

variable "qdrant_url" {
  type        = string
  description = "Qdrant Cloud HTTPS URL (include port if non-default), e.g. https://xxx.aws.cloud.qdrant.io:6333"
}

variable "qdrant_secret_id" {
  type        = string
  description = "Secret Manager secret id containing the Qdrant API key (create before apply)."
}

variable "groq_secret_id" {
  type        = string
  description = "Secret Manager secret id containing the Groq API key used for answer generation."
}

variable "app_api_keys_secret_id" {
  type        = string
  default     = ""
  description = "Optional Secret Manager secret id: comma-separated values for API_KEYS (X-API-Key). Leave empty to disable API key auth."
}

variable "allow_unauthenticated" {
  type        = bool
  default     = false
  description = "If true, anyone on the internet can call the service URL (still needs X-API-Key when API_KEYS is set)."
}

variable "cors_origins" {
  type        = string
  default     = ""
  description = "Comma-separated origins for CORS, e.g. https://gcp-rag-app-xxxxx.run.app"
}

variable "min_instances" {
  type        = number
  default     = 0
  description = "Set to 1 to avoid cold starts (adds baseline cost)."
}
