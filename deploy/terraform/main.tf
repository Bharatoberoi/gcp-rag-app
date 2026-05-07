locals {
  cors = trimspace(var.cors_origins)
}

resource "google_service_account" "rag" {
  account_id   = "rag-app-runner"
  display_name = "RAG Cloud Run (Groq + Qdrant)"
  project      = var.project_id
}

data "google_secret_manager_secret" "qdrant" {
  secret_id = var.qdrant_secret_id
  project   = var.project_id
}

data "google_secret_manager_secret" "groq" {
  secret_id = var.groq_secret_id
  project   = var.project_id
}

resource "google_secret_manager_secret_iam_member" "qdrant_accessor" {
  secret_id = data.google_secret_manager_secret.qdrant.id
  role        = "roles/secretmanager.secretAccessor"
  member      = "serviceAccount:${google_service_account.rag.email}"
}

resource "google_secret_manager_secret_iam_member" "groq_accessor" {
  secret_id = data.google_secret_manager_secret.groq.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.rag.email}"
}

data "google_secret_manager_secret" "app_keys" {
  count     = trimspace(var.app_api_keys_secret_id) != "" ? 1 : 0
  secret_id = var.app_api_keys_secret_id
  project   = var.project_id
}

resource "google_secret_manager_secret_iam_member" "app_keys_accessor" {
  count      = trimspace(var.app_api_keys_secret_id) != "" ? 1 : 0
  secret_id  = data.google_secret_manager_secret.app_keys[0].id
  role       = "roles/secretmanager.secretAccessor"
  member     = "serviceAccount:${google_service_account.rag.email}"
}

resource "google_cloud_run_v2_service" "rag" {
  name     = var.service_name
  location = var.region
  project  = var.project_id

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_iam_member.qdrant_accessor,
    google_secret_manager_secret_iam_member.groq_accessor,
  ]

  template {
    service_account                  = google_service_account.rag.email
    max_instance_request_concurrency = 16

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = 10
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GCP_LOCATION"
        value = var.region
      }
      env {
        name  = "QDRANT_URL"
        value = var.qdrant_url
      }
      env {
        name  = "LLM_MODEL"
        value = "llama-3.1-8b-instant"
      }
      env {
        name  = "PRODUCTION_MODE"
        value = "true"
      }
      env {
        name  = "DOCS_ENABLED"
        value = "false"
      }
      env {
        name  = "QDRANT_API_KEY"
        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.qdrant.name
            version = "latest"
          }
        }
      }
      env {
        name = "GROQ_API_KEY"
        value_source {
          secret_key_ref {
            secret  = data.google_secret_manager_secret.groq.name
            version = "latest"
          }
        }
      }
      dynamic "env" {
        for_each = local.cors != "" ? [1] : []
        content {
          name  = "CORS_ORIGINS"
          value = local.cors
        }
      }
      dynamic "env" {
        for_each = trimspace(var.app_api_keys_secret_id) != "" ? [1] : []
        content {
          name = "API_KEYS"
          value_source {
            secret_key_ref {
              secret  = data.google_secret_manager_secret.app_keys[0].name
              version = "latest"
            }
          }
        }
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

resource "google_cloud_run_v2_service_iam_member" "invoker_public" {
  count    = var.allow_unauthenticated ? 1 : 0
  name     = google_cloud_run_v2_service.rag.name
  location = var.region
  project  = var.project_id
  role     = "roles/run.invoker"
  member   = "allUsers"
}
