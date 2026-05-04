output "cloud_run_uri" {
  value       = google_cloud_run_v2_service.rag.uri
  description = "HTTPS URL of the service (append /ui/ for the web UI)."
}

output "service_account" {
  value       = google_service_account.rag.email
  description = "Runtime service account (Vertex AI User attached)."
}
