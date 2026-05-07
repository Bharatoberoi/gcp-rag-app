## How to Deploy a Model/App to GCP - Step by Step

Here's exactly what I did, explained simply:

---

### **Step 1: Create Container Registry (Store for Docker Image)**
```
gcloud artifacts repositories create rag --repository-format=docker --location=us-central1
```
**What it does:** Creates a storage place on GCP to hold your application's Docker image (like a folder to store your packaged app).

---

### **Step 2: Build & Push Docker Image**
```
gcloud builds submit --tag us-central1-docker.pkg.dev/test1-493607/rag/gcp-rag-app:latest .
```
**What it does:** 
- Takes the Dockerfile from your local project
- Builds it on GCP cloud servers (not on your computer)
- Stores the built image in the registry we created in Step 1

---

### **Step 3: Prepare Terraform Configuration File**
```
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
```
**What it does:** Creates a config file where you tell GCP:
- Which project to use
- Which Docker image to run
- How much CPU/memory it needs
- Security settings
- Database credentials

**Updated values:**
- `project_id` = your GCP project
- `container_image` = the image we built in Step 2
- `qdrant_url` = database address
- `qdrant_secret_id` = credential storage location
- `groq_secret_id` = LLM API key storage location

---

### **Step 4: Store Secrets Safely**
```
gcloud secrets create rag-qdrant-api-key --data-file=qdrant_key.txt
gcloud secrets create rag-groq-api-key --data-file=groq_key.txt
```
**What it does:** Stores sensitive info (database/API keys) securely in GCP Secret Manager so it's not exposed in code.

---

### **Step 5: Initialize Terraform**
```
terraform init
```
**What it does:** Downloads Terraform plugins and prepares to read your config file.

---

### **Step 6: Plan (Preview What Will Be Created)**
```
terraform plan
```
**What it does:** Shows what resources Terraform will create on GCP (like a preview before clicking "buy").

---

### **Step 7: Deploy (Create Everything on GCP)**
```
terraform apply -auto-approve
```
**What it does:** Terraform creates on GCP:
- **Service Account** = special user account for your app
- **IAM Roles** = permissions (like a key card)
- **Cloud Run Service** = the actual running app
- **Networking** = makes it reachable from the internet

---

### **Step 8: Configure Public Access**
```
allow_unauthenticated = true
terraform apply -auto-approve
```
**What it does:** Allows anyone on the internet to reach your app's URL (like unlocking the front door).

---

### **Step 9: Fix Any Issues**
When the app wouldn't talk to the database:
```
gcloud secrets versions add rag-qdrant-api-key --data-file=new_key.txt
gcloud run services update gcp-rag-app --image <latest-image>
```
**What it does:** Updates the database password and restarts the app with the new credentials.

---

## **Summary in One Picture**

```
Your Local Code
      |
Step 2: Build Docker Image (on GCP)
      |
Store in Registry
      |
Step 7: Terraform creates Cloud Run + IAM + Secrets
      |
App is now running publicly on:
https://gcp-rag-app-1042144655480.us-central1.run.app
```

---

## **Key GCP Services Used**

| Service | Purpose |
|---------|---------|
| **Artifact Registry** | Store Docker images |
| **Cloud Build** | Build Docker images |
| **Cloud Run** | Run your app (serverless) |
| **IAM** | Control who can do what |
| **Secret Manager** | Store passwords/API keys safely |
| **Terraform** | Infrastructure as Code (automate creation) |
| **Groq** | LLM answer generation API used by your app |
| **Qdrant** | Vector database (stores document embeddings) |

---

## **Why This Approach?**

- **Repeatable** - Run the same commands again = same setup  
- **Secure** - Passwords never in code  
- **Scalable** - Auto-scales from 0 to 10 instances based on traffic  
- **Production-ready** - Proper logging, health checks, monitoring  

Does this make sense? Any specific step you want me to explain more?
