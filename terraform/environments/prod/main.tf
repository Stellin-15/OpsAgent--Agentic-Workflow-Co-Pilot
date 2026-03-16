terraform {
  required_version = ">= 1.9"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state in GCS — team members share a single state file
  backend "gcs" {
    bucket = "opsagent-terraform-state"
    prefix = "prod"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─── Variables ────────────────────────────────────────────────────────────────

variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "image_tag" {
  type        = string
  description = "Docker image tag to deploy (e.g. sha-abc1234)"
}

# ─── VPC ──────────────────────────────────────────────────────────────────────

resource "google_compute_network" "opsagent" {
  name                    = "opsagent-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "opsagent" {
  name          = "opsagent-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.opsagent.id
}

# Private Services Access for Cloud SQL + Memorystore
resource "google_compute_global_address" "private_services" {
  name          = "opsagent-private-services"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.opsagent.id
}

resource "google_service_networking_connection" "private" {
  network                 = google_compute_network.opsagent.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_services.name]
}

# ─── Modules ──────────────────────────────────────────────────────────────────

module "postgres" {
  source      = "../../modules/postgres"
  project_id  = var.project_id
  region      = var.region
  vpc_network = google_compute_network.opsagent.id
  db_password = var.db_password
  depends_on  = [google_service_networking_connection.private]
}

module "redis" {
  source      = "../../modules/redis"
  project_id  = var.project_id
  region      = var.region
  vpc_network = google_compute_network.opsagent.id
  depends_on  = [google_service_networking_connection.private]
}

# ─── Cloud Run — API ──────────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "api" {
  name     = "opsagent-api"
  location = var.region

  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 10
    }

    containers {
      image = "ghcr.io/your-org/opsagent:${var.image_tag}"

      ports {
        container_port = 8000
      }

      env {
        name  = "ENVIRONMENT"
        value = "production"
      }
      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_url.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_URL"
        value = "redis://${module.redis.host}:${module.redis.port}/0"
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      liveness_probe {
        http_get {
          path = "/health/live"
          port = 8000
        }
        initial_delay_seconds = 15
        period_seconds        = 30
      }

      startup_probe {
        http_get {
          path = "/health/ready"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 12
      }
    }

    vpc_access {
      network_interfaces {
        network    = google_compute_network.opsagent.id
        subnetwork = google_compute_subnetwork.opsagent.id
      }
      egress = "ALL_TRAFFIC"
    }
  }
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ─── Secret Manager ───────────────────────────────────────────────────────────

resource "google_secret_manager_secret" "db_url" {
  secret_id = "opsagent-database-url"
  replication {
    auto {}
  }
}

# ─── Outputs ──────────────────────────────────────────────────────────────────

output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "db_connection_name" {
  value = module.postgres.connection_name
}

output "redis_url" {
  value     = module.redis.redis_url
  sensitive = true
}
