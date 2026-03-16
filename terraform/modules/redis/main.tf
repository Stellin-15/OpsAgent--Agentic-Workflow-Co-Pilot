terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" { type = string }
variable "region" { type = string }
variable "vpc_network" { type = string }

resource "google_redis_instance" "opsagent" {
  name           = "opsagent-redis"
  tier           = "STANDARD_HA"    # High availability with failover replica
  memory_size_gb = 2
  region         = var.region

  authorized_network = var.vpc_network
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version = "REDIS_7_0"

  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
      }
    }
  }
}

output "host" {
  value = google_redis_instance.opsagent.host
}

output "port" {
  value = google_redis_instance.opsagent.port
}

output "redis_url" {
  value = "redis://${google_redis_instance.opsagent.host}:${google_redis_instance.opsagent.port}/0"
}
