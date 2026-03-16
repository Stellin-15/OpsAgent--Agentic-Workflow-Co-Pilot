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
variable "db_name" { type = string, default = "opsagent" }
variable "db_tier" { type = string, default = "db-custom-2-7680" }  # 2 vCPU, 7.5 GB

resource "google_sql_database_instance" "opsagent" {
  name             = "opsagent-postgres"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier = var.db_tier

    backup_configuration {
      enabled            = true
      start_time         = "02:00"
      transaction_log_retention_days = 7
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.vpc_network
    }

    database_flags {
      name  = "max_connections"
      value = "500"
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "opsagent" {
  name     = var.db_name
  instance = google_sql_database_instance.opsagent.name
}

resource "google_sql_user" "opsagent" {
  name     = "opsagent"
  instance = google_sql_database_instance.opsagent.name
  password = var.db_password
}

# Enable pgvector extension via Cloud SQL
resource "null_resource" "enable_pgvector" {
  depends_on = [google_sql_database.opsagent]
  provisioner "local-exec" {
    command = <<-EOT
      gcloud sql connect ${google_sql_database_instance.opsagent.name} \
        --database=${var.db_name} \
        --user=opsagent \
        -- -c "CREATE EXTENSION IF NOT EXISTS vector;"
    EOT
  }
}

variable "vpc_network" { type = string }
variable "db_password" {
  type      = string
  sensitive = true
}

output "connection_name" {
  value = google_sql_database_instance.opsagent.connection_name
}

output "private_ip" {
  value = google_sql_database_instance.opsagent.private_ip_address
}
