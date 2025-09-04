# Multi-Cloud Cost Optimizer Infrastructure
# Terraform configuration for deploying the platform across multiple cloud providers

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "multi-cloud-cost-optimizer"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "azure_location" {
  description = "Azure location"
  type        = string
  default     = "East US"
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = local.common_tags
  }
}

# Azure Provider
provider "azurerm" {
  features {}
}

# GCP Provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Kubernetes Provider (will be configured after EKS cluster creation)
provider "kubernetes" {
  host                   = module.aws_infrastructure.eks_cluster_endpoint
  cluster_ca_certificate = base64decode(module.aws_infrastructure.eks_cluster_ca_certificate)
  token                  = module.aws_infrastructure.eks_cluster_token
}

# AWS Infrastructure Module
module "aws_infrastructure" {
  source = "./modules/aws"
  
  environment    = var.environment
  project_name   = var.project_name
  aws_region     = var.aws_region
  
  # VPC Configuration
  vpc_cidr = "10.0.0.0/16"
  
  # EKS Configuration
  eks_cluster_version = "1.27"
  eks_node_groups = {
    main = {
      instance_types = ["t3.medium"]
      min_size      = 1
      max_size      = 5
      desired_size  = 2
    }
  }
  
  # RDS Configuration
  rds_instance_class = "db.t3.micro"
  rds_allocated_storage = 20
  
  tags = local.common_tags
}

# Azure Infrastructure Module
module "azure_infrastructure" {
  source = "./modules/azure"
  
  environment     = var.environment
  project_name    = var.project_name
  azure_location  = var.azure_location
  
  # AKS Configuration
  aks_node_count = 2
  aks_vm_size    = "Standard_B2s"
  
  # Database Configuration
  postgres_sku_name = "B_Gen5_1"
  
  tags = local.common_tags
}

# GCP Infrastructure Module
module "gcp_infrastructure" {
  source = "./modules/gcp"
  
  environment    = var.environment
  project_name   = var.project_name
  gcp_project_id = var.gcp_project_id
  gcp_region     = var.gcp_region
  
  # GKE Configuration
  gke_node_count = 2
  gke_machine_type = "e2-medium"
  
  # Cloud SQL Configuration
  cloudsql_tier = "db-f1-micro"
  
  labels = {
    project     = var.project_name
    environment = var.environment
  }
}

# Shared Resources Module
module "shared_resources" {
  source = "./modules/shared"
  
  environment  = var.environment
  project_name = var.project_name
  
  # Application Configuration
  app_image = "multi-cloud-cost-optimizer:latest"
  app_port  = 8000
  
  # Database Configuration
  postgres_databases = {
    aws = {
      host     = module.aws_infrastructure.rds_endpoint
      port     = 5432
      database = "cost_optimizer"
      username = module.aws_infrastructure.rds_username
      password = module.aws_infrastructure.rds_password
    }
    azure = {
      host     = module.azure_infrastructure.postgres_fqdn
      port     = 5432
      database = "cost_optimizer"
      username = module.azure_infrastructure.postgres_username
      password = module.azure_infrastructure.postgres_password
    }
    gcp = {
      host     = module.gcp_infrastructure.cloudsql_ip
      port     = 5432
      database = "cost_optimizer"
      username = module.gcp_infrastructure.cloudsql_username
      password = module.gcp_infrastructure.cloudsql_password
    }
  }
  
  # Redis Configuration
  redis_configs = {
    aws = {
      endpoint = module.aws_infrastructure.elasticache_endpoint
      port     = 6379
    }
    azure = {
      endpoint = module.azure_infrastructure.redis_hostname
      port     = 6380
    }
    gcp = {
      endpoint = module.gcp_infrastructure.memorystore_host
      port     = 6379
    }
  }
}

# Monitoring and Alerting
module "monitoring" {
  source = "./modules/monitoring"
  
  environment  = var.environment
  project_name = var.project_name
  
  # Prometheus Configuration
  prometheus_retention = "30d"
  prometheus_storage_size = "50Gi"
  
  # Grafana Configuration
  grafana_admin_password = random_password.grafana_password.result
  
  # Alert Manager Configuration
  alert_webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  
  depends_on = [
    module.aws_infrastructure,
    module.azure_infrastructure,
    module.gcp_infrastructure
  ]
}

# Security and Secrets
resource "random_password" "grafana_password" {
  length  = 16
  special = true
}

resource "random_password" "api_secret_key" {
  length  = 32
  special = true
}

# Outputs
output "aws_outputs" {
  description = "AWS infrastructure outputs"
  value = {
    vpc_id                = module.aws_infrastructure.vpc_id
    eks_cluster_name      = module.aws_infrastructure.eks_cluster_name
    eks_cluster_endpoint  = module.aws_infrastructure.eks_cluster_endpoint
    rds_endpoint         = module.aws_infrastructure.rds_endpoint
    elasticache_endpoint = module.aws_infrastructure.elasticache_endpoint
  }
}

output "azure_outputs" {
  description = "Azure infrastructure outputs"
  value = {
    resource_group_name = module.azure_infrastructure.resource_group_name
    aks_cluster_name    = module.azure_infrastructure.aks_cluster_name
    postgres_fqdn       = module.azure_infrastructure.postgres_fqdn
    redis_hostname      = module.azure_infrastructure.redis_hostname
  }
}

output "gcp_outputs" {
  description = "GCP infrastructure outputs"
  value = {
    gke_cluster_name = module.gcp_infrastructure.gke_cluster_name
    cloudsql_ip      = module.gcp_infrastructure.cloudsql_ip
    memorystore_host = module.gcp_infrastructure.memorystore_host
  }
}

output "application_urls" {
  description = "Application access URLs"
  value = {
    aws_app_url     = "https://${module.aws_infrastructure.alb_dns_name}"
    azure_app_url   = "https://${module.azure_infrastructure.app_gateway_fqdn}"
    gcp_app_url     = "https://${module.gcp_infrastructure.load_balancer_ip}"
    grafana_url     = "https://${module.monitoring.grafana_url}"
    prometheus_url  = "https://${module.monitoring.prometheus_url}"
  }
}

output "database_connections" {
  description = "Database connection strings"
  value = {
    aws_postgres   = "postgresql://${module.aws_infrastructure.rds_username}:${module.aws_infrastructure.rds_password}@${module.aws_infrastructure.rds_endpoint}:5432/cost_optimizer"
    azure_postgres = "postgresql://${module.azure_infrastructure.postgres_username}:${module.azure_infrastructure.postgres_password}@${module.azure_infrastructure.postgres_fqdn}:5432/cost_optimizer"
    gcp_postgres   = "postgresql://${module.gcp_infrastructure.cloudsql_username}:${module.gcp_infrastructure.cloudsql_password}@${module.gcp_infrastructure.cloudsql_ip}:5432/cost_optimizer"
  }
  sensitive = true
}

output "monitoring_credentials" {
  description = "Monitoring system credentials"
  value = {
    grafana_admin_password = random_password.grafana_password.result
    api_secret_key        = random_password.api_secret_key.result
  }
  sensitive = true
}