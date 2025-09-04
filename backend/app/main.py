"""
Multi-Cloud Cost Optimizer - Main Application
============================================

FastAPI application for multi-cloud cost optimization and resource management.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List, Dict, Any
import asyncio
from datetime import datetime, timedelta

# Internal imports
from app.core.config import settings
from app.core.database import engine, Base
from app.api.routes import (
    cloud_resources,
    cost_analytics,
    optimization,
    alerts,
    reports
)
from app.services.cloud_discovery import CloudDiscoveryService
from app.services.cost_analyzer import CostAnalyzer
from app.services.ml_optimizer import MLOptimizer
from app.core.scheduler import start_background_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Multi-Cloud Cost Optimizer...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Initialize services
    app.state.discovery_service = CloudDiscoveryService()
    app.state.cost_analyzer = CostAnalyzer()
    app.state.ml_optimizer = MLOptimizer()
    
    # Start background tasks
    asyncio.create_task(start_background_tasks())
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Cloud Cost Optimizer...")

# Create FastAPI application
app = FastAPI(
    title="Multi-Cloud Cost Optimizer",
    description="AI-powered multi-cloud cost optimization platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info."""
    # In production, implement proper JWT validation
    # For demo purposes, we'll use a simple token check
    if credentials.credentials != settings.API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return {"user_id": "demo_user", "permissions": ["read", "write"]}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "cloud_apis": "connected",
            "ml_models": "loaded"
        }
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Cloud Cost Optimizer API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "cloud_resources": "/api/v1/resources",
            "cost_analytics": "/api/v1/costs",
            "optimization": "/api/v1/optimize",
            "alerts": "/api/v1/alerts",
            "reports": "/api/v1/reports"
        }
    }

# Dashboard summary endpoint
@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary(user=Depends(get_current_user)):
    """Get dashboard summary with key metrics."""
    try:
        # Get current month's data
        current_month = datetime.now().replace(day=1)
        
        # Mock data for demonstration
        summary = {
            "total_monthly_cost": 45678.90,
            "cost_savings": 12345.67,
            "savings_percentage": 27.0,
            "active_resources": 1247,
            "optimization_opportunities": 23,
            "alerts_count": 5,
            "top_spending_services": [
                {"service": "EC2", "cost": 15678.90, "percentage": 34.3},
                {"service": "RDS", "cost": 8901.23, "percentage": 19.5},
                {"service": "S3", "cost": 5432.10, "percentage": 11.9},
                {"service": "Lambda", "cost": 3456.78, "percentage": 7.6},
                {"service": "Others", "cost": 12209.89, "percentage": 26.7}
            ],
            "cost_trend": [
                {"date": "2024-08-01", "cost": 42000},
                {"date": "2024-08-08", "cost": 44500},
                {"date": "2024-08-15", "cost": 46200},
                {"date": "2024-08-22", "cost": 45800},
                {"date": "2024-08-29", "cost": 45678}
            ],
            "cloud_distribution": {
                "aws": {"cost": 25678.90, "percentage": 56.2},
                "azure": {"cost": 12345.67, "percentage": 27.0},
                "gcp": {"cost": 7654.33, "percentage": 16.8}
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Real-time metrics endpoint
@app.get("/api/v1/metrics/realtime")
async def get_realtime_metrics(user=Depends(get_current_user)):
    """Get real-time metrics for monitoring."""
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_hourly_spend": 1.89,
            "active_instances": {
                "aws": 45,
                "azure": 23,
                "gcp": 12
            },
            "cpu_utilization": {
                "average": 67.5,
                "peak": 89.2,
                "idle_resources": 8
            },
            "storage_usage": {
                "total_gb": 15678,
                "growth_rate": 2.3,
                "unused_storage": 1234
            },
            "network_traffic": {
                "ingress_gb": 234.5,
                "egress_gb": 567.8,
                "cost_impact": 45.67
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Cost optimization recommendations
@app.get("/api/v1/recommendations")
async def get_optimization_recommendations(user=Depends(get_current_user)):
    """Get AI-powered optimization recommendations."""
    try:
        recommendations = [
            {
                "id": "rec_001",
                "type": "rightsizing",
                "priority": "high",
                "potential_savings": 1234.56,
                "resource": "i3.2xlarge instance in us-east-1",
                "recommendation": "Downsize to i3.xlarge based on 30-day usage analysis",
                "confidence": 92,
                "impact": "No performance impact expected",
                "implementation": "Schedule during maintenance window"
            },
            {
                "id": "rec_002",
                "type": "reserved_instance",
                "priority": "medium",
                "potential_savings": 2345.67,
                "resource": "RDS MySQL instances",
                "recommendation": "Purchase 1-year reserved instances for consistent workloads",
                "confidence": 87,
                "impact": "37% cost reduction",
                "implementation": "Immediate purchase recommended"
            },
            {
                "id": "rec_003",
                "type": "storage_optimization",
                "priority": "medium",
                "potential_savings": 567.89,
                "resource": "S3 buckets with infrequent access",
                "recommendation": "Move to S3 Intelligent Tiering",
                "confidence": 95,
                "impact": "Automatic cost optimization",
                "implementation": "Apply lifecycle policies"
            },
            {
                "id": "rec_004",
                "type": "idle_resources",
                "priority": "high",
                "potential_savings": 890.12,
                "resource": "Unused load balancers and NAT gateways",
                "recommendation": "Remove idle resources identified over 7 days",
                "confidence": 99,
                "impact": "No service disruption",
                "implementation": "Safe to remove immediately"
            }
        ]
        
        return {
            "total_recommendations": len(recommendations),
            "total_potential_savings": sum(r["potential_savings"] for r in recommendations),
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Trigger optimization action
@app.post("/api/v1/optimize/execute")
async def execute_optimization(
    recommendation_id: str,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """Execute an optimization recommendation."""
    try:
        # Add background task to execute optimization
        background_tasks.add_task(execute_optimization_task, recommendation_id, user["user_id"])
        
        return {
            "status": "accepted",
            "message": f"Optimization {recommendation_id} queued for execution",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing optimization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

async def execute_optimization_task(recommendation_id: str, user_id: str):
    """Background task to execute optimization."""
    logger.info(f"Executing optimization {recommendation_id} for user {user_id}")
    
    # Simulate optimization execution
    await asyncio.sleep(5)
    
    logger.info(f"Optimization {recommendation_id} completed successfully")

# Include API routers
app.include_router(
    cloud_resources.router,
    prefix="/api/v1/resources",
    tags=["Cloud Resources"]
)

app.include_router(
    cost_analytics.router,
    prefix="/api/v1/costs",
    tags=["Cost Analytics"]
)

app.include_router(
    optimization.router,
    prefix="/api/v1/optimize",
    tags=["Optimization"]
)

app.include_router(
    alerts.router,
    prefix="/api/v1/alerts",
    tags=["Alerts"]
)

app.include_router(
    reports.router,
    prefix="/api/v1/reports",
    tags=["Reports"]
)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Resource not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )