"""
Cloud Discovery Service
======================

Service for discovering and inventorying resources across multiple cloud providers.
Supports AWS, Azure, and Google Cloud Platform.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Cloud provider SDKs
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.resource import ResourceManagementClient
    from azure.mgmt.compute import ComputeManagementClient
    from azure.mgmt.storage import StorageManagementClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from google.cloud import resource_manager
    from google.cloud import compute_v1
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"

class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORK = "network"
    SERVERLESS = "serverless"

@dataclass
class CloudResource:
    """Represents a cloud resource across providers."""
    id: str
    name: str
    provider: CloudProvider
    resource_type: ResourceType
    region: str
    state: str
    created_date: datetime
    tags: Dict[str, str]
    cost_estimate: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['provider'] = self.provider.value
        data['resource_type'] = self.resource_type.value
        data['created_date'] = self.created_date.isoformat()
        return data

class CloudDiscoveryService:
    """Service for discovering resources across multiple cloud providers."""
    
    def __init__(self):
        self.aws_client = None
        self.azure_client = None
        self.gcp_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize cloud provider clients."""
        # AWS
        if AWS_AVAILABLE:
            try:
                self.aws_session = boto3.Session()
                logger.info("AWS client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AWS client: {e}")
        
        # Azure
        if AZURE_AVAILABLE:
            try:
                self.azure_credential = DefaultAzureCredential()
                logger.info("Azure client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure client: {e}")
        
        # GCP
        if GCP_AVAILABLE:
            try:
                self.gcp_resource_client = resource_manager.Client()
                logger.info("GCP client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize GCP client: {e}")
    
    async def discover_all_resources(self) -> List[CloudResource]:
        """Discover resources across all configured cloud providers."""
        all_resources = []
        
        # Discover AWS resources
        if AWS_AVAILABLE and self.aws_session:
            aws_resources = await self._discover_aws_resources()
            all_resources.extend(aws_resources)
        
        # Discover Azure resources
        if AZURE_AVAILABLE and self.azure_credential:
            azure_resources = await self._discover_azure_resources()
            all_resources.extend(azure_resources)
        
        # Discover GCP resources
        if GCP_AVAILABLE and self.gcp_resource_client:
            gcp_resources = await self._discover_gcp_resources()
            all_resources.extend(gcp_resources)
        
        logger.info(f"Discovered {len(all_resources)} total resources")
        return all_resources
    
    async def _discover_aws_resources(self) -> List[CloudResource]:
        """Discover AWS resources."""
        resources = []
        
        try:
            # Get all AWS regions
            ec2 = self.aws_session.client('ec2', region_name='us-east-1')
            regions = [region['RegionName'] for region in ec2.describe_regions()['Regions']]
            
            # Discover resources in each region
            for region in regions[:3]:  # Limit to first 3 regions for demo
                region_resources = await self._discover_aws_region_resources(region)
                resources.extend(region_resources)
        
        except Exception as e:
            logger.error(f"Error discovering AWS resources: {e}")
        
        return resources
    
    async def _discover_aws_region_resources(self, region: str) -> List[CloudResource]:
        """Discover AWS resources in a specific region."""
        resources = []
        
        try:
            # EC2 Instances
            ec2 = self.aws_session.client('ec2', region_name=region)
            instances = ec2.describe_instances()
            
            for reservation in instances['Reservations']:
                for instance in reservation['Instances']:
                    resource = CloudResource(
                        id=instance['InstanceId'],
                        name=self._get_aws_resource_name(instance.get('Tags', [])),
                        provider=CloudProvider.AWS,
                        resource_type=ResourceType.COMPUTE,
                        region=region,
                        state=instance['State']['Name'],
                        created_date=instance['LaunchTime'],
                        tags=self._parse_aws_tags(instance.get('Tags', [])),
                        cost_estimate=self._estimate_ec2_cost(instance),
                        metadata={
                            'instance_type': instance['InstanceType'],
                            'platform': instance.get('Platform', 'linux'),
                            'vpc_id': instance.get('VpcId'),
                            'subnet_id': instance.get('SubnetId')
                        }
                    )
                    resources.append(resource)
            
            # RDS Instances
            try:
                rds = self.aws_session.client('rds', region_name=region)
                db_instances = rds.describe_db_instances()
                
                for db in db_instances['DBInstances']:
                    resource = CloudResource(
                        id=db['DBInstanceIdentifier'],
                        name=db['DBInstanceIdentifier'],
                        provider=CloudProvider.AWS,
                        resource_type=ResourceType.DATABASE,
                        region=region,
                        state=db['DBInstanceStatus'],
                        created_date=db['InstanceCreateTime'],
                        tags=self._parse_aws_tags(db.get('TagList', [])),
                        cost_estimate=self._estimate_rds_cost(db),
                        metadata={
                            'engine': db['Engine'],
                            'instance_class': db['DBInstanceClass'],
                            'allocated_storage': db['AllocatedStorage'],
                            'multi_az': db['MultiAZ']
                        }
                    )
                    resources.append(resource)
            except Exception as e:
                logger.warning(f"Could not discover RDS in {region}: {e}")
            
            # S3 Buckets (global service, only check in us-east-1)
            if region == 'us-east-1':
                try:
                    s3 = self.aws_session.client('s3')
                    buckets = s3.list_buckets()
                    
                    for bucket in buckets['Buckets']:
                        # Get bucket location
                        try:
                            location = s3.get_bucket_location(Bucket=bucket['Name'])
                            bucket_region = location['LocationConstraint'] or 'us-east-1'
                        except:
                            bucket_region = 'us-east-1'
                        
                        resource = CloudResource(
                            id=bucket['Name'],
                            name=bucket['Name'],
                            provider=CloudProvider.AWS,
                            resource_type=ResourceType.STORAGE,
                            region=bucket_region,
                            state='active',
                            created_date=bucket['CreationDate'],
                            tags={},
                            cost_estimate=self._estimate_s3_cost(bucket['Name']),
                            metadata={
                                'bucket_type': 'standard',
                                'versioning': 'unknown'
                            }
                        )
                        resources.append(resource)
                except Exception as e:
                    logger.warning(f"Could not discover S3 buckets: {e}")
        
        except Exception as e:
            logger.error(f"Error discovering AWS resources in {region}: {e}")
        
        return resources
    
    async def _discover_azure_resources(self) -> List[CloudResource]:
        """Discover Azure resources."""
        resources = []
        
        try:
            # This is a simplified example - in production, you'd need subscription ID
            subscription_id = "your-subscription-id"  # Should come from config
            
            # Resource Management Client
            resource_client = ResourceManagementClient(
                self.azure_credential, subscription_id
            )
            
            # Get all resource groups
            resource_groups = resource_client.resource_groups.list()
            
            for rg in resource_groups:
                # Get resources in each resource group
                rg_resources = resource_client.resources.list_by_resource_group(rg.name)
                
                for resource in rg_resources:
                    azure_resource = CloudResource(
                        id=resource.id,
                        name=resource.name,
                        provider=CloudProvider.AZURE,
                        resource_type=self._map_azure_resource_type(resource.type),
                        region=resource.location,
                        state='active',  # Azure doesn't have a simple state field
                        created_date=datetime.now(),  # Would need to get from resource details
                        tags=resource.tags or {},
                        cost_estimate=0.0,  # Would need Azure Cost Management API
                        metadata={
                            'resource_type': resource.type,
                            'resource_group': rg.name,
                            'kind': getattr(resource, 'kind', None)
                        }
                    )
                    resources.append(azure_resource)
        
        except Exception as e:
            logger.error(f"Error discovering Azure resources: {e}")
        
        return resources
    
    async def _discover_gcp_resources(self) -> List[CloudResource]:
        """Discover GCP resources."""
        resources = []
        
        try:
            # This is a simplified example
            project_id = "your-project-id"  # Should come from config
            
            # Compute instances
            compute_client = compute_v1.InstancesClient()
            
            # Get all zones
            zones_client = compute_v1.ZonesClient()
            zones = zones_client.list(project=project_id)
            
            for zone in zones:
                instances = compute_client.list(project=project_id, zone=zone.name)
                
                for instance in instances:
                    gcp_resource = CloudResource(
                        id=str(instance.id),
                        name=instance.name,
                        provider=CloudProvider.GCP,
                        resource_type=ResourceType.COMPUTE,
                        region=zone.region.split('/')[-1],
                        state=instance.status,
                        created_date=datetime.fromisoformat(instance.creation_timestamp.rstrip('Z')),
                        tags=dict(instance.labels) if instance.labels else {},
                        cost_estimate=0.0,  # Would need GCP Billing API
                        metadata={
                            'machine_type': instance.machine_type.split('/')[-1],
                            'zone': zone.name,
                            'network_interfaces': len(instance.network_interfaces)
                        }
                    )
                    resources.append(gcp_resource)
        
        except Exception as e:
            logger.error(f"Error discovering GCP resources: {e}")
        
        return resources
    
    def _get_aws_resource_name(self, tags: List[Dict]) -> str:
        """Extract name from AWS tags."""
        for tag in tags:
            if tag['Key'] == 'Name':
                return tag['Value']
        return 'Unnamed'
    
    def _parse_aws_tags(self, tags: List[Dict]) -> Dict[str, str]:
        """Parse AWS tags into dictionary."""
        return {tag['Key']: tag['Value'] for tag in tags}
    
    def _estimate_ec2_cost(self, instance: Dict) -> float:
        """Estimate EC2 instance cost (simplified)."""
        # This is a very simplified cost estimation
        # In production, you'd use AWS Pricing API or Cost Explorer
        instance_type = instance['InstanceType']
        
        # Basic hourly rates (simplified)
        rates = {
            't2.micro': 0.0116,
            't2.small': 0.023,
            't2.medium': 0.046,
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            'm5.large': 0.096,
            'm5.xlarge': 0.192,
            'c5.large': 0.085,
            'c5.xlarge': 0.17
        }
        
        hourly_rate = rates.get(instance_type, 0.1)  # Default rate
        
        # Estimate monthly cost (24 hours * 30 days)
        return hourly_rate * 24 * 30
    
    def _estimate_rds_cost(self, db_instance: Dict) -> float:
        """Estimate RDS instance cost (simplified)."""
        instance_class = db_instance['DBInstanceClass']
        
        # Basic monthly rates (simplified)
        rates = {
            'db.t3.micro': 15.0,
            'db.t3.small': 30.0,
            'db.t3.medium': 60.0,
            'db.m5.large': 140.0,
            'db.m5.xlarge': 280.0
        }
        
        return rates.get(instance_class, 100.0)  # Default rate
    
    def _estimate_s3_cost(self, bucket_name: str) -> float:
        """Estimate S3 bucket cost (simplified)."""
        # This would require getting bucket size and request metrics
        # For demo, return a random estimate
        import random
        return random.uniform(10.0, 500.0)
    
    def _map_azure_resource_type(self, azure_type: str) -> ResourceType:
        """Map Azure resource type to our enum."""
        type_mapping = {
            'Microsoft.Compute/virtualMachines': ResourceType.COMPUTE,
            'Microsoft.Storage/storageAccounts': ResourceType.STORAGE,
            'Microsoft.Sql/servers': ResourceType.DATABASE,
            'Microsoft.Network/virtualNetworks': ResourceType.NETWORK,
            'Microsoft.Web/sites': ResourceType.SERVERLESS
        }
        
        return type_mapping.get(azure_type, ResourceType.COMPUTE)
    
    async def get_resource_utilization(self, resource_id: str, provider: CloudProvider) -> Dict[str, Any]:
        """Get resource utilization metrics."""
        try:
            if provider == CloudProvider.AWS:
                return await self._get_aws_utilization(resource_id)
            elif provider == CloudProvider.AZURE:
                return await self._get_azure_utilization(resource_id)
            elif provider == CloudProvider.GCP:
                return await self._get_gcp_utilization(resource_id)
        except Exception as e:
            logger.error(f"Error getting utilization for {resource_id}: {e}")
            return {}
    
    async def _get_aws_utilization(self, instance_id: str) -> Dict[str, Any]:
        """Get AWS CloudWatch metrics for utilization."""
        try:
            cloudwatch = self.aws_session.client('cloudwatch')
            
            # Get CPU utilization for the last 24 hours
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[
                    {
                        'Name': 'InstanceId',
                        'Value': instance_id
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Average', 'Maximum']
            )
            
            if response['Datapoints']:
                avg_cpu = sum(dp['Average'] for dp in response['Datapoints']) / len(response['Datapoints'])
                max_cpu = max(dp['Maximum'] for dp in response['Datapoints'])
                
                return {
                    'cpu_average': round(avg_cpu, 2),
                    'cpu_maximum': round(max_cpu, 2),
                    'data_points': len(response['Datapoints']),
                    'period_hours': 24
                }
            
        except Exception as e:
            logger.error(f"Error getting AWS utilization: {e}")
        
        return {'cpu_average': 0, 'cpu_maximum': 0}
    
    async def _get_azure_utilization(self, resource_id: str) -> Dict[str, Any]:
        """Get Azure Monitor metrics for utilization."""
        # Placeholder for Azure Monitor integration
        return {'cpu_average': 45.0, 'cpu_maximum': 78.0}
    
    async def _get_gcp_utilization(self, resource_id: str) -> Dict[str, Any]:
        """Get GCP Monitoring metrics for utilization."""
        # Placeholder for GCP Monitoring integration
        return {'cpu_average': 52.0, 'cpu_maximum': 85.0}

# Example usage and testing
async def main():
    """Example usage of the CloudDiscoveryService."""
    discovery = CloudDiscoveryService()
    
    print("Starting cloud resource discovery...")
    resources = await discovery.discover_all_resources()
    
    print(f"\\nDiscovered {len(resources)} resources:")
    
    # Group by provider
    by_provider = {}
    for resource in resources:
        provider = resource.provider.value
        if provider not in by_provider:
            by_provider[provider] = []
        by_provider[provider].append(resource)
    
    for provider, provider_resources in by_provider.items():
        print(f"\\n{provider.upper()}: {len(provider_resources)} resources")
        
        # Group by type
        by_type = {}
        for resource in provider_resources:
            res_type = resource.resource_type.value
            if res_type not in by_type:
                by_type[res_type] = 0
            by_type[res_type] += 1
        
        for res_type, count in by_type.items():
            print(f"  {res_type}: {count}")
    
    # Calculate total estimated cost
    total_cost = sum(resource.cost_estimate for resource in resources)
    print(f"\\nTotal estimated monthly cost: ${total_cost:.2f}")

if __name__ == "__main__":
    asyncio.run(main())