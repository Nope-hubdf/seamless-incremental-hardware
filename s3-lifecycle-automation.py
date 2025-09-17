#!/usr/bin/env python3
"""
AWS S3 Lifecycle Management Automation
Author: Himanshu Nitin Nehete
Case Study: XYZ Corporation S3 Storage Infrastructure
Institution: iHub Divyasampark, IIT Roorkee

This script automates S3 lifecycle policies for cost optimization:
- Standard to IA after 30 days
- IA to Glacier after 60 days
- Permanent deletion after 75 days
"""

import boto3
import json
import logging
import sys
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('s3-lifecycle-automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class S3LifecycleManager:
    def __init__(self, region_name='ap-south-1'):
        """Initialize S3 client with error handling"""
        try:
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.region = region_name
            logger.info(f"S3 client initialized for region: {region_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            sys.exit(1)
    
    def create_lifecycle_policy(self, bucket_name, policy_name="XYZ-Corp-Lifecycle-Policy"):
        """
        Create comprehensive lifecycle policy for cost optimization
        
        Args:
            bucket_name (str): Name of the S3 bucket
            policy_name (str): Name of the lifecycle policy
        """
        
        lifecycle_configuration = {
            'Rules': [
                {
                    'ID': f'{policy_name}-Standard-to-IA',
                    'Status': 'Enabled',
                    'Filter': {},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        }
                    ]
                },
                {
                    'ID': f'{policy_name}-IA-to-Glacier',
                    'Status': 'Enabled',
                    'Filter': {},
                    'Transitions': [
                        {
                            'Days': 60,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                },
                {
                    'ID': f'{policy_name}-Permanent-Deletion',
                    'Status': 'Enabled',
                    'Filter': {},
                    'Expiration': {
                        'Days': 75
                    }
                },
                {
                    'ID': f'{policy_name}-Multipart-Upload-Cleanup',
                    'Status': 'Enabled',
                    'Filter': {},
                    'AbortIncompleteMultipartUpload': {
                        'DaysAfterInitiation': 7
                    }
                },
                {
                    'ID': f'{policy_name}-Versioning-Cleanup',
                    'Status': 'Enabled',
                    'Filter': {},
                    'NoncurrentVersionTransitions': [
                        {
                            'NoncurrentDays': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'NoncurrentDays': 60,
                            'StorageClass': 'GLACIER'
                        }
                    ],
                    'NoncurrentVersionExpiration': {
                        'NoncurrentDays': 90
                    }
                }
            ]
        }
        
        try:
            response = self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration=lifecycle_configuration
            )
            
            logger.info(f"✅ Lifecycle policy applied successfully to bucket: {bucket_name}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                logger.error(f"❌ Bucket {bucket_name} does not exist")
            elif error_code == 'AccessDenied':
                logger.error(f"❌ Access denied to bucket {bucket_name}")
            else:
                logger.error(f"❌ Failed to apply lifecycle policy: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return False
    
    def get_lifecycle_policy(self, bucket_name):
        """Get current lifecycle policy for a bucket"""
        try:
            response = self.s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
            logger.info(f"📋 Current lifecycle policy for {bucket_name}:")
            logger.info(json.dumps(response['Rules'], indent=2))
            return response['Rules']
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchLifecycleConfiguration':
                logger.info(f"📋 No lifecycle policy exists for bucket: {bucket_name}")
            else:
                logger.error(f"❌ Failed to get lifecycle policy: {str(e)}")
            return None
    
    def delete_lifecycle_policy(self, bucket_name):
        """Delete lifecycle policy from a bucket"""
        try:
            self.s3_client.delete_bucket_lifecycle(Bucket=bucket_name)
            logger.info(f"🗑️ Lifecycle policy deleted from bucket: {bucket_name}")
            return True
        except ClientError as e:
            logger.error(f"❌ Failed to delete lifecycle policy: {str(e)}")
            return False
    
    def list_buckets_with_lifecycle(self):
        """List all buckets and their lifecycle policy status"""
        try:
            response = self.s3_client.list_buckets()
            buckets = response['Buckets']
            
            logger.info("📊 Bucket Lifecycle Policy Status:")
            logger.info("-" * 60)
            
            for bucket in buckets:
                bucket_name = bucket['Name']
                try:
                    self.s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
                    status = "✅ Has Lifecycle Policy"
                except ClientError as e:
                    if e.response['Error']['Code'] == 'NoSuchLifecycleConfiguration':
                        status = "❌ No Lifecycle Policy"
                    else:
                        status = "⚠️ Error checking policy"
                
                logger.info(f"{bucket_name:<30} | {status}")
            
            logger.info("-" * 60)
            
        except ClientError as e:
            logger.error(f"❌ Failed to list buckets: {str(e)}")
    
    def calculate_cost_savings(self, bucket_name):
        """Calculate estimated cost savings from lifecycle policies"""
        try:
            # Get bucket size (simplified calculation)
            response = self.s3_client.list_objects_v2(Bucket=bucket_name)
            
            if 'Contents' not in response:
                logger.info(f"📊 Bucket {bucket_name} is empty")
                return
            
            total_size_gb = sum(obj['Size'] for obj in response['Contents']) / (1024**3)
            
            # Cost calculations (approximate)
            standard_cost = total_size_gb * 0.023  # $0.023 per GB for Standard
            ia_cost = total_size_gb * 0.0125      # $0.0125 per GB for Standard-IA
            glacier_cost = total_size_gb * 0.004   # $0.004 per GB for Glacier
            
            monthly_savings = (standard_cost - ia_cost - glacier_cost)
            annual_savings = monthly_savings * 12
            
            logger.info(f"💰 Cost Analysis for {bucket_name}:")
            logger.info(f"   Total Size: {total_size_gb:.2f} GB")
            logger.info(f"   Standard Storage Cost: ${standard_cost:.2f}/month")
            logger.info(f"   With Lifecycle Policies: ${ia_cost + glacier_cost:.2f}/month")
            logger.info(f"   Monthly Savings: ${monthly_savings:.2f}")
            logger.info(f"   Annual Savings: ${annual_savings:.2f}")
            
        except ClientError as e:
            logger.error(f"❌ Failed to calculate cost savings: {str(e)}")

def main():
    """Main execution function"""
    logger.info("🚀 Starting S3 Lifecycle Management Automation")
    
    # Initialize lifecycle manager
    lifecycle_manager = S3LifecycleManager()
    
    # Example usage - replace with your bucket names
    bucket_names = [
        "xyz-corp-storage-example",
        "xyz-corp-website-example"
    ]
    
    # Apply lifecycle policies to multiple buckets
    for bucket_name in bucket_names:
        logger.info(f"🔄 Processing bucket: {bucket_name}")
        
        # Check if bucket exists first
        try:
            lifecycle_manager.s3_client.head_bucket(Bucket=bucket_name)
            
            # Apply lifecycle policy
            if lifecycle_manager.create_lifecycle_policy(bucket_name):
                # Get and display current policy
                lifecycle_manager.get_lifecycle_policy(bucket_name)
                
                # Calculate cost savings
                lifecycle_manager.calculate_cost_savings(bucket_name)
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"⚠️ Bucket {bucket_name} does not exist, skipping...")
            else:
                logger.error(f"❌ Error accessing bucket {bucket_name}: {str(e)}")
        
        logger.info("-" * 80)
    
    # List all buckets and their lifecycle status
    lifecycle_manager.list_buckets_with_lifecycle()
    
    logger.info("✅ S3 Lifecycle Management Automation completed!")

if __name__ == "__main__":
    main()