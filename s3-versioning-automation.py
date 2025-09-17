#!/usr/bin/env python3
"""
AWS S3 Versioning Setup Automation
Author: Himanshu Nitin Nehete
Case Study: XYZ Corporation S3 Storage Infrastructure
Institution: iHub Divyasampark, IIT Roorkee

This script automates S3 versioning setup and provides recovery utilities:
- Enable/disable versioning on buckets
- List all versions of objects
- Restore previous versions
- Cleanup old versions
"""

import boto3
import json
import logging
import sys
from datetime import datetime, timedelta
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('s3-versioning-automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class S3VersioningManager:
    def __init__(self, region_name='ap-south-1'):
        """Initialize S3 client with error handling"""
        try:
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.s3_resource = boto3.resource('s3', region_name=region_name)
            self.region = region_name
            logger.info(f"S3 client initialized for region: {region_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            sys.exit(1)
    
    def enable_versioning(self, bucket_name):
        """Enable versioning on S3 bucket"""
        try:
            versioning_config = {
                'Status': 'Enabled'
            }
            
            response = self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration=versioning_config
            )
            
            logger.info(f"✅ Versioning enabled for bucket: {bucket_name}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                logger.error(f"❌ Bucket {bucket_name} does not exist")
            elif error_code == 'AccessDenied':
                logger.error(f"❌ Access denied to bucket {bucket_name}")
            else:
                logger.error(f"❌ Failed to enable versioning: {str(e)}")
            return False
    
    def suspend_versioning(self, bucket_name):
        """Suspend versioning on S3 bucket"""
        try:
            versioning_config = {
                'Status': 'Suspended'
            }
            
            response = self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration=versioning_config
            )
            
            logger.info(f"⏸️ Versioning suspended for bucket: {bucket_name}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ Failed to suspend versioning: {str(e)}")
            return False
    
    def get_versioning_status(self, bucket_name):
        """Get versioning status of a bucket"""
        try:
            response = self.s3_client.get_bucket_versioning(Bucket=bucket_name)
            status = response.get('Status', 'Disabled')
            logger.info(f"📋 Versioning status for {bucket_name}: {status}")
            return status
        except ClientError as e:
            logger.error(f"❌ Failed to get versioning status: {str(e)}")
            return None
    
    def list_object_versions(self, bucket_name, object_key=None, max_keys=100):
        """List all versions of objects in bucket"""
        try:
            params = {
                'Bucket': bucket_name,
                'MaxKeys': max_keys
            }
            
            if object_key:
                params['Prefix'] = object_key
            
            response = self.s3_client.list_object_versions(**params)
            
            versions = response.get('Versions', [])
            delete_markers = response.get('DeleteMarkers', [])
            
            logger.info(f"📋 Object versions in {bucket_name}:")
            logger.info("-" * 100)
            
            # Display current versions
            if versions:
                logger.info("CURRENT VERSIONS:")
                for version in versions:
                    size_mb = version['Size'] / (1024 * 1024)
                    logger.info(f"  {version['Key']:<40} | "
                              f"Version: {version['VersionId'][:8]}... | "
                              f"Size: {size_mb:.2f} MB | "
                              f"Modified: {version['LastModified']}")
            
            # Display delete markers
            if delete_markers:
                logger.info("\nDELETE MARKERS:")
                for marker in delete_markers:
                    logger.info(f"  {marker['Key']:<40} | "
                              f"Version: {marker['VersionId'][:8]}... | "
                              f"Modified: {marker['LastModified']}")
            
            logger.info("-" * 100)
            logger.info(f"Total Versions: {len(versions)}, Delete Markers: {len(delete_markers)}")
            
            return versions, delete_markers
            
        except ClientError as e:
            logger.error(f"❌ Failed to list object versions: {str(e)}")
            return [], []
    
    def restore_object_version(self, bucket_name, object_key, version_id):
        """Restore a specific version of an object"""
        try:
            # Copy the specific version to make it current
            copy_source = {
                'Bucket': bucket_name,
                'Key': object_key,
                'VersionId': version_id
            }
            
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=bucket_name,
                Key=object_key
            )
            
            logger.info(f"✅ Restored {object_key} to version {version_id[:8]}...")
            return True
            
        except ClientError as e:
            logger.error(f"❌ Failed to restore object version: {str(e)}")
            return False
    
    def delete_object_version(self, bucket_name, object_key, version_id):
        """Delete a specific version of an object"""
        try:
            self.s3_client.delete_object(
                Bucket=bucket_name,
                Key=object_key,
                VersionId=version_id
            )
            
            logger.info(f"🗑️ Deleted version {version_id[:8]}... of {object_key}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ Failed to delete object version: {str(e)}")
            return False
    
    def cleanup_old_versions(self, bucket_name, days_old=30, dry_run=True):
        """Clean up versions older than specified days"""
        try:
            cutoff_date = datetime.now().replace(tzinfo=None) - timedelta(days=days_old)
            
            response = self.s3_client.list_object_versions(Bucket=bucket_name)
            versions = response.get('Versions', [])
            
            old_versions = []
            for version in versions:
                # Skip current versions (IsLatest=True)
                if version.get('IsLatest', False):
                    continue
                
                last_modified = version['LastModified'].replace(tzinfo=None)
                if last_modified < cutoff_date:
                    old_versions.append(version)
            
            logger.info(f"🔍 Found {len(old_versions)} versions older than {days_old} days")
            
            if dry_run:
                logger.info("🚨 DRY RUN MODE - No versions will be deleted")
                for version in old_versions[:10]:  # Show first 10
                    logger.info(f"  Would delete: {version['Key']} | "
                              f"Version: {version['VersionId'][:8]}... | "
                              f"Size: {version['Size']} bytes | "
                              f"Modified: {version['LastModified']}")
                if len(old_versions) > 10:
                    logger.info(f"  ... and {len(old_versions) - 10} more versions")
            else:
                deleted_count = 0
                for version in old_versions:
                    if self.delete_object_version(
                        bucket_name, 
                        version['Key'], 
                        version['VersionId']
                    ):
                        deleted_count += 1
                
                logger.info(f"✅ Deleted {deleted_count} old versions")
            
            return len(old_versions)
            
        except ClientError as e:
            logger.error(f"❌ Failed to cleanup old versions: {str(e)}")
            return 0
    
    def get_bucket_size_by_versions(self, bucket_name):
        """Calculate bucket size including all versions"""
        try:
            response = self.s3_client.list_object_versions(Bucket=bucket_name)
            versions = response.get('Versions', [])
            
            total_size = 0
            current_size = 0
            version_count = 0
            
            for version in versions:
                total_size += version['Size']
                if version.get('IsLatest', False):
                    current_size += version['Size']
                else:
                    version_count += 1
            
            total_size_gb = total_size / (1024**3)
            current_size_gb = current_size / (1024**3)
            version_size_gb = (total_size - current_size) / (1024**3)
            
            logger.info(f"📊 Storage Analysis for {bucket_name}:")
            logger.info(f"  Current Version Size: {current_size_gb:.3f} GB")
            logger.info(f"  Previous Versions Size: {version_size_gb:.3f} GB")
            logger.info(f"  Total Size: {total_size_gb:.3f} GB")
            logger.info(f"  Previous Versions Count: {version_count}")
            logger.info(f"  Storage Overhead: {(version_size_gb/total_size_gb)*100:.1f}%")
            
            return {
                'total_size_gb': total_size_gb,
                'current_size_gb': current_size_gb,
                'version_size_gb': version_size_gb,
                'version_count': version_count
            }
            
        except ClientError as e:
            logger.error(f"❌ Failed to calculate bucket size: {str(e)}")
            return None
    
    def create_version_report(self, bucket_name, output_file=None):
        """Create comprehensive versioning report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_file or f"versioning_report_{bucket_name}_{timestamp}.json"
        
        try:
            # Collect all data
            versioning_status = self.get_versioning_status(bucket_name)
            versions, delete_markers = self.list_object_versions(bucket_name, max_keys=1000)
            size_analysis = self.get_bucket_size_by_versions(bucket_name)
            
            report = {
                'bucket_name': bucket_name,
                'generated_at': datetime.now().isoformat(),
                'versioning_status': versioning_status,
                'summary': {
                    'total_versions': len(versions),
                    'delete_markers': len(delete_markers),
                    'unique_objects': len(set(v['Key'] for v in versions))
                },
                'size_analysis': size_analysis,
                'versions': versions[:100],  # Limit to first 100 for report size
                'delete_markers': delete_markers[:50]  # Limit to first 50
            }
            
            # Save report to file
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Versioning report saved to: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"❌ Failed to create versioning report: {str(e)}")
            return None

def main():
    """Main execution function"""
    logger.info("🚀 Starting S3 Versioning Automation")
    
    # Initialize versioning manager
    version_manager = S3VersioningManager()
    
    # Example bucket names - replace with your actual buckets
    bucket_names = [
        "xyz-corp-storage-example",
        "xyz-corp-website-example"
    ]
    
    # Process each bucket
    for bucket_name in bucket_names:
        logger.info(f"🔄 Processing bucket: {bucket_name}")
        
        try:
            # Check if bucket exists
            version_manager.s3_client.head_bucket(Bucket=bucket_name)
            
            # Enable versioning
            if version_manager.enable_versioning(bucket_name):
                # Get versioning status
                version_manager.get_versioning_status(bucket_name)
                
                # List current versions
                version_manager.list_object_versions(bucket_name, max_keys=10)
                
                # Get size analysis
                version_manager.get_bucket_size_by_versions(bucket_name)
                
                # Create comprehensive report
                version_manager.create_version_report(bucket_name)
                
                # Cleanup old versions (dry run)
                version_manager.cleanup_old_versions(bucket_name, days_old=90, dry_run=True)
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"⚠️ Bucket {bucket_name} does not exist, skipping...")
            else:
                logger.error(f"❌ Error processing bucket {bucket_name}: {str(e)}")
        
        logger.info("-" * 80)
    
    logger.info("✅ S3 Versioning Automation completed!")

if __name__ == "__main__":
    main()