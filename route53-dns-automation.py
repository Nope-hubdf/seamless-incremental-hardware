#!/usr/bin/env python3
"""
AWS Route 53 DNS Configuration Automation
Author: Himanshu Nitin Nehete
Case Study: XYZ Corporation S3 Storage Infrastructure
Institution: iHub Divyasampark, IIT Roorkee

This script automates Route 53 DNS configuration for S3 website hosting:
- Create hosted zone for custom domain
- Configure DNS records for website
- Set up www redirection
- Configure health checks
- Generate DNS propagation report
"""

import boto3
import json
import logging
import sys
import time
import socket
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('route53-dns-automation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Route53DNSManager:
    def __init__(self, region_name='ap-south-1'):
        """Initialize Route 53 client with error handling"""
        try:
            self.route53_client = boto3.client('route53')
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.region = region_name
            logger.info("Route 53 and S3 clients initialized successfully")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS CLI.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            sys.exit(1)
    
    def create_hosted_zone(self, domain_name, comment=None):
        """Create a hosted zone for the domain"""
        if comment is None:
            comment = f"Hosted zone for {domain_name} - XYZ Corporation Case Study"
        
        try:
            response = self.route53_client.create_hosted_zone(
                Name=domain_name,
                CallerReference=str(int(time.time())),
                HostedZoneConfig={
                    'Comment': comment,
                    'PrivateZone': False
                }
            )
            
            hosted_zone_id = response['HostedZone']['Id'].replace('/hostedzone/', '')
            name_servers = response['DelegationSet']['NameServers']
            
            logger.info(f"✅ Hosted zone created successfully")
            logger.info(f"   Domain: {domain_name}")
            logger.info(f"   Hosted Zone ID: {hosted_zone_id}")
            logger.info("📋 Name Servers:")
            for ns in name_servers:
                logger.info(f"   - {ns}")
            
            return {
                'hosted_zone_id': hosted_zone_id,
                'name_servers': name_servers,
                'domain': domain_name
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'HostedZoneAlreadyExists':
                logger.warning(f"⚠️ Hosted zone for {domain_name} already exists")
                return self.get_hosted_zone_by_domain(domain_name)
            else:
                logger.error(f"❌ Failed to create hosted zone: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return None
    
    def get_hosted_zone_by_domain(self, domain_name):
        """Get hosted zone information by domain name"""
        try:
            response = self.route53_client.list_hosted_zones()
            
            for zone in response['HostedZones']:
                if zone['Name'].rstrip('.') == domain_name.rstrip('.'):
                    hosted_zone_id = zone['Id'].replace('/hostedzone/', '')
                    
                    # Get name servers
                    ns_response = self.route53_client.get_hosted_zone(Id=hosted_zone_id)
                    name_servers = ns_response['DelegationSet']['NameServers']
                    
                    return {
                        'hosted_zone_id': hosted_zone_id,
                        'name_servers': name_servers,
                        'domain': domain_name
                    }
            
            logger.warning(f"⚠️ Hosted zone for {domain_name} not found")
            return None
            
        except ClientError as e:
            logger.error(f"❌ Failed to get hosted zone: {str(e)}")
            return None
    
    def create_s3_website_records(self, hosted_zone_id, domain_name, s3_bucket_name, create_www_redirect=True):
        """Create DNS records for S3 website hosting"""
        website_endpoint = f"{s3_bucket_name}.s3-website.{self.region}.amazonaws.com"
        
        records_to_create = []
        
        # Root domain A record (ALIAS to S3 website endpoint)
        records_to_create.append({
            'Action': 'CREATE',
            'ResourceRecordSet': {
                'Name': domain_name,
                'Type': 'A',
                'AliasTarget': {
                    'HostedZoneId': self._get_s3_website_hosted_zone_id(),
                    'DNSName': website_endpoint,
                    'EvaluateTargetHealth': False
                }
            }
        })
        
        # WWW subdomain redirect (if requested)
        if create_www_redirect:
            www_bucket_name = f"www.{domain_name}"
            
            # Create www bucket for redirect
            self._create_www_redirect_bucket(www_bucket_name, s3_bucket_name)
            
            www_endpoint = f"{www_bucket_name}.s3-website.{self.region}.amazonaws.com"
            
            records_to_create.append({
                'Action': 'CREATE',
                'ResourceRecordSet': {
                    'Name': f"www.{domain_name}",
                    'Type': 'A',
                    'AliasTarget': {
                        'HostedZoneId': self._get_s3_website_hosted_zone_id(),
                        'DNSName': www_endpoint,
                        'EvaluateTargetHealth': False
                    }
                }
            })
        
        # Create the records
        try:
            response = self.route53_client.change_resource_record_sets(
                HostedZoneId=hosted_zone_id,
                ChangeBatch={
                    'Comment': f'DNS records for {domain_name} S3 website hosting',
                    'Changes': records_to_create
                }
            )
            
            change_id = response['ChangeInfo']['Id']
            
            logger.info(f"✅ DNS records created successfully")
            logger.info(f"   Root Domain: {domain_name} -> {website_endpoint}")
            if create_www_redirect:
                logger.info(f"   WWW Redirect: www.{domain_name} -> {domain_name}")
            logger.info(f"   Change ID: {change_id}")
            
            # Wait for changes to propagate
            self._wait_for_dns_propagation(change_id)
            
            return True
            
        except ClientError as e:
            logger.error(f"❌ Failed to create DNS records: {str(e)}")
            return False
    
    def _get_s3_website_hosted_zone_id(self):
        """Get the hosted zone ID for S3 website endpoints by region"""
        # S3 website hosting hosted zone IDs by region
        zone_ids = {
            'us-east-1': 'Z3AQBSTGFYJSTF',
            'us-east-2': 'Z2O1EMRO9K5GLX',
            'us-west-1': 'Z2F56UZL2M1ACD',
            'us-west-2': 'Z3BJ6K6RIION7M',
            'ap-south-1': 'Z11RGJOFQNVJUP',
            'ap-northeast-1': 'Z2M4EHUR26P7ZW',
            'ap-northeast-2': 'Z3W03O7B5YMIYP',
            'ap-southeast-1': 'Z3O0SRN1WG5ATM',
            'ap-southeast-2': 'Z1WCIGYICN2BYD',
            'eu-west-1': 'Z1BKCTXD74EZPE',
            'eu-west-2': 'Z3GKZC51ZF0DB4',
            'eu-central-1': 'Z21DNDUVLTQW6Q',
            'sa-east-1': 'Z7KQH4QJS55SO'
        }
        return zone_ids.get(self.region, 'Z3AQBSTGFYJSTF')  # Default to us-east-1
    
    def _create_www_redirect_bucket(self, www_bucket_name, target_bucket_name):
        """Create www bucket for redirection"""
        try:
            # Create bucket
            if self.region != 'us-east-1':
                self.s3_client.create_bucket(
                    Bucket=www_bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            else:
                self.s3_client.create_bucket(Bucket=www_bucket_name)
            
            # Configure redirect
            redirect_config = {
                'RedirectAllRequestsTo': {
                    'HostName': target_bucket_name.replace('.s3-website.' + self.region + '.amazonaws.com', ''),
                    'Protocol': 'http'
                }
            }
            
            self.s3_client.put_bucket_website(
                Bucket=www_bucket_name,
                WebsiteConfiguration=redirect_config
            )
            
            logger.info(f"✅ WWW redirect bucket created: {www_bucket_name}")
            
        except ClientError as e:
            if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
                logger.warning(f"⚠️ Could not create www redirect bucket: {str(e)}")
    
    def _wait_for_dns_propagation(self, change_id, max_wait_time=600):
        """Wait for DNS changes to propagate"""
        logger.info("⏳ Waiting for DNS changes to propagate...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                response = self.route53_client.get_change(Id=change_id)
                status = response['ChangeInfo']['Status']
                
                if status == 'INSYNC':
                    logger.info("✅ DNS changes have propagated successfully")
                    return True
                
                logger.info(f"   Status: {status} - waiting...")
                time.sleep(30)
                
            except ClientError as e:
                logger.error(f"❌ Error checking DNS propagation: {str(e)}")
                return False
        
        logger.warning(f"⚠️ DNS propagation check timed out after {max_wait_time} seconds")
        return False
    
    def create_health_check(self, domain_name, path="/", port=80):
        """Create Route 53 health check for the website"""
        try:
            response = self.route53_client.create_health_check(
                Type='HTTP',
                ResourcePath=path,
                FullyQualifiedDomainName=domain_name,
                Port=port,
                RequestInterval=30,
                FailureThreshold=3,
                Tags=[
                    {
                        'Key': 'Name',
                        'Value': f'Health Check for {domain_name}'
                    },
                    {
                        'Key': 'Project',
                        'Value': 'XYZ Corporation S3 Case Study'
                    }
                ]
            )
            
            health_check_id = response['HealthCheck']['Id']
            
            logger.info(f"✅ Health check created successfully")
            logger.info(f"   Health Check ID: {health_check_id}")
            logger.info(f"   Domain: {domain_name}")
            logger.info(f"   Path: {path}")
            
            return health_check_id
            
        except ClientError as e:
            logger.error(f"❌ Failed to create health check: {str(e)}")
            return None
    
    def test_dns_resolution(self, domain_name):
        """Test DNS resolution for the domain"""
        logger.info(f"🧪 Testing DNS resolution for {domain_name}")
        
        try:
            # Test A record
            ip_addresses = socket.gethostbyname_ex(domain_name)[2]
            logger.info(f"✅ A record resolved successfully")
            for ip in ip_addresses:
                logger.info(f"   IP: {ip}")
            
            # Test www subdomain
            try:
                www_ips = socket.gethostbyname_ex(f"www.{domain_name}")[2]
                logger.info(f"✅ WWW record resolved successfully")
                for ip in www_ips:
                    logger.info(f"   WWW IP: {ip}")
            except socket.gaierror:
                logger.warning("⚠️ WWW subdomain not configured or not propagated yet")
            
            return True
            
        except socket.gaierror as e:
            logger.error(f"❌ DNS resolution failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ DNS test error: {str(e)}")
            return False
    
    def generate_dns_report(self, hosted_zone_info, domain_name):
        """Generate comprehensive DNS configuration report"""
        try:
            # Get all records in the hosted zone
            response = self.route53_client.list_resource_record_sets(
                HostedZoneId=hosted_zone_info['hosted_zone_id']
            )
            
            records = response['ResourceRecordSets']
            
            report = {
                'dns_configuration': {
                    'domain': domain_name,
                    'hosted_zone_id': hosted_zone_info['hosted_zone_id'],
                    'name_servers': hosted_zone_info['name_servers'],
                    'generation_time': datetime.now().isoformat()
                },
                'dns_records': [],
                'summary': {
                    'total_records': len(records),
                    'record_types': {}
                }
            }
            
            # Process records
            for record in records:
                record_info = {
                    'name': record['Name'],
                    'type': record['Type'],
                    'ttl': record.get('TTL', 'N/A')
                }
                
                # Count record types
                record_type = record['Type']
                if record_type not in report['summary']['record_types']:
                    report['summary']['record_types'][record_type] = 0
                report['summary']['record_types'][record_type] += 1
                
                # Get record values
                if 'ResourceRecords' in record:
                    record_info['values'] = [rr['Value'] for rr in record['ResourceRecords']]
                elif 'AliasTarget' in record:
                    record_info['alias_target'] = {
                        'dns_name': record['AliasTarget']['DNSName'],
                        'hosted_zone_id': record['AliasTarget']['HostedZoneId']
                    }
                
                report['dns_records'].append(record_info)
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"dns_configuration_report_{domain_name.replace('.', '_')}_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 DNS configuration report saved to: {report_file}")
            
            # Display summary
            logger.info("📊 DNS Configuration Summary:")
            logger.info(f"   Domain: {domain_name}")
            logger.info(f"   Total Records: {report['summary']['total_records']}")
            logger.info("   Record Types:")
            for record_type, count in report['summary']['record_types'].items():
                logger.info(f"     {record_type}: {count}")
            
            return report_file
            
        except ClientError as e:
            logger.error(f"❌ Failed to generate DNS report: {str(e)}")
            return None

def main():
    """Main execution function"""
    logger.info("🚀 Starting Route 53 DNS Configuration Automation")
    
    # Initialize DNS manager
    dns_manager = Route53DNSManager()
    
    # Configuration - Replace with your actual values
    domain_name = "example.com"  # Replace with your domain
    s3_bucket_name = "example.com"  # Should match your domain
    
    try:
        # Step 1: Create or get hosted zone
        logger.info(f"🌐 Setting up hosted zone for {domain_name}")
        hosted_zone_info = dns_manager.create_hosted_zone(domain_name)
        
        if not hosted_zone_info:
            logger.error("❌ Failed to create or get hosted zone")
            return
        
        # Step 2: Display name server information
        logger.info("📋 IMPORTANT: Update your domain's name servers with your registrar:")
        for i, ns in enumerate(hosted_zone_info['name_servers'], 1):
            logger.info(f"   Name Server {i}: {ns}")
        
        # Step 3: Create DNS records for S3 website
        logger.info("📝 Creating DNS records for S3 website hosting...")
        if dns_manager.create_s3_website_records(
            hosted_zone_info['hosted_zone_id'], 
            domain_name, 
            s3_bucket_name,
            create_www_redirect=True
        ):
            logger.info("✅ DNS records created successfully")
        else:
            logger.error("❌ Failed to create DNS records")
            return
        
        # Step 4: Create health check
        logger.info("🩺 Creating health check...")
        health_check_id = dns_manager.create_health_check(domain_name)
        
        # Step 5: Test DNS resolution
        logger.info("🧪 Testing DNS resolution...")
        dns_manager.test_dns_resolution(domain_name)
        
        # Step 6: Generate DNS report
        logger.info("📊 Generating DNS configuration report...")
        report_file = dns_manager.generate_dns_report(hosted_zone_info, domain_name)
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎉 ROUTE 53 DNS CONFIGURATION COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"🌐 Domain: {domain_name}")
        logger.info(f"🆔 Hosted Zone ID: {hosted_zone_info['hosted_zone_id']}")
        if health_check_id:
            logger.info(f"🩺 Health Check ID: {health_check_id}")
        logger.info(f"📄 Configuration Report: {report_file}")
        logger.info("")
        logger.info("📋 NEXT STEPS:")
        logger.info("1. Update your domain registrar with the name servers listed above")
        logger.info("2. Wait 24-48 hours for DNS propagation")
        logger.info("3. Test your website using your custom domain")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ DNS configuration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()