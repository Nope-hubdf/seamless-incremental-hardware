#!/usr/bin/env python3
"""
AWS S3 Website Deployment Automation
Author: Himanshu Nitin Nehete
Case Study: XYZ Corporation S3 Storage Infrastructure
Institution: iHub Divyasampark, IIT Roorkee

This script automates S3 static website deployment:
- Upload website files with proper MIME types
- Configure bucket for website hosting
- Set up custom error pages
- Configure CORS and caching headers
- Invalidate CloudFront (if applicable)
"""

import boto3
import os
import mimetypes
import json
import logging
import sys
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('s3-website-deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class S3WebsiteDeployer:
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
    
    def create_sample_website_files(self, local_dir="website-files"):
        """Create sample website files for deployment"""
        os.makedirs(local_dir, exist_ok=True)
        os.makedirs(f"{local_dir}/assets/css", exist_ok=True)
        os.makedirs(f"{local_dir}/assets/js", exist_ok=True)
        os.makedirs(f"{local_dir}/assets/images", exist_ok=True)
        
        # Create index.html
        index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XYZ Corporation - Cloud Storage Solution</title>
    <link rel="stylesheet" href="assets/css/style.css">
    <link rel="icon" type="image/x-icon" href="assets/images/favicon.ico">
</head>
<body>
    <header>
        <nav class="navbar">
            <div class="nav-container">
                <h1 class="logo">XYZ Corporation</h1>
                <ul class="nav-menu">
                    <li><a href="#home">Home</a></li>
                    <li><a href="#services">Services</a></li>
                    <li><a href="#about">About</a></li>
                    <li><a href="#contact">Contact</a></li>
                </ul>
            </div>
        </nav>
    </header>

    <main>
        <section id="home" class="hero">
            <div class="hero-content">
                <h1>Welcome to XYZ Corporation</h1>
                <p>Professional Cloud Storage & Website Hosting Solutions</p>
                <div class="hero-stats">
                    <div class="stat">
                        <h3>Unlimited</h3>
                        <p>Cloud Storage</p>
                    </div>
                    <div class="stat">
                        <h3>99.9%</h3>
                        <p>Availability</p>
                    </div>
                    <div class="stat">
                        <h3>Global</h3>
                        <p>Accessibility</p>
                    </div>
                </div>
            </div>
        </section>

        <section id="services" class="services">
            <div class="container">
                <h2>Our Services</h2>
                <div class="service-grid">
                    <div class="service-card">
                        <h3>🗄️ Cloud Storage</h3>
                        <p>Unlimited scalable storage with automated lifecycle management</p>
                    </div>
                    <div class="service-card">
                        <h3>🌐 Website Hosting</h3>
                        <p>Static website hosting with global content delivery</p>
                    </div>
                    <div class="service-card">
                        <h3>🔐 Data Security</h3>
                        <p>Enterprise-grade security with encryption and versioning</p>
                    </div>
                    <div class="service-card">
                        <h3>💰 Cost Optimization</h3>
                        <p>Pay-only-for-what-you-use pricing with intelligent tiering</p>
                    </div>
                </div>
            </div>
        </section>

        <section id="about" class="about">
            <div class="container">
                <h2>About This Project</h2>
                <p>This website demonstrates AWS S3 static hosting capabilities as part of our comprehensive case study on cloud storage solutions.</p>
                <ul>
                    <li>✅ Unlimited Cloud Storage Implementation</li>
                    <li>✅ Automated Lifecycle Management</li>
                    <li>✅ Version Control & Recovery</li>
                    <li>✅ Static Website Hosting</li>
                    <li>✅ Custom Domain Integration</li>
                </ul>
            </div>
        </section>
    </main>

    <footer>
        <div class="container">
            <p>&copy; 2025 XYZ Corporation. Part of AWS S3 Case Study - iHub Divyasampark, IIT Roorkee</p>
            <p>Developed by: Himanshu Nitin Nehete</p>
        </div>
    </footer>

    <script src="assets/js/script.js"></script>
</body>
</html>"""
        
        with open(f"{local_dir}/index.html", "w") as f:
            f.write(index_html)
        
        # Create error.html
        error_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Page Not Found | XYZ Corporation</title>
    <link rel="stylesheet" href="assets/css/style.css">
</head>
<body>
    <div class="error-container">
        <div class="error-content">
            <h1 class="error-code">404</h1>
            <h2 class="error-message">Oops! Page Not Found</h2>
            <p class="error-description">
                The page you're looking for seems to have vanished into the cloud. 
                Don't worry, our cloud storage is much more reliable than this error!
            </p>
            <div class="error-actions">
                <a href="/" class="btn-primary">Return Home</a>
                <a href="mailto:himanshunehete2025@gmail.com" class="btn-secondary">Report Issue</a>
            </div>
            <div class="error-info">
                <p><strong>Common Causes:</strong></p>
                <ul>
                    <li>The page URL might be misspelled</li>
                    <li>The page may have been moved or deleted</li>
                    <li>You might not have permission to access this resource</li>
                </ul>
            </div>
        </div>
        <div class="cloud-animation">
            <div class="cloud"></div>
            <div class="cloud"></div>
            <div class="cloud"></div>
        </div>
    </div>
</body>
</html>"""
        
        with open(f"{local_dir}/error.html", "w") as f:
            f.write(error_html)
        
        # Create CSS file
        css_content = """/* XYZ Corporation Website Styles */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

/* Navigation */
.navbar {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 1rem 0;
    position: fixed;
    top: 0;
    width: 100%;
    z-index: 1000;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
}

.nav-container {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 20px;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: #667eea;
}

.nav-menu {
    display: flex;
    list-style: none;
    gap: 2rem;
}

.nav-menu a {
    text-decoration: none;
    color: #333;
    font-weight: 500;
    transition: color 0.3s ease;
}

.nav-menu a:hover {
    color: #667eea;
}

/* Hero Section */
.hero {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: white;
    padding-top: 80px;
}

.hero-content h1 {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.hero-content p {
    font-size: 1.3rem;
    margin-bottom: 3rem;
    opacity: 0.9;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 4rem;
    flex-wrap: wrap;
}

.stat {
    text-align: center;
}

.stat h3 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    color: #ffd700;
}

.stat p {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Services Section */
.services {
    padding: 5rem 0;
    background: white;
}

.services h2 {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 3rem;
    color: #333;
}

.service-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
}

.service-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.service-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
}

.service-card h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: #667eea;
}

/* About Section */
.about {
    padding: 5rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.about h2 {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 2rem;
}

.about p {
    font-size: 1.2rem;
    text-align: center;
    margin-bottom: 2rem;
    opacity: 0.9;
}

.about ul {
    max-width: 600px;
    margin: 0 auto;
    list-style: none;
}

.about li {
    padding: 0.8rem 0;
    font-size: 1.1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

/* Footer */
footer {
    background: #333;
    color: white;
    text-align: center;
    padding: 2rem 0;
}

footer p {
    margin: 0.5rem 0;
    opacity: 0.8;
}

/* Error Page Styles */
.error-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    position: relative;
    overflow: hidden;
}

.error-content {
    text-align: center;
    max-width: 600px;
    padding: 2rem;
    z-index: 2;
}

.error-code {
    font-size: 8rem;
    font-weight: bold;
    margin-bottom: 1rem;
    text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
    color: #ffd700;
}

.error-message {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.error-description {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    opacity: 0.9;
    line-height: 1.8;
}

.error-actions {
    margin: 2rem 0;
}

.btn-primary, .btn-secondary {
    display: inline-block;
    padding: 12px 24px;
    margin: 0 10px;
    text-decoration: none;
    border-radius: 8px;
    transition: all 0.3s ease;
    font-weight: 500;
}

.btn-primary {
    background: #ffd700;
    color: #333;
}

.btn-primary:hover {
    background: #ffed4a;
    transform: translateY(-2px);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.2);
    color: white;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

.error-info {
    margin-top: 3rem;
    text-align: left;
    background: rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    border-radius: 10px;
}

.error-info ul {
    margin-top: 1rem;
    padding-left: 1.5rem;
}

.error-info li {
    margin: 0.5rem 0;
    opacity: 0.9;
}

/* Cloud Animation */
.cloud-animation {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    z-index: 1;
}

.cloud {
    position: absolute;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50px;
    opacity: 0.6;
    animation: float 6s ease-in-out infinite;
}

.cloud:nth-child(1) {
    width: 100px;
    height: 40px;
    top: 20%;
    left: 10%;
    animation-delay: 0s;
}

.cloud:nth-child(2) {
    width: 150px;
    height: 60px;
    top: 50%;
    right: 10%;
    animation-delay: 2s;
}

.cloud:nth-child(3) {
    width: 80px;
    height: 30px;
    bottom: 20%;
    left: 20%;
    animation-delay: 4s;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-content h1 {
        font-size: 2.5rem;
    }
    
    .hero-stats {
        gap: 2rem;
    }
    
    .stat h3 {
        font-size: 2rem;
    }
    
    .service-grid {
        grid-template-columns: 1fr;
    }
    
    .error-code {
        font-size: 6rem;
    }
    
    .error-message {
        font-size: 2rem;
    }
    
    .nav-menu {
        display: none; /* Simplified mobile navigation */
    }
}"""
        
        with open(f"{local_dir}/assets/css/style.css", "w") as f:
            f.write(css_content)
        
        # Create JavaScript file
        js_content = """// XYZ Corporation Website JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add scroll effect to navbar
    let lastScrollTop = 0;
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > lastScrollTop && scrollTop > 100) {
            // Scrolling down
            navbar.style.transform = 'translateY(-100%)';
        } else {
            // Scrolling up
            navbar.style.transform = 'translateY(0)';
        }
        
        lastScrollTop = scrollTop;
    });
    
    // Add loading animation to service cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe service cards
    const serviceCards = document.querySelectorAll('.service-card');
    serviceCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
    
    // Console log for demonstration
    console.log('🌐 XYZ Corporation Website Loaded Successfully!');
    console.log('📊 This is a demonstration of AWS S3 static website hosting');
    console.log('🎓 Part of iHub Divyasampark, IIT Roorkee Case Study');
    console.log('👨‍💻 Developed by: Himanshu Nitin Nehete');
    
    // Add click counter for demonstration
    let clickCounter = 0;
    document.addEventListener('click', function() {
        clickCounter++;
        if (clickCounter % 10 === 0) {
            console.log(`🖱️ User has made ${clickCounter} clicks on the website`);
        }
    });
});

// Error page specific JavaScript
if (window.location.pathname.includes('error') || document.title.includes('404')) {
    console.log('❌ 404 Error page loaded');
    
    // Add some interactive elements to error page
    const errorCode = document.querySelector('.error-code');
    if (errorCode) {
        errorCode.addEventListener('click', function() {
            this.style.transform = 'scale(1.1)';
            this.style.color = '#ff6b6b';
            
            setTimeout(() => {
                this.style.transform = 'scale(1)';
                this.style.color = '#ffd700';
            }, 200);
        });
    }
}

// Performance monitoring
window.addEventListener('load', function() {
    const loadTime = performance.now();
    console.log(`⚡ Website loaded in ${Math.round(loadTime)} milliseconds`);
    
    // Report Core Web Vitals
    if ('web-vital' in window) {
        console.log('📈 Core Web Vitals monitoring enabled');
    }
});"""
        
        with open(f"{local_dir}/assets/js/script.js", "w") as f:
            f.write(js_content)
        
        logger.info(f"✅ Sample website files created in {local_dir}/")
        return local_dir
    
    def upload_files_to_s3(self, bucket_name, local_dir, prefix=""):
        """Upload website files to S3 with proper MIME types"""
        uploaded_files = []
        
        try:
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, local_dir)
                    s3_key = os.path.join(prefix, relative_path).replace('\\', '/')
                    
                    # Determine MIME type
                    mime_type, _ = mimetypes.guess_type(local_path)
                    if mime_type is None:
                        mime_type = 'binary/octet-stream'
                    
                    # Set cache control for different file types
                    cache_control = 'public, max-age=31536000'  # 1 year for assets
                    if file.endswith(('.html', '.htm')):
                        cache_control = 'public, max-age=3600'  # 1 hour for HTML
                    elif file.endswith(('.css', '.js')):
                        cache_control = 'public, max-age=86400'  # 1 day for CSS/JS
                    
                    # Upload file
                    extra_args = {
                        'ContentType': mime_type,
                        'CacheControl': cache_control,
                        'ACL': 'public-read'
                    }
                    
                    self.s3_client.upload_file(
                        local_path, 
                        bucket_name, 
                        s3_key,
                        ExtraArgs=extra_args
                    )
                    
                    uploaded_files.append({
                        'local_path': local_path,
                        's3_key': s3_key,
                        'mime_type': mime_type,
                        'size': os.path.getsize(local_path)
                    })
                    
                    logger.info(f"📤 Uploaded: {s3_key} ({mime_type})")
            
            logger.info(f"✅ Uploaded {len(uploaded_files)} files to {bucket_name}")
            return uploaded_files
            
        except ClientError as e:
            logger.error(f"❌ Failed to upload files: {str(e)}")
            return []
    
    def configure_website_hosting(self, bucket_name, index_document='index.html', error_document='error.html'):
        """Configure S3 bucket for static website hosting"""
        try:
            website_config = {
                'IndexDocument': {'Suffix': index_document},
                'ErrorDocument': {'Key': error_document}
            }
            
            self.s3_client.put_bucket_website(
                Bucket=bucket_name,
                WebsiteConfiguration=website_config
            )
            
            logger.info(f"✅ Website hosting configured for {bucket_name}")
            logger.info(f"   Index Document: {index_document}")
            logger.info(f"   Error Document: {error_document}")
            
            # Get website endpoint
            website_url = f"http://{bucket_name}.s3-website.{self.region}.amazonaws.com/"
            logger.info(f"🌐 Website URL: {website_url}")
            
            return website_url
            
        except ClientError as e:
            logger.error(f"❌ Failed to configure website hosting: {str(e)}")
            return None
    
    def set_bucket_policy_for_website(self, bucket_name):
        """Set bucket policy to allow public read access for website"""
        bucket_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicReadGetObject",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{bucket_name}/*"
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            logger.info(f"✅ Public read policy applied to {bucket_name}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ Failed to set bucket policy: {str(e)}")
            return False
    
    def configure_cors(self, bucket_name):
        """Configure CORS for the website bucket"""
        cors_config = {
            'CORSRules': [
                {
                    'AllowedHeaders': ['*'],
                    'AllowedMethods': ['GET', 'HEAD'],
                    'AllowedOrigins': ['*'],
                    'ExposeHeaders': ['ETag'],
                    'MaxAgeSeconds': 3000
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_cors(
                Bucket=bucket_name,
                CORSConfiguration=cors_config
            )
            
            logger.info(f"✅ CORS configuration applied to {bucket_name}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ Failed to configure CORS: {str(e)}")
            return False
    
    def test_website_accessibility(self, website_url):
        """Test if the website is accessible"""
        import urllib.request
        import urllib.error
        
        try:
            with urllib.request.urlopen(website_url, timeout=10) as response:
                status_code = response.getcode()
                content_length = len(response.read())
                
            if status_code == 200:
                logger.info(f"✅ Website is accessible: {website_url}")
                logger.info(f"   Status Code: {status_code}")
                logger.info(f"   Content Length: {content_length} bytes")
                return True
            else:
                logger.warning(f"⚠️ Website returned status code: {status_code}")
                return False
                
        except urllib.error.URLError as e:
            logger.error(f"❌ Website accessibility test failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during website test: {str(e)}")
            return False
    
    def generate_deployment_report(self, bucket_name, uploaded_files, website_url):
        """Generate comprehensive deployment report"""
        from datetime import datetime
        
        # Analyze file types
        file_types = {}
        for file_info in uploaded_files:
            ext = os.path.splitext(file_info['s3_key'])[1].lower()
            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size': 0}
            file_types[ext]['count'] += 1
            file_types[ext]['size'] += file_info['size']
        
        report = {
            'deployment_info': {
                'bucket_name': bucket_name,
                'website_url': website_url,
                'deployment_time': datetime.now().isoformat(),
                'region': self.region
            },
            'file_summary': {
                'total_files': len(uploaded_files),
                'total_size_bytes': sum(f['size'] for f in uploaded_files),
                'total_size_mb': round(sum(f['size'] for f in uploaded_files) / (1024*1024), 2),
                'file_types': file_types
            },
            'uploaded_files': uploaded_files,
            'configuration': {
                'website_hosting': 'Enabled',
                'public_read_access': 'Enabled',
                'cors': 'Configured',
                'index_document': 'index.html',
                'error_document': 'error.html'
            }
        }
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"deployment_report_{bucket_name}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Deployment report saved to: {report_file}")
        return report_file

def main():
    """Main execution function"""
    logger.info("🚀 Starting S3 Website Deployment Automation")
    
    # Initialize website deployer
    deployer = S3WebsiteDeployer()
    
    # Configuration
    bucket_name = "xyz-corp-website-example"  # Replace with your bucket name
    local_website_dir = "website-files"
    
    try:
        # Step 1: Create sample website files
        logger.info("📝 Creating sample website files...")
        deployer.create_sample_website_files(local_website_dir)
        
        # Step 2: Check if bucket exists, create if not
        logger.info(f"🪣 Checking bucket: {bucket_name}")
        try:
            deployer.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"✅ Bucket {bucket_name} exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"🆕 Creating bucket: {bucket_name}")
                deployer.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': deployer.region}
                )
        
        # Step 3: Upload website files
        logger.info("📤 Uploading website files...")
        uploaded_files = deployer.upload_files_to_s3(bucket_name, local_website_dir)
        
        if not uploaded_files:
            logger.error("❌ No files uploaded, aborting deployment")
            return
        
        # Step 4: Configure website hosting
        logger.info("🌐 Configuring website hosting...")
        website_url = deployer.configure_website_hosting(bucket_name)
        
        if not website_url:
            logger.error("❌ Failed to configure website hosting")
            return
        
        # Step 5: Set bucket policy for public access
        logger.info("🔓 Setting public read policy...")
        deployer.set_bucket_policy_for_website(bucket_name)
        
        # Step 6: Configure CORS
        logger.info("🔗 Configuring CORS...")
        deployer.configure_cors(bucket_name)
        
        # Step 7: Test website accessibility
        logger.info("🧪 Testing website accessibility...")
        deployer.test_website_accessibility(website_url)
        
        # Step 8: Generate deployment report
        logger.info("📊 Generating deployment report...")
        report_file = deployer.generate_deployment_report(bucket_name, uploaded_files, website_url)
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎉 WEBSITE DEPLOYMENT COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"🪣 Bucket: {bucket_name}")
        logger.info(f"🌐 Website URL: {website_url}")
        logger.info(f"📁 Files Uploaded: {len(uploaded_files)}")
        logger.info(f"💾 Total Size: {sum(f['size'] for f in uploaded_files) / (1024*1024):.2f} MB")
        logger.info(f"📄 Report: {report_file}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()