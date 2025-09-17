#!/usr/bin/env python3
"""
AWS Permission Boundary Scanner
Scans all active AWS accounts to find roles with 'syf-Sandbox-permission-boundary' attached.
Uses multithreading for optimal performance across 217+ accounts.
"""

import boto3
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich import box

console = Console()

@dataclass
class RoleInfo:
    account_id: str
    account_name: str
    role_name: str
    has_permission_boundary: str

class AWSAccountScanner:
    def __init__(self, target_permission_boundary: str = 'syf-Sandbox-permission-boundary', 
                 role_to_assume: str = 'ca-iam-cie-engineer'):
        self.target_permission_boundary = target_permission_boundary
        self.role_to_assume = role_to_assume
        self.results = []
        self.lock = threading.Lock()
        self.session = boto3.Session()
        self.org_client = self.session.client('organizations')
        
    def get_all_active_accounts(self) -> List[Dict]:
        """Retrieve all active AWS accounts from Organizations"""
        try:
            console.print("[bold blue]📋 Fetching all active AWS accounts...[/bold blue]")
            
            accounts = []
            paginator = self.org_client.get_paginator('list_accounts')
            
            for page in paginator.paginate():
                for account in page['Accounts']:
                    if account['Status'] == 'ACTIVE':
                        accounts.append({
                            'Id': account['Id'],
                            'Name': account['Name'],
                            'Email': account['Email']
                        })
            
            console.print(f"[green]✅ Found {len(accounts)} active accounts[/green]")
            return accounts
            
        except Exception as e:
            console.print(f"[red]❌ Error fetching accounts: {str(e)}[/red]")
            return []

    def assume_role_in_account(self, account_id: str) -> Optional[boto3.Session]:
        """Assume the specified role in target account"""
        try:
            sts_client = self.session.client('sts')
            role_arn = f"arn:aws:iam::{account_id}:role/{self.role_to_assume}"
            
            response = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName=f"permission-boundary-scan-{int(time.time())}"
            )
            
            credentials = response['Credentials']
            assumed_session = boto3.Session(
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken']
            )
            
            return assumed_session
            
        except Exception as e:
            # Skip accounts where we can't assume role (expected for some accounts)
            return None

    def scan_account_roles(self, account: Dict) -> List[RoleInfo]:
        """Scan all IAM roles in a specific account for the target permission boundary"""
        account_id = account['Id']
        account_name = account['Name']
        account_results = []
        
        try:
            # Assume role in target account
            assumed_session = self.assume_role_in_account(account_id)
            if not assumed_session:
                # Add entry showing account couldn't be accessed
                account_results.append(RoleInfo(
                    account_id=account_id,
                    account_name=account_name,
                    role_name="ACCESS_DENIED",
                    has_permission_boundary="N/A"
                ))
                return account_results
            
            iam_client = assumed_session.client('iam')
            
            # Get all roles in the account
            paginator = iam_client.get_paginator('list_roles')
            roles_with_boundary = []
            total_roles = 0
            
            for page in paginator.paginate():
                for role in page['Roles']:
                    total_roles += 1
                    role_name = role['RoleName']
                    
                    # Check if role has any permission boundary
                    if 'PermissionsBoundary' in role:
                        boundary_arn = role['PermissionsBoundary']['PermissionsBoundaryArn']
                        boundary_name = boundary_arn.split('/')[-1]
                        
                        if boundary_name == self.target_permission_boundary:
                            roles_with_boundary.append(RoleInfo(
                                account_id=account_id,
                                account_name=account_name,
                                role_name=role_name,
                                has_permission_boundary="Exists"
                            ))
            
            # If no roles found with the boundary, add a summary entry
            if not roles_with_boundary:
                account_results.append(RoleInfo(
                    account_id=account_id,
                    account_name=account_name,
                    role_name=f"NO_ROLES_WITH_BOUNDARY ({total_roles} total roles)",
                    has_permission_boundary="Missing"
                ))
            else:
                account_results.extend(roles_with_boundary)
                
        except Exception as e:
            account_results.append(RoleInfo(
                account_id=account_id,
                account_name=account_name,
                role_name=f"ERROR: {str(e)[:50]}...",
                has_permission_boundary="Error"
            ))
        
        return account_results

    def worker(self, account: Dict, progress: Progress, task_id: int) -> List[RoleInfo]:
        """Worker function for threading"""
        results = self.scan_account_roles(account)
        
        with self.lock:
            self.results.extend(results)
            progress.update(task_id, advance=1)
        
        return results

    def create_results_table(self) -> Table:
        """Create a rich table for displaying results"""
        table = Table(
            title=f"🔍 AWS Permission Boundary Scan Results - '{self.target_permission_boundary}'",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Account ID", style="cyan", no_wrap=True, width=12)
        table.add_column("Account Name", style="blue", width=25)
        table.add_column("Role Name", style="white", width=40)
        table.add_column("Permission Boundary", style="green", justify="center", width=15)
        
        # Sort results by account ID for better readability
        sorted_results = sorted(self.results, key=lambda x: (x.account_id, x.role_name))
        
        for result in sorted_results:
            status_style = "green" if result.has_permission_boundary == "Exists" else "red"
            if result.has_permission_boundary in ["N/A", "Error"]:
                status_style = "yellow"
            
            table.add_row(
                result.account_id,
                result.account_name[:24] + "..." if len(result.account_name) > 24 else result.account_name,
                result.role_name[:39] + "..." if len(result.role_name) > 39 else result.role_name,
                f"[{status_style}]{result.has_permission_boundary}[/{status_style}]"
            )
        
        return table

    def save_to_csv(self, filename: str = None):
        """Save results to CSV file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"permission_boundary_scan_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['AccountID', 'AccountName', 'RoleName', 'syf-Sandbox-permission-boundary']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                sorted_results = sorted(self.results, key=lambda x: (x.account_id, x.role_name))
                
                for result in sorted_results:
                    writer.writerow({
                        'AccountID': result.account_id,
                        'AccountName': result.account_name,
                        'RoleName': result.role_name,
                        'syf-Sandbox-permission-boundary': result.has_permission_boundary
                    })
            
            console.print(f"[green]✅ Results saved to {filename}[/green]")
            return filename
            
        except Exception as e:
            console.print(f"[red]❌ Error saving CSV: {str(e)}[/red]")
            return None

    def scan_all_accounts(self, max_workers: int = 50):
        """Main scanning function with multithreading"""
        start_time = time.time()
        
        # Get all active accounts
        accounts = self.get_all_active_accounts()
        if not accounts:
            console.print("[red]❌ No accounts found or error occurred[/red]")
            return
        
        console.print(f"[bold green]🚀 Starting scan of {len(accounts)} accounts using {max_workers} threads...[/bold green]")
        console.print(f"[bold yellow]🎯 Target Permission Boundary: {self.target_permission_boundary}[/bold yellow]")
        console.print(f"[bold yellow]👤 Assuming Role: {self.role_to_assume}[/bold yellow]")
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Scanning accounts...", total=len(accounts))
            
            # Execute with thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.worker, account, progress, task): account 
                    for account in accounts
                }
                
                # Wait for completion
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        account = futures[future]
                        console.print(f"[red]❌ Error processing account {account['Id']}: {str(e)}[/red]")
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display results
        console.print("\n")
        console.print(self.create_results_table())
        
        # Summary statistics
        total_results = len(self.results)
        roles_with_boundary = len([r for r in self.results if r.has_permission_boundary == "Exists"])
        accessible_accounts = len([r for r in self.results if r.has_permission_boundary != "N/A"])
        
        console.print(f"\n[bold green]📊 Scan Summary:[/bold green]")
        console.print(f"⏱️  Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        console.print(f"🏢 Total Accounts: {len(accounts)}")
        console.print(f"🔑 Accessible Accounts: {accessible_accounts}")
        console.print(f"📋 Total Results: {total_results}")
        console.print(f"✅ Roles with Target Boundary: {roles_with_boundary}")
        
        # Save to CSV
        csv_filename = self.save_to_csv()
        
        return self.results

def main():
    """Main execution function"""
    console.print("[bold blue]🔍 AWS Permission Boundary Scanner[/bold blue]")
    console.print("=" * 60)
    
    # Initialize scanner
    scanner = AWSAccountScanner(
        target_permission_boundary='syf-Sandbox-permission-boundary',
        role_to_assume='ca-iam-cie-engineer'
    )
    
    # Run the scan with optimized thread count for 217+ accounts
    # Using 50 threads to balance speed and API rate limits
    results = scanner.scan_all_accounts(max_workers=50)
    
    if results:
        console.print("\n[bold green]🎉 Scan completed successfully![/bold green]")
    else:
        console.print("\n[bold red]❌ Scan failed or no results found[/bold red]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Scan interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Unexpected error: {str(e)}[/red]")