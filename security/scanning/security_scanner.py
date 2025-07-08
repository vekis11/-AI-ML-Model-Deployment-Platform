"""
Comprehensive security scanner for the ML platform.
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml


class SecurityScanner:
    """Comprehensive security scanner for ML platform."""
    
    def __init__(self, scan_directory: str = "."):
        """Initialize the security scanner."""
        self.scan_directory = scan_directory
        self.logger = logging.getLogger(__name__)
        self.results = {}
    
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scan on Python code."""
        try:
            self.logger.info("Running Bandit security scan...")
            
            cmd = [
                "bandit", "-r", self.scan_directory,
                "-f", "json",
                "-o", "bandit-report.json",
                "--exclude", "tests/,venv/,env/,__pycache__/"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                with open("bandit-report.json", "r") as f:
                    bandit_results = json.load(f)
                
                # Process results
                issues = []
                for issue in bandit_results.get("results", []):
                    issues.append({
                        "tool": "Bandit",
                        "severity": issue.get("issue_severity", "UNKNOWN"),
                        "confidence": issue.get("issue_confidence", "UNKNOWN"),
                        "message": issue.get("issue_text", ""),
                        "file": issue.get("filename", ""),
                        "line": issue.get("line_number", ""),
                        "test_id": issue.get("test_id", ""),
                        "more_info": issue.get("more_info", "")
                    })
                
                return {
                    "tool": "Bandit",
                    "status": "completed",
                    "issues": issues,
                    "summary": {
                        "total_issues": len(issues),
                        "high_issues": len([i for i in issues if i["severity"] == "HIGH"]),
                        "medium_issues": len([i for i in issues if i["severity"] == "MEDIUM"]),
                        "low_issues": len([i for i in issues if i["severity"] == "LOW"])
                    }
                }
            else:
                return {
                    "tool": "Bandit",
                    "status": "failed",
                    "error": result.stderr,
                    "issues": []
                }
                
        except Exception as e:
            self.logger.error(f"Bandit scan failed: {e}")
            return {
                "tool": "Bandit",
                "status": "error",
                "error": str(e),
                "issues": []
            }
    
    def run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety scan for vulnerable dependencies."""
        try:
            self.logger.info("Running Safety dependency scan...")
            
            cmd = ["safety", "check", "--json", "--output", "safety-report.json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                with open("safety-report.json", "r") as f:
                    safety_results = json.load(f)
                
                # Process results
                issues = []
                for issue in safety_results:
                    issues.append({
                        "tool": "Safety",
                        "severity": "HIGH",  # Safety only reports vulnerabilities
                        "package": issue.get("package", ""),
                        "installed_version": issue.get("installed_version", ""),
                        "vulnerable_spec": issue.get("vulnerable_spec", ""),
                        "advisory": issue.get("advisory", ""),
                        "message": f"Vulnerable package: {issue.get('package', '')}",
                        "file": "requirements.txt"
                    })
                
                return {
                    "tool": "Safety",
                    "status": "completed",
                    "issues": issues,
                    "summary": {
                        "total_issues": len(issues),
                        "high_issues": len(issues),
                        "medium_issues": 0,
                        "low_issues": 0
                    }
                }
            else:
                return {
                    "tool": "Safety",
                    "status": "failed",
                    "error": result.stderr,
                    "issues": []
                }
                
        except Exception as e:
            self.logger.error(f"Safety scan failed: {e}")
            return {
                "tool": "Safety",
                "status": "error",
                "error": str(e),
                "issues": []
            }
    
    def run_semgrep_scan(self) -> Dict[str, Any]:
        """Run Semgrep static analysis."""
        try:
            self.logger.info("Running Semgrep static analysis...")
            
            cmd = [
                "semgrep", "scan",
                "--config=auto",
                "--json",
                "--output=semgrep-report.json",
                self.scan_directory
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode in [0, 1]:  # Semgrep returns 1 when issues are found
                with open("semgrep-report.json", "r") as f:
                    semgrep_results = json.load(f)
                
                # Process results
                issues = []
                for result_item in semgrep_results.get("results", []):
                    issues.append({
                        "tool": "Semgrep",
                        "severity": result_item.get("extra", {}).get("severity", "UNKNOWN"),
                        "message": result_item.get("extra", {}).get("message", ""),
                        "file": result_item.get("path", ""),
                        "line": result_item.get("start", {}).get("line", ""),
                        "rule_id": result_item.get("check_id", ""),
                        "more_info": result_item.get("extra", {}).get("metadata", {})
                    })
                
                return {
                    "tool": "Semgrep",
                    "status": "completed",
                    "issues": issues,
                    "summary": {
                        "total_issues": len(issues),
                        "high_issues": len([i for i in issues if i["severity"] == "ERROR"]),
                        "medium_issues": len([i for i in issues if i["severity"] == "WARNING"]),
                        "low_issues": len([i for i in issues if i["severity"] == "INFO"])
                    }
                }
            else:
                return {
                    "tool": "Semgrep",
                    "status": "failed",
                    "error": result.stderr,
                    "issues": []
                }
                
        except Exception as e:
            self.logger.error(f"Semgrep scan failed: {e}")
            return {
                "tool": "Semgrep",
                "status": "error",
                "error": str(e),
                "issues": []
            }
    
    def run_trivy_scan(self, image_name: Optional[str] = None) -> Dict[str, Any]:
        """Run Trivy container vulnerability scan."""
        try:
            self.logger.info("Running Trivy container scan...")
            
            if not image_name:
                # Look for Dockerfile and build image
                if os.path.exists("Dockerfile"):
                    image_name = "ml-platform:latest"
                    build_cmd = ["docker", "build", "-t", image_name, "."]
                    subprocess.run(build_cmd, check=True)
                else:
                    return {
                        "tool": "Trivy",
                        "status": "skipped",
                        "message": "No Dockerfile found",
                        "issues": []
                    }
            
            cmd = [
                "trivy", "image",
                "--format", "json",
                "--output", "trivy-report.json",
                image_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode in [0, 1]:  # Trivy returns 1 when vulnerabilities are found
                with open("trivy-report.json", "r") as f:
                    trivy_results = json.load(f)
                
                # Process results
                issues = []
                for vuln in trivy_results.get("Results", []):
                    for vulnerability in vuln.get("Vulnerabilities", []):
                        issues.append({
                            "tool": "Trivy",
                            "severity": vulnerability.get("Severity", "UNKNOWN"),
                            "package": vulnerability.get("PkgName", ""),
                            "installed_version": vulnerability.get("InstalledVersion", ""),
                            "fixed_version": vulnerability.get("FixedVersion", ""),
                            "message": vulnerability.get("Title", ""),
                            "description": vulnerability.get("Description", ""),
                            "cve_id": vulnerability.get("VulnerabilityID", "")
                        })
                
                return {
                    "tool": "Trivy",
                    "status": "completed",
                    "issues": issues,
                    "summary": {
                        "total_issues": len(issues),
                        "high_issues": len([i for i in issues if i["severity"] == "HIGH"]),
                        "medium_issues": len([i for i in issues if i["severity"] == "MEDIUM"]),
                        "low_issues": len([i for i in issues if i["severity"] == "LOW"])
                    }
                }
            else:
                return {
                    "tool": "Trivy",
                    "status": "failed",
                    "error": result.stderr,
                    "issues": []
                }
                
        except Exception as e:
            self.logger.error(f"Trivy scan failed: {e}")
            return {
                "tool": "Trivy",
                "status": "error",
                "error": str(e),
                "issues": []
            }
    
    def check_secrets(self) -> Dict[str, Any]:
        """Check for hardcoded secrets in code."""
        try:
            self.logger.info("Checking for hardcoded secrets...")
            
            # Common secret patterns
            secret_patterns = [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]",
                r"private_key\s*=\s*['\"][^'\"]+['\"]",
                r"connection_string\s*=\s*['\"][^'\"]+['\"]"
            ]
            
            issues = []
            
            for root, dirs, files in os.walk(self.scan_directory):
                # Skip common directories
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'env', 'node_modules']]
                
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.yaml', '.yml', '.json', '.env')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                for pattern in secret_patterns:
                                    import re
                                    matches = re.finditer(pattern, content, re.IGNORECASE)
                                    for match in matches:
                                        issues.append({
                                            "tool": "Secret Scanner",
                                            "severity": "HIGH",
                                            "message": f"Potential hardcoded secret found: {match.group()}",
                                            "file": file_path,
                                            "line": content[:match.start()].count('\n') + 1,
                                            "pattern": pattern
                                        })
                        except Exception as e:
                            self.logger.warning(f"Could not read file {file_path}: {e}")
            
            return {
                "tool": "Secret Scanner",
                "status": "completed",
                "issues": issues,
                "summary": {
                    "total_issues": len(issues),
                    "high_issues": len(issues),
                    "medium_issues": 0,
                    "low_issues": 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Secret scan failed: {e}")
            return {
                "tool": "Secret Scanner",
                "status": "error",
                "error": str(e),
                "issues": []
            }
    
    def run_comprehensive_scan(self, include_container: bool = True) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        self.logger.info("Starting comprehensive security scan...")
        
        scan_results = []
        
        # Run all scans
        scan_results.append(self.run_bandit_scan())
        scan_results.append(self.run_safety_scan())
        scan_results.append(self.run_semgrep_scan())
        scan_results.append(self.check_secrets())
        
        if include_container:
            scan_results.append(self.run_trivy_scan())
        
        # Aggregate results
        all_issues = []
        total_high = 0
        total_medium = 0
        total_low = 0
        
        for result in scan_results:
            if result["status"] == "completed":
                all_issues.extend(result["issues"])
                summary = result.get("summary", {})
                total_high += summary.get("high_issues", 0)
                total_medium += summary.get("medium_issues", 0)
                total_low += summary.get("low_issues", 0)
        
        # Generate comprehensive report
        comprehensive_report = {
            "scan_timestamp": datetime.now().isoformat(),
            "scan_directory": self.scan_directory,
            "tools_used": [r["tool"] for r in scan_results],
            "overall_summary": {
                "total_issues": len(all_issues),
                "high_issues": total_high,
                "medium_issues": total_medium,
                "low_issues": total_low,
                "risk_level": self._calculate_risk_level(total_high, total_medium, total_low)
            },
            "tool_results": scan_results,
            "all_issues": all_issues
        }
        
        # Save report
        with open("security-scan-report.json", "w") as f:
            json.dump(comprehensive_report, f, indent=2)
        
        self.logger.info(f"Security scan completed. Found {len(all_issues)} total issues.")
        self.logger.info(f"Risk level: {comprehensive_report['overall_summary']['risk_level']}")
        
        return comprehensive_report
    
    def _calculate_risk_level(self, high: int, medium: int, low: int) -> str:
        """Calculate overall risk level."""
        if high > 0:
            return "CRITICAL"
        elif medium > 5:
            return "HIGH"
        elif medium > 0 or low > 10:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_html_report(self, report_data: Dict[str, Any], output_file: str = "security-report.html"):
        """Generate HTML security report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .risk-critical { color: #d32f2f; font-weight: bold; }
        .risk-high { color: #f57c00; font-weight: bold; }
        .risk-medium { color: #fbc02d; font-weight: bold; }
        .risk-low { color: #388e3c; font-weight: bold; }
        .issue { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
        .issue.high { border-left-color: #d32f2f; background-color: #ffebee; }
        .issue.medium { border-left-color: #f57c00; background-color: #fff3e0; }
        .issue.low { border-left-color: #fbc02d; background-color: #fffde7; }
        .tool-section { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Scan Report</h1>
        <p>Generated on: {timestamp}</p>
        <p>Scan Directory: {directory}</p>
    </div>
    
    <div class="summary">
        <h2>Overall Summary</h2>
        <p>Risk Level: <span class="risk-{risk_level.lower()}">{risk_level}</span></p>
        <p>Total Issues: {total_issues}</p>
        <p>High Priority: {high_issues}</p>
        <p>Medium Priority: {medium_issues}</p>
        <p>Low Priority: {low_issues}</p>
    </div>
    
    <div class="tool-section">
        <h2>Tool Results</h2>
        {tool_results}
    </div>
    
    <div class="tool-section">
        <h2>All Issues</h2>
        {all_issues}
    </div>
</body>
</html>
"""
        
        # Generate tool results HTML
        tool_results_html = ""
        for tool_result in report_data.get("tool_results", []):
            tool_results_html += f"""
            <h3>{tool_result['tool']}</h3>
            <p>Status: {tool_result['status']}</p>
            <p>Issues Found: {len(tool_result.get('issues', []))}</p>
            """
        
        # Generate all issues HTML
        all_issues_html = ""
        for issue in report_data.get("all_issues", []):
            severity_class = issue.get("severity", "UNKNOWN").lower()
            all_issues_html += f"""
            <div class="issue {severity_class}">
                <strong>{issue.get('tool', 'Unknown')} - {issue.get('severity', 'UNKNOWN')}</strong><br>
                <strong>File:</strong> {issue.get('file', 'N/A')}<br>
                <strong>Line:</strong> {issue.get('line', 'N/A')}<br>
                <strong>Message:</strong> {issue.get('message', 'N/A')}
            </div>
            """
        
        # Fill template
        html_content = html_template.format(
            timestamp=report_data.get("scan_timestamp", ""),
            directory=report_data.get("scan_directory", ""),
            risk_level=report_data.get("overall_summary", {}).get("risk_level", "UNKNOWN"),
            total_issues=report_data.get("overall_summary", {}).get("total_issues", 0),
            high_issues=report_data.get("overall_summary", {}).get("high_issues", 0),
            medium_issues=report_data.get("overall_summary", {}).get("medium_issues", 0),
            low_issues=report_data.get("overall_summary", {}).get("low_issues", 0),
            tool_results=tool_results_html,
            all_issues=all_issues_html
        )
        
        with open(output_file, "w") as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_file}")


def main():
    """Main function to run security scan."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive security scan")
    parser.add_argument("--directory", default=".", help="Directory to scan")
    parser.add_argument("--output", default="security-report.json", help="Output file")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--no-container", action="store_true", help="Skip container scan")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run scan
    scanner = SecurityScanner(args.directory)
    report = scanner.run_comprehensive_scan(include_container=not args.no_container)
    
    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report if requested
    if args.html:
        scanner.generate_html_report(report)
    
    print(f"Security scan completed. Report saved to {args.output}")
    print(f"Risk level: {report['overall_summary']['risk_level']}")
    print(f"Total issues: {report['overall_summary']['total_issues']}")


if __name__ == "__main__":
    main() 