# Fix port conflicts for Kind cluster setup (Windows PowerShell)
# This script helps identify and resolve port conflicts

Write-Host "🔍 Checking for port conflicts..." -ForegroundColor Yellow

# Check common ports that might conflict
$PORTS_TO_CHECK = @(80, 443, 8080, 8443)

foreach ($port in $PORTS_TO_CHECK) {
    $processes = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq "Listen" }
    
    if ($processes) {
        Write-Host "⚠️  Port $port is in use by:" -ForegroundColor Red
        foreach ($process in $processes) {
            $processInfo = Get-Process -Id $process.OwningProcess -ErrorAction SilentlyContinue
            if ($processInfo) {
                Write-Host "   Process: $($processInfo.ProcessName) (PID: $($processInfo.Id))" -ForegroundColor Red
            }
        }
        Write-Host ""
    } else {
        Write-Host "✅ Port $port is available" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🔧 Common services that use port 80/443:" -ForegroundColor Yellow
Write-Host "   - IIS: Stop-Service W3SVC"
Write-Host "   - Apache: Stop-Service Apache2.4"
Write-Host "   - Nginx: Stop-Service nginx"
Write-Host "   - Docker containers: docker ps (check for web servers)"
Write-Host "   - Other web servers: Get-NetTCPConnection -LocalPort 80"
Write-Host ""

# Check for common web servers
Write-Host "🔍 Checking for common web servers..." -ForegroundColor Yellow

# Check IIS
$iisService = Get-Service -Name "W3SVC" -ErrorAction SilentlyContinue
if ($iisService -and $iisService.Status -eq "Running") {
    Write-Host "⚠️  IIS is running. Stop it with: Stop-Service W3SVC" -ForegroundColor Red
}

# Check Apache
$apacheService = Get-Service -Name "Apache2.4" -ErrorAction SilentlyContinue
if ($apacheService -and $apacheService.Status -eq "Running") {
    Write-Host "⚠️  Apache is running. Stop it with: Stop-Service Apache2.4" -ForegroundColor Red
}

# Check Nginx
$nginxService = Get-Service -Name "nginx" -ErrorAction SilentlyContinue
if ($nginxService -and $nginxService.Status -eq "Running") {
    Write-Host "⚠️  Nginx is running. Stop it with: Stop-Service nginx" -ForegroundColor Red
}

# Check Docker containers
try {
    $dockerContainers = docker ps --format "table {{.Names}}\t{{.Ports}}" 2>$null
    if ($dockerContainers -and $dockerContainers -match ":80->") {
        Write-Host "⚠️  Docker containers are using port 80:" -ForegroundColor Red
        $dockerContainers | Where-Object { $_ -match ":80->" } | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
    }
} catch {
    Write-Host "ℹ️  Docker not available or not running" -ForegroundColor Gray
}

Write-Host ""
Write-Host "💡 Quick fixes:" -ForegroundColor Yellow
Write-Host "   1. Stop IIS: Stop-Service W3SVC"
Write-Host "   2. Stop Apache: Stop-Service Apache2.4"
Write-Host "   3. Stop Nginx: Stop-Service nginx"
Write-Host "   4. Stop Docker containers: docker stop <container_name>"
Write-Host "   5. Use the updated setup script: ./scripts/setup-kind.sh"
Write-Host ""
Write-Host "🎯 The updated setup script will automatically use alternative ports if needed." -ForegroundColor Green 