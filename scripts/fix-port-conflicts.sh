#!/bin/bash

# Fix port conflicts for Kind cluster setup
# This script helps identify and resolve port conflicts

echo "🔍 Checking for port conflicts..."

# Check common ports that might conflict
PORTS_TO_CHECK=(80 443 8080 8443)

for port in "${PORTS_TO_CHECK[@]}"; do
    if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port ${port} is in use by:"
        lsof -Pi :${port} -sTCP:LISTEN
        echo ""
    else
        echo "✅ Port ${port} is available"
    fi
done

echo ""
echo "🔧 Common services that use port 80/443:"
echo "   - Apache2: sudo systemctl stop apache2"
echo "   - Nginx: sudo systemctl stop nginx"
echo "   - Docker containers: docker ps (check for web servers)"
echo "   - Other web servers: check with 'sudo netstat -tlnp | grep :80'"
echo ""

# Check for common web servers
echo "🔍 Checking for common web servers..."

if systemctl is-active --quiet apache2; then
    echo "⚠️  Apache2 is running. Stop it with: sudo systemctl stop apache2"
fi

if systemctl is-active --quiet nginx; then
    echo "⚠️  Nginx is running. Stop it with: sudo systemctl stop nginx"
fi

if docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -q ":80->"; then
    echo "⚠️  Docker containers are using port 80:"
    docker ps --format "table {{.Names}}\t{{.Ports}}" | grep ":80->"
fi

echo ""
echo "💡 Quick fixes:"
echo "   1. Stop Apache2: sudo systemctl stop apache2"
echo "   2. Stop Nginx: sudo systemctl stop nginx"
echo "   3. Stop Docker containers: docker stop <container_name>"
echo "   4. Use the updated setup script: ./scripts/setup-kind.sh"
echo ""
echo "🎯 The updated setup script will automatically use alternative ports if needed." 