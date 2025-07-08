#!/bin/bash

# Setup Kind cluster for ML Platform
# This script creates a local Kubernetes cluster using Kind

set -e

# Configuration
CLUSTER_NAME="ml-platform-cluster"
KUBERNETES_VERSION="v1.27.3"
NODE_COUNT=2

# Use different ports to avoid conflicts
HTTP_PORT=8080
HTTPS_PORT=8443

echo "🚀 Setting up Kind cluster for ML Platform..."

# Check if Kind is installed
if ! command -v kind &> /dev/null; then
    echo "❌ Kind is not installed. Please install Kind first:"
    echo "   https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check for port conflicts
echo "🔍 Checking for port conflicts..."
if lsof -Pi :${HTTP_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port ${HTTP_PORT} is already in use. Using alternative port 8081"
    HTTP_PORT=8081
fi

if lsof -Pi :${HTTPS_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port ${HTTPS_PORT} is already in use. Using alternative port 8444"
    HTTPS_PORT=8444
fi

echo "📋 Using ports: HTTP=${HTTP_PORT}, HTTPS=${HTTPS_PORT}"

# Create Kind cluster configuration
cat << EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: ${CLUSTER_NAME}
nodes:
- role: control-plane
  image: kindest/node:${KUBERNETES_VERSION}
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: ${HTTP_PORT}
    protocol: TCP
  - containerPort: 443
    hostPort: ${HTTPS_PORT}
    protocol: TCP
- role: worker
  image: kindest/node:${KUBERNETES_VERSION}
- role: worker
  image: kindest/node:${KUBERNETES_VERSION}
EOF

# Delete existing cluster if it exists
if kind get clusters | grep -q "${CLUSTER_NAME}"; then
    echo "🗑️  Deleting existing cluster..."
    kind delete cluster --name "${CLUSTER_NAME}"
fi

# Create new cluster
echo "🏗️  Creating Kind cluster..."
if ! kind create cluster --config kind-config.yaml; then
    echo "❌ Failed to create Kind cluster. This might be due to:"
    echo "   1. Port conflicts (try stopping other services using ports 80/443)"
    echo "   2. Docker not having enough resources"
    echo "   3. Docker daemon issues"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "   1. Stop any web servers: sudo systemctl stop apache2 nginx"
    echo "   2. Check Docker resources: docker system df"
    echo "   3. Restart Docker: sudo systemctl restart docker"
    echo "   4. Try again: ./scripts/setup-kind.sh"
    exit 1
fi

# Wait for cluster to be ready
echo "⏳ Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

# Install NGINX Ingress Controller
echo "📦 Installing NGINX Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# Wait for ingress controller to be ready
echo "⏳ Waiting for Ingress Controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s

# Install cert-manager
echo "📦 Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Wait for cert-manager to be ready
echo "⏳ Waiting for cert-manager to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/cert-manager-webhook -n cert-manager

# Create namespace for ML platform
echo "📁 Creating namespace..."
kubectl create namespace ml-platform --dry-run=client -o yaml | kubectl apply -f -

# Set default namespace
kubectl config set-context --current --namespace=ml-platform

# Install monitoring stack (optional)
if [ "$1" = "--with-monitoring" ]; then
    echo "📊 Installing monitoring stack..."
    
    # Install Prometheus Operator
    kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/setup/0-namespace.yaml
    kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/setup/
    kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/kube-prometheus/main/manifests/
    
    echo "⏳ Waiting for monitoring stack to be ready..."
    kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s
fi

# Clean up config file
rm -f kind-config.yaml

# Display cluster information
echo ""
echo "✅ Kind cluster setup complete!"
echo ""
echo "📋 Cluster Information:"
echo "   Name: ${CLUSTER_NAME}"
echo "   Kubernetes Version: ${KUBERNETES_VERSION}"
echo "   Nodes: ${NODE_COUNT}"
echo ""
echo "🔧 Useful Commands:"
echo "   kubectl cluster-info"
echo "   kubectl get nodes"
echo "   kubectl get pods -A"
echo "   kubectl get ingress -A"
echo ""
echo "🌐 Access Points:"
echo "   Kubernetes API: https://localhost:6443"
echo "   Ingress Controller: http://localhost:${HTTP_PORT}"
echo ""
echo "📚 Next Steps:"
echo "   1. Build and push your Docker image:"
echo "      docker build -t ml-platform-inference ."
echo "      docker tag ml-platform-inference:latest your-registry/ml-platform-inference:latest"
echo "      docker push your-registry/ml-platform-inference:latest"
echo ""
echo "   2. Deploy the application:"
echo "      kubectl apply -f infrastructure/kubernetes/"
echo ""
echo "   3. Check deployment status:"
echo "      kubectl get pods"
echo "      kubectl get services"
echo "      kubectl get ingress"
echo ""

# Set up port forwarding for local development
echo "🔗 Setting up port forwarding for local development..."
echo "   You can access the application at: http://localhost:8080"
echo "   (Run this in a separate terminal: kubectl port-forward service/ml-inference-service 8080:80)"
echo ""

# Export cluster info for other scripts
export KIND_CLUSTER_NAME="${CLUSTER_NAME}"
export KUBECONFIG="$(kind get kubeconfig-path --name="${CLUSTER_NAME}")"

echo "🎉 Setup complete! Your Kind cluster is ready for development." 