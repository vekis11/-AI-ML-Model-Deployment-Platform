#!/bin/bash

# Deploy ML Platform to local Kind cluster
# This script builds and deploys the application to the local Kind cluster

set -e

# Configuration
IMAGE_NAME="ml-platform-inference"
TAG="latest"
REGISTRY="localhost:5000"  # Local registry for Kind
NAMESPACE="ml-platform"

echo "🚀 Deploying ML Platform to local Kind cluster..."

# Check if Kind cluster exists
if ! kind get clusters | grep -q "ml-platform-cluster"; then
    echo "❌ Kind cluster not found. Please run setup-kind.sh first."
    echo ""
    echo "🔧 To fix this:"
    echo "   1. Run: ./scripts/setup-kind.sh"
    echo "   2. If you get port conflicts, run: ./scripts/fix-port-conflicts.sh"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Set kubectl context
echo "🔧 Setting kubectl context..."
kubectl config use-context kind-ml-platform-cluster
kubectl config set-context --current --namespace=${NAMESPACE}

# Create local registry if it doesn't exist
if ! docker ps | grep -q "registry:2"; then
    echo "📦 Creating local Docker registry..."
    docker run -d --name registry -p 5000:5000 --restart=always registry:2
    
    # Connect registry to Kind network
    docker network connect kind registry 2>/dev/null || true
fi

# Build Docker image
echo "🏗️  Building Docker image..."
if ! docker build -t ${IMAGE_NAME}:${TAG} .; then
    echo "❌ Docker build failed. Please check your Dockerfile and try again."
    exit 1
fi

# Tag for local registry
docker tag ${IMAGE_NAME}:${TAG} ${REGISTRY}/${IMAGE_NAME}:${TAG}

# Push to local registry
echo "📤 Pushing image to local registry..."
if ! docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}; then
    echo "❌ Failed to push to local registry. This might be due to:"
    echo "   1. Registry not running"
    echo "   2. Network connectivity issues"
    echo "   3. Docker daemon problems"
    echo ""
    echo "🔧 Try restarting the registry:"
    echo "   docker stop registry && docker rm registry"
    echo "   docker run -d --name registry -p 5000:5000 --restart=always registry:2"
    exit 1
fi

# Update deployment manifest with local image
echo "📝 Updating deployment manifest..."
sed -i.bak "s|mlplatformacr.azurecr.io/ml-model-inference:latest|${REGISTRY}/${IMAGE_NAME}:${TAG}|g" infrastructure/kubernetes/deployment.yaml

# Apply Kubernetes manifests
echo "📦 Deploying to Kubernetes..."
kubectl apply -f infrastructure/kubernetes/deployment.yaml
kubectl apply -f infrastructure/kubernetes/service.yaml
kubectl apply -f infrastructure/kubernetes/ingress.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
if ! kubectl wait --for=condition=available --timeout=300s deployment/ml-model-inference; then
    echo "❌ Deployment failed to become ready. Checking pod status..."
    kubectl get pods
    kubectl describe pod -l app=ml-model-inference
    echo ""
    echo "🔧 Common issues:"
    echo "   1. Image pull issues: Check if registry is accessible"
    echo "   2. Resource constraints: Check Docker resources"
    echo "   3. Configuration issues: Check deployment manifest"
    exit 1
fi

# Check deployment status
echo "📊 Deployment status:"
kubectl get pods
kubectl get services
kubectl get ingress

# Set up port forwarding
echo "🔗 Setting up port forwarding..."
echo "   Application will be available at: http://localhost:8080"
echo "   Health check: http://localhost:8080/health"
echo "   Model info: http://localhost:8080/model/info"
echo ""

# Start port forwarding in background
kubectl port-forward service/ml-inference-service 8080:80 &
PF_PID=$!

# Function to cleanup port forwarding on exit
cleanup() {
    echo "🧹 Cleaning up..."
    kill $PF_PID 2>/dev/null || true
    # Restore original deployment file
    mv infrastructure/kubernetes/deployment.yaml.bak infrastructure/kubernetes/deployment.yaml 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait a moment for port forwarding to start
sleep 5

# Test the application
echo "🧪 Testing application..."
for i in {1..10}; do
    if curl -f http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Application is running successfully!"
        echo ""
        echo "🎉 Deployment complete!"
        echo ""
        echo "📋 Access Information:"
        echo "   Health Check: http://localhost:8080/health"
        echo "   API Root: http://localhost:8080/"
        echo "   Model Info: http://localhost:8080/model/info"
        echo "   Predict: http://localhost:8080/predict"
        echo ""
        echo "🔧 Useful Commands:"
        echo "   kubectl logs -f deployment/ml-model-inference"
        echo "   kubectl describe pod -l app=ml-model-inference"
        echo "   kubectl get events --sort-by='.lastTimestamp'"
        echo ""
        echo "🛑 Press Ctrl+C to stop port forwarding and cleanup"
        break
    else
        if [ $i -eq 10 ]; then
            echo "❌ Application health check failed after 10 attempts"
            echo "   Check logs: kubectl logs deployment/ml-model-inference"
            cleanup
        else
            echo "⏳ Waiting for application to start... (attempt $i/10)"
            sleep 2
        fi
    fi
done

# Keep the script running to maintain port forwarding
wait $PF_PID 