# ML Platform Setup Guide

This guide will help you set up the complete ML platform with CI/CD workflow using Snyk security scanning, Docker Hub, and Kind (Kubernetes in Docker).

## Prerequisites

### Required Tools

1. **Docker**
   ```bash
   # Install Docker Desktop or Docker Engine
   # https://docs.docker.com/get-docker/
   ```

2. **Kind (Kubernetes in Docker)**
   ```bash
   # macOS/Linux
   curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
   chmod +x ./kind
   sudo mv ./kind /usr/local/bin/kind
   
   # Windows
   # Download from: https://kind.sigs.k8s.io/dl/v0.20.0/kind-windows-amd64
   ```

3. **kubectl**
   ```bash
   # macOS
   brew install kubectl
   
   # Linux
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   chmod +x kubectl
   sudo mv kubectl /usr/local/bin/
   
   # Windows
   # Download from: https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/
   ```

4. **Python 3.8+**
   ```bash
   # Install Python 3.8 or later
   # https://www.python.org/downloads/
   ```

5. **Git**
   ```bash
   # Install Git
   # https://git-scm.com/downloads
   ```

### Required Accounts

1. **GitHub Account**
   - Create a GitHub account if you don't have one
   - Fork or clone this repository

2. **Docker Hub Account**
   - Create a Docker Hub account
   - Create a repository for your ML platform

3. **Snyk Account** (Optional but recommended)
   - Sign up for a free Snyk account
   - Get your Snyk API token

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd AI-ML-Model-Deployment-Platform
```

### 2. Set Up GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions, and add the following secrets:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token
- `SNYK_TOKEN`: Your Snyk API token (optional)

### 3. Local Development Setup

#### Option A: Using Setup Scripts (Recommended)

```bash
# Make scripts executable
chmod +x scripts/setup-kind.sh
chmod +x scripts/deploy-local.sh

# Setup Kind cluster
./scripts/setup-kind.sh

# Deploy application locally
./scripts/deploy-local.sh
```

#### Option B: Manual Setup

```bash
# 1. Create Kind cluster
kind create cluster --name ml-platform-cluster

# 2. Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# 3. Create namespace
kubectl create namespace ml-platform

# 4. Build and deploy
docker build -t ml-platform-inference .
kubectl apply -f infrastructure/kubernetes/
```

### 4. Configure Docker Hub

1. **Create Docker Hub Repository**
   - Go to Docker Hub and create a new repository
   - Note the repository name (e.g., `yourusername/ml-platform`)

2. **Update GitHub Workflow**
   - Edit `.github/workflows/ci-cd.yml`
   - Update the `IMAGE_NAME` variable with your repository name

### 5. Configure Snyk (Optional)

1. **Get Snyk Token**
   ```bash
   # Install Snyk CLI
   npm install -g snyk
   
   # Login to Snyk
   snyk auth
   ```

2. **Add Snyk Token to GitHub Secrets**
   - Copy your Snyk token from the CLI or Snyk dashboard
   - Add it as `SNYK_TOKEN` in GitHub secrets

## CI/CD Workflow Overview

The CI/CD pipeline consists of the following stages:

### 1. Security Scan
- **Snyk**: Vulnerability scanning for dependencies and code
- **Bandit**: Python security linting
- **Safety**: Dependency vulnerability checking
- **Semgrep**: Static analysis
- **Trivy**: Container vulnerability scanning

### 2. Testing
- Unit tests with pytest
- Code coverage reporting
- Integration tests

### 3. Build and Push
- Docker image building
- Push to Docker Hub
- Container vulnerability scanning

### 4. Deployment
- Kind cluster setup
- Kubernetes deployment
- Health checks and verification

## Usage

### Local Development

```bash
# Start the application locally
python -m uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000

# Or use Docker
docker build -t ml-platform-inference .
docker run -p 8000:8000 ml-platform-inference
```

### API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `GET /model/info`: Model information
- `POST /predict`: Make predictions
- `POST /predict/image`: Predict on uploaded image
- `POST /model/reload`: Reload model

### Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html
```

### Security Scanning

```bash
# Run security scan locally
python security/scanning/security_scanner.py

# Run with specific options
python security/scanning/security_scanner.py --directory src/ --output security-report.json --html
```

## Monitoring and Observability

### Application Monitoring
- Health checks at `/health`
- Model performance metrics
- Request/response logging

### Kubernetes Monitoring
```bash
# Check pod status
kubectl get pods -n ml-platform

# View logs
kubectl logs -f deployment/ml-model-inference -n ml-platform

# Check resource usage
kubectl top pods -n ml-platform
```

### Security Monitoring
- Snyk vulnerability reports
- Container vulnerability scanning
- Dependency vulnerability alerts

## Troubleshooting

### Common Issues

1. **Kind Cluster Issues**
   ```bash
   # Delete and recreate cluster
   kind delete cluster --name ml-platform-cluster
   ./scripts/setup-kind.sh
   ```

2. **Docker Build Issues**
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker build --no-cache -t ml-platform-inference .
   ```

3. **Kubernetes Deployment Issues**
   ```bash
   # Check pod events
   kubectl describe pod -l app=ml-model-inference -n ml-platform
   
   # Check logs
   kubectl logs deployment/ml-model-inference -n ml-platform
   ```

4. **Security Scan Issues**
   ```bash
   # Check Snyk token
   snyk auth
   
   # Run security scan manually
   python security/scanning/security_scanner.py --debug
   ```

### Performance Optimization

1. **Docker Build Optimization**
   - Use multi-stage builds
   - Optimize layer caching
   - Use .dockerignore effectively

2. **Kubernetes Optimization**
   - Set appropriate resource limits
   - Use horizontal pod autoscaling
   - Optimize image pull policies

3. **Security Optimization**
   - Regular dependency updates
   - Automated vulnerability scanning
   - Security policy enforcement

## Next Steps

1. **Customize the Platform**
   - Modify model architecture
   - Add new API endpoints
   - Implement custom monitoring

2. **Scale the Deployment**
   - Deploy to production Kubernetes cluster
   - Set up monitoring and alerting
   - Implement blue-green deployments

3. **Enhance Security**
   - Implement RBAC
   - Add network policies
   - Set up secrets management

4. **Add Features**
   - Model versioning
   - A/B testing
   - Automated retraining

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review GitHub issues
3. Create a new issue with detailed information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and security scans
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 