"""
Integration tests for the ML inference API.
"""

import pytest
import requests
import time
import os


class TestAPIIntegration:
    """Integration tests for the ML inference API."""
    
    @pytest.fixture(scope="class")
    def base_url(self):
        """Get the base URL for the API."""
        return os.getenv("SERVICE_URL", "http://localhost:8000")
    
    def test_health_endpoint(self, base_url):
        """Test the health check endpoint."""
        response = requests.get(f"{base_url}/health", timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "model_info" in data
        
        # Check if model is loaded (may be false in some test environments)
        assert isinstance(data["model_loaded"], bool)
    
    def test_root_endpoint(self, base_url):
        """Test the root endpoint."""
        response = requests.get(f"{base_url}/", timeout=10)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert data["message"] == "ML Model Inference API"
    
    def test_model_info_endpoint(self, base_url):
        """Test the model info endpoint."""
        response = requests.get(f"{base_url}/model/info", timeout=10)
        
        # This might return 503 if model is not loaded, which is acceptable
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check for expected fields
        assert "model_path" in data or "error" in data
    
    def test_predict_endpoint_without_data(self, base_url):
        """Test predict endpoint without providing data."""
        response = requests.post(f"{base_url}/predict", json={}, timeout=10)
        
        # Should return 400 for missing data
        assert response.status_code == 400
    
    def test_predict_endpoint_with_invalid_data(self, base_url):
        """Test predict endpoint with invalid data."""
        invalid_data = {
            "data": [[1, 2, 3]]  # Invalid shape
        }
        
        response = requests.post(f"{base_url}/predict", json=invalid_data, timeout=10)
        
        # Should return 500 for invalid data
        assert response.status_code in [400, 500]
    
    def test_api_response_time(self, base_url):
        """Test API response time."""
        start_time = time.time()
        response = requests.get(f"{base_url}/health", timeout=10)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_api_headers(self, base_url):
        """Test API response headers."""
        response = requests.get(f"{base_url}/health", timeout=10)
        
        assert response.status_code == 200
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]
    
    def test_cors_headers(self, base_url):
        """Test CORS headers."""
        response = requests.options(f"{base_url}/health", timeout=10)
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
    
    @pytest.mark.skipif(
        not os.path.exists("test_data/sample_image.jpg"),
        reason="Test image not available"
    )
    def test_image_prediction_endpoint(self, base_url):
        """Test image prediction endpoint."""
        with open("test_data/sample_image.jpg", "rb") as f:
            files = {"file": f}
            response = requests.post(f"{base_url}/predict/image", files=files, timeout=30)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded in test environment")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "prediction" in data
        assert "processing_time" in data
        assert "filename" in data
    
    def test_model_reload_endpoint(self, base_url):
        """Test model reload endpoint."""
        response = requests.post(f"{base_url}/model/reload", timeout=30)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "Model reload started" in data["message"]
    
    def test_concurrent_requests(self, base_url):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return requests.get(f"{base_url}/health", timeout=10)
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_error_handling(self, base_url):
        """Test error handling for invalid endpoints."""
        response = requests.get(f"{base_url}/invalid-endpoint", timeout=10)
        
        assert response.status_code == 404
    
    def test_large_payload_handling(self, base_url):
        """Test handling of large payloads."""
        # Create a large payload
        large_data = {
            "data": [[1.0] * 1000] * 100  # 1000 features, 100 samples
        }
        
        response = requests.post(f"{base_url}/predict", json=large_data, timeout=30)
        
        # Should handle large payloads gracefully
        assert response.status_code in [200, 400, 500]  # Accept various responses


class TestAPIPerformance:
    """Performance tests for the API."""
    
    @pytest.fixture(scope="class")
    def base_url(self):
        """Get the base URL for the API."""
        return os.getenv("SERVICE_URL", "http://localhost:8000")
    
    def test_health_endpoint_performance(self, base_url):
        """Test health endpoint performance."""
        response_times = []
        
        for _ in range(10):
            start_time = time.time()
            response = requests.get(f"{base_url}/health", timeout=10)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Performance assertions
        assert avg_response_time < 1.0  # Average response time < 1 second
        assert max_response_time < 3.0  # Max response time < 3 seconds
    
    def test_throughput(self, base_url):
        """Test API throughput."""
        import concurrent.futures
        
        def make_request():
            return requests.get(f"{base_url}/health", timeout=10)
        
        start_time = time.time()
        
        # Make 50 requests concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        total_time = end_time - start_time
        requests_per_second = 50 / total_time
        
        # All requests should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        
        assert success_count >= 45  # At least 90% success rate
        assert requests_per_second > 5  # At least 5 requests per second 