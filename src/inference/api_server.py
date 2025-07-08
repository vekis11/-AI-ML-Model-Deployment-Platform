"""
FastAPI inference server for model serving.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from opencensus.ext.azure.log_exporter import AzureLogHandler

from .predictor import ModelPredictor
from ..training.model_config import ModelConfig


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    image_urls: Optional[List[str]] = None
    data: Optional[List[List[float]]] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]


class InferenceServer:
    """FastAPI inference server."""
    
    def __init__(self, model_path: str, config: ModelConfig = None, 
                 host: str = "0.0.0.0", port: int = 8000):
        """Initialize the inference server."""
        self.model_path = model_path
        self.config = config
        self.host = host
        self.port = port
        self.predictor = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="ML Model Inference API",
            description="REST API for ML model inference",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup Azure logging if available
        self._setup_logging()
        
        # Setup routes
        self._setup_routes()
        
        # Load model
        self._load_model()
    
    def _setup_logging(self):
        """Setup logging with Azure Application Insights if available."""
        try:
            connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
            if connection_string:
                self.logger.addHandler(AzureLogHandler(connection_string=connection_string))
                self.logger.info("Azure Application Insights logging enabled")
        except Exception as e:
            self.logger.warning(f"Failed to setup Azure logging: {e}")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {"message": "ML Model Inference API", "version": "1.0.0"}
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            model_loaded = self.predictor is not None
            model_info = self.predictor.get_model_info() if model_loaded else {}
            
            return HealthResponse(
                status="healthy" if model_loaded else "unhealthy",
                model_loaded=model_loaded,
                model_info=model_info
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Predict endpoint for data."""
            if self.predictor is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = time.time()
            
            try:
                if request.data:
                    # Convert to numpy array
                    input_data = np.array(request.data)
                    predictions = self.predictor.predict_with_confidence(input_data)
                elif request.image_urls:
                    # Predict on image URLs
                    predictions = self.predictor.predict_images_batch(request.image_urls)
                else:
                    raise HTTPException(status_code=400, detail="No data provided")
                
                processing_time = time.time() - start_time
                
                return PredictionResponse(
                    predictions=predictions if isinstance(predictions, list) else [predictions],
                    model_info=self.predictor.get_model_info(),
                    processing_time=processing_time
                )
            
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/image")
        async def predict_image(file: UploadFile = File(...)):
            """Predict endpoint for image upload."""
            if self.predictor is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = time.time()
            
            try:
                # Save uploaded file temporarily
                temp_path = f"/tmp/{file.filename}"
                with open(temp_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Make prediction
                prediction = self.predictor.predict_image(temp_path)
                
                # Clean up
                os.remove(temp_path)
                
                processing_time = time.time() - start_time
                
                return {
                    "prediction": prediction,
                    "processing_time": processing_time,
                    "filename": file.filename
                }
            
            except Exception as e:
                self.logger.error(f"Image prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/model/info")
        async def get_model_info():
            """Get model information."""
            if self.predictor is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            return self.predictor.get_model_info()
        
        @self.app.post("/model/reload")
        async def reload_model(background_tasks: BackgroundTasks):
            """Reload model in background."""
            background_tasks.add_task(self._load_model)
            return {"message": "Model reload started"}
    
    def _load_model(self):
        """Load the model."""
        try:
            self.predictor = ModelPredictor(self.model_path, self.config)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.predictor = None
    
    def run(self, debug: bool = False):
        """Run the inference server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            log_level="info"
        )
    
    def get_app(self):
        """Get the FastAPI app instance."""
        return self.app


def create_inference_server(model_path: str, config: ModelConfig = None,
                          host: str = "0.0.0.0", port: int = 8000) -> InferenceServer:
    """Factory function to create inference server."""
    return InferenceServer(model_path, config, host, port)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Start ML inference server")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and run server
    server = create_inference_server(args.model_path, host=args.host, port=args.port)
    server.run(debug=args.debug) 