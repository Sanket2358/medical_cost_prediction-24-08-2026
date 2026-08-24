import pickle
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PredictorAPI")

# --- 2. MODEL LIFESPAN MANAGEMENT ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model once when the server starts
    logger.info("Starting up server and loading XGBoost model...")
    try:
        with open("xg_model.pkl", "rb") as f:
            ml_models["xg_model"] = pickle.load(f)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error("xg_model.pkl not found! Please ensure it is in the root directory.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    yield # Server is running
    
    # Shutdown: Clean up resources
    logger.info("Shutting down server...")
    ml_models.clear()

# --- 3. APP INITIALIZATION ---
app = FastAPI(
    title="Advanced XGBoost Predictor",
    description="Production-ready API for ML predictions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for security and future frontend separation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. PYDANTIC DATA VALIDATION ---
class PredictionRequest(BaseModel):
    Age: int = Field(..., gt=0, lt=120, description="Age in years")
    Sex: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
    BMI: float = Field(..., gt=10.0, lt=60.0, description="Body Mass Index")
    Children: int = Field(..., ge=0, le=15, description="Number of children/dependents")
    Smoker: int = Field(..., ge=0, le=1, description="0 = Non-smoker, 1 = Smoker")
    Region: int = Field(..., ge=0, le=3, description="Region code (0 to 3)")

# --- 5. EMBEDDED ADVANCED UI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise AI Predictor</title>
    <style>
        :root {
            --primary: #4A90E2;
            --bg-gradient: linear-gradient(135deg, #1A1A2E, #16213E, #0F3460);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text-color: #E0E0E0;
        }
        body {
            margin: 0; padding: 20px; font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-gradient); color: var(--text-color);
            min-height: 100vh; display: flex; justify-content: center; align-items: center;
        }
        .glass-card {
            background: var(--glass-bg); backdrop-filter: blur(20px);
            border-radius: 16px; border: 1px solid var(--glass-border);
            padding: 40px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
            width: 100%; max-width: 600px;
            animation: fadeIn 0.8s cubic-bezier(0.22, 1, 0.36, 1);
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        h1 { margin-top: 0; font-size: 1.8rem; text-align: center; color: #fff; letter-spacing: -0.5px; }
        p.subtitle { text-align: center; color: #A0A0B0; margin-bottom: 30px; font-size: 0.9rem; }
        
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 500px) { .form-grid { grid-template-columns: 1fr; } }
        
        .input-group { display: flex; flex-direction: column; }
        label { font-size: 0.85rem; font-weight: 500; margin-bottom: 8px; color: #B0C4DE; }
        input {
            background: rgba(0, 0, 0, 0.2); border: 1px solid var(--glass-border);
            color: #fff; padding: 12px 16px; border-radius: 8px; font-size: 1rem;
            transition: all 0.3s ease;
        }
        input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.2); }
        
        button {
            grid-column: 1 / -1; background: var(--primary); color: white;
            border: none; padding: 14px; border-radius: 8px; font-size: 1.1rem;
            font-weight: 600; cursor: pointer; transition: transform 0.2s, background 0.2s;
            margin-top: 10px; position: relative; overflow: hidden;
        }
        button:hover { background: #357ABD; transform: translateY(-2px); }
        button:disabled { opacity: 0.7; cursor: not-allowed; transform: none; }
        
        .loader {
            display: none; border: 3px solid rgba(255,255,255,0.3); border-top: 3px solid white;
            border-radius: 50%; width: 20px; height: 20px; animation: spin 1s linear infinite;
            position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
        }
        @keyframes spin { 0% { transform: translate(-50%, -50%) rotate(0deg); } 100% { transform: translate(-50%, -50%) rotate(360deg); } }
        
        .btn-text { transition: opacity 0.2s; }
        .loading .btn-text { opacity: 0; }
        .loading .loader { display: block; }

        #result-box {
            margin-top: 25px; padding: 20px; border-radius: 8px; background: rgba(74, 144, 226, 0.1);
            border: 1px solid rgba(74, 144, 226, 0.3); text-align: center; display: none;
            animation: fadeIn 0.4s ease;
        }
        .result-label { font-size: 0.9rem; color: #A0A0B0; text-transform: uppercase; letter-spacing: 1px; }
        .result-value { font-size: 2.5rem; font-weight: 700; color: #fff; margin-top: 5px; }
        .error { color: #FF6B6B; font-size: 0.9rem; text-align: center; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="glass-card">
        <h1>Health Cost Intelligence</h1>
        <p class="subtitle">Enter patient metrics to generate an AI-driven estimate.</p>
        
        <form id="aiForm">
            <div class="form-grid">
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" id="Age" required min="1" max="120" placeholder="e.g., 35">
                </div>
                <div class="input-group">
                    <label>Sex (0:F, 1:M)</label>
                    <input type="number" id="Sex" required min="0" max="1" placeholder="0 or 1">
                </div>
                <div class="input-group">
                    <label>BMI</label>
                    <input type="number" id="BMI" step="0.1" required min="10" max="60" placeholder="e.g., 25.5">
                </div>
                <div class="input-group">
                    <label>Children</label>
                    <input type="number" id="Children" required min="0" max="15" placeholder="e.g., 2">
                </div>
                <div class="input-group">
                    <label>Smoker (0:N, 1:Y)</label>
                    <input type="number" id="Smoker" required min="0" max="1" placeholder="0 or 1">
                </div>
                <div class="input-group">
                    <label>Region Code</label>
                    <input type="number" id="Region" required min="0" max="3" placeholder="0 - 3">
                </div>
                <button type="submit" id="submitBtn">
                    <span class="btn-text">Generate Prediction</span>
                    <div class="loader"></div>
                </button>
            </div>
        </form>
        
        <div id="error-msg" class="error"></div>
        <div id="result-box">
            <div class="result-label">Estimated Cost</div>
            <div class="result-value" id="prediction-value">$0.00</div>
        </div>
    </div>

    <script>
        document.getElementById('aiForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btn = document.getElementById('submitBtn');
            const errorMsg = document.getElementById('error-msg');
            const resultBox = document.getElementById('result-box');
            const valueSpan = document.getElementById('prediction-value');
            
            // Reset UI
            errorMsg.textContent = '';
            resultBox.style.display = 'none';
            btn.classList.add('loading');
            btn.disabled = true;
            
            // Build JSON Payload
            const payload = {
                Age: parseInt(document.getElementById('Age').value),
                Sex: parseInt(document.getElementById('Sex').value),
                BMI: parseFloat(document.getElementById('BMI').value),
                Children: parseInt(document.getElementById('Children').value),
                Smoker: parseInt(document.getElementById('Smoker').value),
                Region: parseInt(document.getElementById('Region').value)
            };

            try {
                const response = await fetch('/api/v1/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.detail ? JSON.stringify(data.detail) : 'Server error');
                }
                
                // Format as currency and animate in
                const formattedValue = new Intl.NumberFormat('en-US', { 
                    style: 'currency', currency: 'USD' 
                }).format(data.prediction);
                
                valueSpan.textContent = formattedValue;
                resultBox.style.display = 'block';
                
            } catch (err) {
                errorMsg.textContent = "Validation Error: Please check your inputs.";
                console.error(err);
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# --- 6. API ROUTES ---
@app.get("/", tags=["UI"])
async def serve_ui():
    """Serves the frontend application."""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/api/v1/predict", tags=["Machine Learning"])
async def generate_prediction(request: PredictionRequest):
    """
    Accepts patient metrics via JSON and returns the XGBoost prediction.
    """
    if "xg_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is currently unavailable. Please try again later.")
    
    try:
        # Convert validated Pydantic model directly to pandas DataFrame
        input_df = pd.DataFrame([request.model_dump()])
        
        # Log the incoming request for monitoring
        logger.info(f"Processing prediction for Age: {request.Age}, BMI: {request.BMI}")
        
        # Predict
        prediction = ml_models["xg_model"].predict(input_df)[0]
        
        return {
            "status": "success",
            "prediction": round(float(prediction), 2)
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")
