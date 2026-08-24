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
    logger.info("Starting up server and loading XGBoost model...")
    try:
        with open("xg_model.pkl", "rb") as f:
            ml_models["xg_model"] = pickle.load(f)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
    
    yield 
    ml_models.clear()

# --- 3. APP INITIALIZATION ---
app = FastAPI(title="Premium AI Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. PYDANTIC DATA VALIDATION ---
class PredictionRequest(BaseModel):
    Age: int = Field(..., gt=0, lt=120)
    Sex: int = Field(..., ge=0, le=1)
    BMI: float = Field(..., gt=10.0, lt=60.0)
    Children: int = Field(..., ge=0, le=15)
    Smoker: int = Field(..., ge=0, le=1)
    Region: int = Field(..., ge=0, le=3)

# --- 5. PREMIUM UI TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Medical Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0; padding: 20px; font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #ffffff; min-height: 100vh;
            display: flex; justify-content: center; align-items: center;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px; padding: 40px;
            box-shadow: 0 25px 45px rgba(0,0,0,0.5);
            width: 100%; max-width: 650px;
            animation: slideIn 0.8s ease-out forwards;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 { margin: 0 0 5px 0; font-size: 2.2rem; text-align: center; font-weight: 800; background: -webkit-linear-gradient(#00d2ff, #3a7bd5); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        p.subtitle { text-align: center; color: #b0c4de; margin-bottom: 35px; font-weight: 300; }
        
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 550px) { .form-grid { grid-template-columns: 1fr; } }
        
        .input-group { display: flex; flex-direction: column; }
        label { font-size: 0.9rem; font-weight: 600; margin-bottom: 8px; color: #00d2ff; letter-spacing: 0.5px; }
        
        input, select {
            background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(0, 210, 255, 0.3);
            color: #fff; padding: 14px 18px; border-radius: 10px; font-size: 1rem;
            font-family: 'Poppins', sans-serif; transition: all 0.3s ease;
            appearance: none; /* Removes default OS styling for dropdowns */
        }
        input:focus, select:focus {
            outline: none; border-color: #00d2ff;
            box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
            background: rgba(0, 0, 0, 0.5);
        }
        select option { background: #203a43; color: white; } /* Dropdown list color */

        button {
            grid-column: 1 / -1; background: linear-gradient(to right, #00d2ff, #3a7bd5);
            color: white; border: none; padding: 16px; border-radius: 10px;
            font-size: 1.2rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;
            cursor: pointer; transition: transform 0.3s, box-shadow 0.3s;
            margin-top: 15px; position: relative; display: flex; justify-content: center; align-items: center;
        }
        button:hover {
            transform: scale(1.02); box-shadow: 0 10px 20px rgba(0, 210, 255, 0.4);
        }
        
        .loader {
            display: none; border: 3px solid rgba(255,255,255,0.3); border-top: 3px solid white;
            border-radius: 50%; width: 24px; height: 24px; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* BIG CENTERED ANIMATED RESULT BOX */
        .result-container {
            grid-column: 1 / -1;
            margin-top: 30px;
            display: none; /* Hidden by default */
            justify-content: center;
        }
        .result-box {
            background: rgba(0, 255, 136, 0.1);
            border: 2px solid #00ff88;
            border-radius: 15px;
            padding: 30px 40px;
            text-align: center;
            width: 100%;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
            animation: popUp 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
            opacity: 0; transform: scale(0.5);
        }
        @keyframes popUp {
            to { opacity: 1; transform: scale(1); }
        }
        
        .result-label { font-size: 1.1rem; color: #a0f0c0; text-transform: uppercase; font-weight: 600; letter-spacing: 2px; }
        .result-value { 
            font-size: 3.5rem; /* BADA FONT SIZE */
            font-weight: 800; color: #00ff88; 
            margin-top: 10px; text-shadow: 0 0 15px rgba(0,255,136,0.6);
        }
        .error { color: #ff4b4b; font-size: 1rem; text-align: center; grid-column: 1 / -1; margin-top: 10px; font-weight: 600;}
    </style>
</head>
<body>
    <div class="glass-card">
        <h1>AI Health Cost Predictor</h1>
        <p class="subtitle">Enter your details to generate a precise cost estimation.</p>
        
        <form id="aiForm">
            <div class="form-grid">
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" id="Age" required min="1" max="120" placeholder="e.g., 35">
                </div>
                
                <div class="input-group">
                    <label>Gender</label>
                    <!-- DROPDOWN ADDED HERE -->
                    <select id="Sex" required>
                        <option value="" disabled selected>Select Gender</option>
                        <option value="0">Female</option>
                        <option value="1">Male</option>
                    </select>
                </div>
                
                <div class="input-group">
                    <label>BMI (Body Mass Index)</label>
                    <input type="number" id="BMI" step="0.1" required min="10" max="60" placeholder="e.g., 25.5">
                </div>
                
                <div class="input-group">
                    <label>Number of Children</label>
                    <input type="number" id="Children" required min="0" max="15" placeholder="e.g., 2">
                </div>
                
                <div class="input-group">
                    <label>Smoker?</label>
                    <!-- DROPDOWN ADDED HERE -->
                    <select id="Smoker" required>
                        <option value="" disabled selected>Select option</option>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                
                <div class="input-group">
                    <label>Region</label>
                    <!-- DROPDOWN ADDED HERE -->
                    <select id="Region" required>
                        <option value="" disabled selected>Select your region</option>
                        <option value="0">Northeast</option>
                        <option value="1">Northwest</option>
                        <option value="2">Southeast</option>
                        <option value="3">Southwest</option>
                    </select>
                </div>
                
                <div id="error-msg" class="error"></div>
                
                <button type="submit" id="submitBtn">
                    <span id="btn-text">Calculate Premium</span>
                    <div class="loader" id="btn-loader"></div>
                </button>
                
                <!-- BADA ANIMATED RESULT BOX -->
                <div class="result-container" id="result-container">
                    <div class="result-box">
                        <div class="result-label">Estimated Insurance Cost</div>
                        <div class="result-value" id="prediction-value">$0.00</div>
                    </div>
                </div>

            </div>
        </form>
    </div>

    <script>
        document.getElementById('aiForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btnText = document.getElementById('btn-text');
            const btnLoader = document.getElementById('btn-loader');
            const btn = document.getElementById('submitBtn');
            const errorMsg = document.getElementById('error-msg');
            const resultContainer = document.getElementById('result-container');
            const valueSpan = document.getElementById('prediction-value');
            
            // UI Reset for processing
            errorMsg.textContent = '';
            resultContainer.style.display = 'none'; // Hide result box temporarily
            btnText.style.display = 'none';
            btnLoader.style.display = 'block';
            btn.disabled = true;
            
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
                
                // Format output
                const formattedValue = new Intl.NumberFormat('en-US', { 
                    style: 'currency', currency: 'USD' 
                }).format(data.prediction);
                
                // Set value and Show animated box
                valueSpan.textContent = formattedValue;
                resultContainer.style.display = 'flex'; // Triggers CSS PopUp Animation
                
            } catch (err) {
                errorMsg.textContent = "Error: Something went wrong. Make sure all fields are filled.";
                console.error(err);
            } finally {
                btnText.style.display = 'block';
                btnLoader.style.display = 'none';
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
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/api/v1/predict", tags=["Machine Learning"])
async def generate_prediction(request: PredictionRequest):
    if "xg_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    
    try:
        input_df = pd.DataFrame([request.model_dump()])
        prediction = ml_models["xg_model"].predict(input_df)[0]
        
        return {
            "status": "success",
            "prediction": round(float(prediction), 2)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")
