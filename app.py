import pickle
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

app = FastAPI()

# Load the model
with open("xg_model.pkl", "rb") as f:
    model = pickle.load(f)

# Modern, animated UI embedded in HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Model Predictor</title>
    <style>
        body {
            margin: 0; padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            display: flex; justify-content: center; align-items: center;
            min-height: 100vh; color: white;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .glass-panel {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px; padding: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.2);
            width: 320px;
            animation: slideUp 1s ease-out forwards;
        }
        @keyframes slideUp {
            from { transform: translateY(40px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        h2 { text-align: center; margin-top: 0; font-weight: 300; }
        .input-group { margin-bottom: 15px; }
        .input-group label { display: block; margin-bottom: 5px; font-size: 0.9em; }
        .input-group input {
            width: 100%; padding: 10px; border: none; border-radius: 8px;
            background: rgba(255, 255, 255, 0.2); color: white;
            box-sizing: border-box; outline: none; transition: 0.3s;
        }
        .input-group input:focus { background: rgba(255, 255, 255, 0.3); box-shadow: 0 0 10px rgba(255,255,255,0.5); }
        .input-group input::placeholder { color: rgba(255,255,255,0.6); }
        button {
            width: 100%; padding: 12px; border: none; border-radius: 8px;
            background: rgba(255, 255, 255, 0.3); color: white;
            font-size: 1.1em; font-weight: bold; cursor: pointer;
            transition: all 0.3s ease; margin-top: 10px;
        }
        button:hover { background: rgba(255, 255, 255, 0.5); transform: translateY(-2px); }
        #result { text-align: center; margin-top: 20px; font-size: 1.2em; font-weight: bold; min-height: 24px;}
    </style>
</head>
<body>
    <div class="glass-panel">
        <h2>Predictor AI</h2>
        <form id="predictForm">
            <div class="input-group">
                <label>Age</label>
                <input type="number" name="Age" required placeholder="e.g. 35">
            </div>
            <div class="input-group">
                <label>Sex (0 or 1)</label>
                <input type="number" name="Sex" required placeholder="e.g. 0">
            </div>
            <div class="input-group">
                <label>BMI</label>
                <input type="number" step="0.1" name="BMI" required placeholder="e.g. 25.5">
            </div>
            <div class="input-group">
                <label>Children</label>
                <input type="number" name="Children" required placeholder="e.g. 2">
            </div>
            <div class="input-group">
                <label>Smoker (0 or 1)</label>
                <input type="number" name="Smoker" required placeholder="e.g. 1">
            </div>
            <div class="input-group">
                <label>Region (0 to 3)</label>
                <input type="number" name="Region" required placeholder="e.g. 2">
            </div>
            <button type="submit">Analyze</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = e.target.querySelector('button');
            const resultDiv = document.getElementById('result');
            btn.textContent = 'Processing...';
            
            const formData = new FormData(e.target);
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                resultDiv.innerHTML = `Predicted Value: <span style="color: #ffe600;">${data.prediction}</span>`;
            } catch (err) {
                resultDiv.textContent = "Error making prediction.";
            } finally {
                btn.textContent = 'Analyze';
            }
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def get_ui():
    return HTMLResponse(content=HTML_TEMPLATE)

@app.post("/predict")
async def predict(
    Age: int = Form(...),
    Sex: int = Form(...),
    BMI: float = Form(...),
    Children: int = Form(...),
    Smoker: int = Form(...),
    Region: int = Form(...)
):
    # Map features exactly as defined in the XGBoost model
    input_data = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "BMI": BMI,
        "Children": Children,
        "Smoker": Smoker,
        "Region": Region
    }])
    
    # Generate prediction
    pred = model.predict(input_data)[0]
    return {"prediction": round(float(pred), 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
