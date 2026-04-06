import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")

# ==========================================
# GLOBAL ASSET LOADING (Startup)
# ==========================================
print("Loading Model Assets...")

# Dataset 1: Efficiency (City KMPL)
yeo_1 = joblib.load('deployment_models/d1_yeo_transform.joblib')
preprocessor_1 = joblib.load('deployment_models/d1_preprocessor.joblib')
model_1 = joblib.load('deployment_models/d1_xgboost_model.joblib')

# Dataset 2: Emissions (CO2 g/km)
yeo_2 = joblib.load('deployment_models/d2_yeo_transform.joblib')
preprocessor_2 = joblib.load('deployment_models/d2_preprocessor.joblib')
model_2 = joblib.load('deployment_models/d2_svr_model.joblib')

# ==========================================
# INFERENCE LOGIC FUNCTIONS
# ==========================================
def get_d1_prediction(cylinders, displacement, highway_kmpl, make, car_class, drive, fuel, trans):
    # 1. Include 'highway_kmpl' in this dictionary
    data = pd.DataFrame([{
        'cylinders': cylinders, 
        'displacement': displacement,
        'highway_kmpl': highway_kmpl,  # Added this line
        'make_std': make, 
        'class_std': car_class, 
        'drive_std': drive,
        'fuel_std': fuel, 
        'trans_std': trans
    }])

    # 2. Update the transform call to include all numerical columns seen during fit
    # Based on your error, yeo_1 was fitted on [cylinders, displacement, highway_kmpl]
    cols_to_transform = ['cylinders', 'displacement', 'highway_kmpl']
    data[cols_to_transform] = yeo_1.transform(data[cols_to_transform])

    # 3. Preprocess & Predict
    X_proc = preprocessor_1.transform(data)
    return float(model_1.predict(X_proc)[0])

def get_d2_prediction(displacement, cylinders, hwy_kmpl, city_kmpl, make, car_class, drive, fuel, trans):
    data = pd.DataFrame([{
        'displacement': displacement, 'cylinders': cylinders,
        'highway_kmpl': hwy_kmpl, 'city_kmpl': city_kmpl,
        'make_std': make, 'class_std': car_class, 'drive_std': drive,
        'fuel_std': fuel, 'trans_std': trans
    }])
    # Yeo-Johnson for D2
    num_cols = ['displacement', 'cylinders', 'highway_kmpl', 'city_kmpl']
    data[num_cols] = yeo_2.transform(data[num_cols])
    # Preprocess & Predict
    X_proc = preprocessor_2.transform(data)
    return float(model_2.predict(X_proc)[0])

# ==========================================
# ROUTES
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
   # New, explicit way
    return templates.TemplateResponse(
        request=request, 
        name="home.html", 
        context={"request": request}
    )

@app.post("/predict_all", response_class=HTMLResponse)
async def predict_all(
    request: Request,
    displacement: float = Form(...),
    cylinders: float = Form(...),
    city_kmpl: float = Form(...),
    highway_kmpl: float = Form(...),
    make_std: str = Form(...),
    class_std: str = Form(...),
    drive_std: str = Form(...),
    fuel_std: str = Form(...),
    trans_std: str = Form(...)
):
    # Store inputs in a dictionary to send back to UI
    user_inputs = {
        "displacement": displacement, "cylinders": cylinders,
        "city_kmpl": city_kmpl, "highway_kmpl": highway_kmpl,
        "make_std": make_std, "class_std": class_std,
        "drive_std": drive_std, "fuel_std": fuel_std, "trans_std": trans_std
    }

    # Run your inference
    res_city_kmpl = get_d1_prediction(cylinders, displacement, highway_kmpl, make_std, class_std, drive_std, fuel_std, trans_std)
    res_co2 = get_d2_prediction(displacement, cylinders, highway_kmpl, res_city_kmpl, make_std, class_std, drive_std, fuel_std, trans_std)
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "inputs": user_inputs,  # <--- THIS PASSES DATA BACK
        "prediction_text": f"{res_city_kmpl:.2f}",
        "prediction_text_2": f"{res_co2:.2f} g/km"
    })

