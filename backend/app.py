import os
import io
import time
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

# Add parent directory to path so we can import our existing models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import IDX_TO_CLASS, NUM_CLASSES
from dataset import get_val_transforms
from models import create_model, get_model_names

app = FastAPI(title="Brain Tumor AI Analysis")

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints_model')

# Global variables to hold loaded models
ensemble_models = []
is_models_loaded = False

RECOMMENDATIONS = {
    'glioma_tumor': "A glioma is a common type of tumor originating in the brain or spinal cord. Immediate consultation with a neuro-oncologist is recommended to plan surgical, astrocytic, or radiotherapeutic interventions based on the tumor grade.",
    'meningioma_tumor': "Meningiomas are typically slow-growing tumors that form on the membranes covering the brain and spinal cord just inside the skull. While often benign, careful monitoring via regular MRI scans or surgical removal depending on symptoms is advised.",
    'pituitary_tumor': "Pituitary tumors are abnormal growths that develop in your pituitary gland. Blood tests to check hormone levels and consultation with an endocrinologist and a neurosurgeon should be the immediate next step.",
    'no_tumor': "No signs of a brain tumor detected in this MRI scan. If symptoms persist, consider consulting a neurologist for alternative diagnoses."
}

def load_ensemble():
    global ensemble_models, is_models_loaded
    if is_models_loaded:
        return
    
    print(f"Loading models from {CHECKPOINT_DIR} on {DEVICE}...")
    model_names = get_model_names()
    
    for model_name in model_names:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        if os.path.exists(ckpt_path):
            model = create_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
            checkpoint = torch.load(ckpt_path, weights_only=True, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)
            model.eval()
            ensemble_models.append((model_name, model))
            print(f"Loaded {model_name}")
        else:
            print(f"WARNING: Checkpoint {ckpt_path} not found.")
            
    is_models_loaded = True
    print(f"Successfully loaded {len(ensemble_models)} models for ensemble.")

@app.on_event("startup")
async def startup_event():
    # Attempt to pre-load models in background
    try:
        load_ensemble()
    except Exception as e:
        print(f"Startup loading failed (will retry on predict): {e}")

@app.get("/")
def read_root():
    return {"status": "Analysis Engine Running", "models_loaded": len(ensemble_models)}

@app.post("/analyze")
async def analyze_mri(file: UploadFile = File(...)):
    if not is_models_loaded:
        load_ensemble()
        
    if not ensemble_models:
        return {"error": "No models were loaded. Please check model checkpoints."}
        
    start_time = time.time()
    
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Preprocess
    transform = get_val_transforms()
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    import matplotlib.pyplot as plt
    import base64
    
    all_probs = []
    saliency_b64 = None
    
    # Run through ensemble
    input_tensor.requires_grad_()
    
    for i, (name, model) in enumerate(ensemble_models):
        if i == 0:
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
            all_probs.append(probs)
            
            # Calculate Saliency Heatmap
            pred_idx_sal = outputs.argmax(dim=1).item()
            score = outputs[0, pred_idx_sal]
            score.backward()
            
            saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
            saliency = saliency.squeeze().cpu().numpy()
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            colormap = plt.get_cmap('jet')
            heatmap = (colormap(saliency) * 255).astype(np.uint8)[:,:,:3]
            heatmap_img = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)
            blended = Image.blend(image.convert('RGB'), heatmap_img, alpha=0.45)
            
            buffered = io.BytesIO()
            blended.save(buffered, format="JPEG")
            saliency_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            input_tensor.grad.zero_()
        else:
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                all_probs.append(probs)
            
    # Soft voting
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = float(np.max(avg_probs))
    
    # Prepare individual probabilities mapping
    prob_dict = {IDX_TO_CLASS[i]: float(avg_probs[i]) * 100 for i in range(NUM_CLASSES)}
    
    # Sort for UI descending
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Create the detailed breakdown
    breakdown = [{"label": format_label(k), "percentage": round(v, 2)} for k, v in sorted_probs]
    
    elapsed = time.time() - start_time
    
    return {
        "diagnosis": format_label(pred_label),
        "raw_label": pred_label,
        "confidence": round(confidence * 100, 2),
        "breakdown": breakdown,
        "recommendation": RECOMMENDATIONS[pred_label],
        "processing_time_ms": round(elapsed * 1000),
        "saliency_b64": saliency_b64
    }

@app.post("/chat")
async def chat(message: str = Form(...), diagnosis: str = Form("")):
    # A simple deterministic rule-based chatbot for demo purposes.
    # We could plug an LLM here, but rule-based ensures it runs smoothly locally.
    
    msg_lower = message.lower()
    
    response = "I am a medical assistant specialized in brain tumors. I can explain MRI results, types of tumors, or next steps."
    
    if "treatment" in msg_lower or "cure" in msg_lower or "what to do" in msg_lower:
        if "glioma" in diagnosis.lower():
            response = "For Gliomas, treatment typically involves surgery followed by radiation and chemotherapy. The exact plan depends heavily on the tumor's grade."
        elif "meningioma" in diagnosis.lower():
            response = "Many meningiomas are benign. If small, your doctor may suggest 'watchful waiting'. Otherwise, surgery is the primary treatment."
        elif "pituitary" in diagnosis.lower():
            response = "Pituitary tumors are often treated with surgery through the nose (transsphenoidal), or sometimes medication to control hormone levels."
        else:
            response = "Treatments vary drastically depending on the tumor type. Surgery, radiation, and chemotherapy are standard. What specifically are you concerned about?"
            
    elif "dangerous" in msg_lower or "fatal" in msg_lower or "benign" in msg_lower:
        response = "The danger heavily depends on the grading (I-IV). Meningiomas are mostly benign (non-cancerous), while Gliomas can range from slow-growing to aggressive (Glioblastoma). Always discuss gradings with your oncologist."
        
    elif "thank" in msg_lower:
        response = "You are very welcome. Let me know if you have any other questions regarding the diagnosis."
        
    elif "what is" in msg_lower and "glioma" in msg_lower:
        response = "A glioma is a tumor that starts in the glial cells of the brain or spine, making up about 33% of all brain tumors."
        
    elif "accuracy" in msg_lower or "sure" in msg_lower:
        response = "Our state-of-the-art AI ensemble achieves roughly 96-97% analytical accuracy. However, this is an assisting tool; a clinical biopsy and professional review remain completely necessary."
        
    else:
        if diagnosis and "no_tumor" not in diagnosis.lower():
            response = f"Given the preliminary finding of {diagnosis}, I recommend compiling a list of questions to ask your neurosurgeon during your consultation."
            
    return {"reply": response}

def format_label(label_str):
    if label_str == 'no_tumor':
        return 'No Tumor Detected'
    return label_str.replace('_tumor', '').capitalize() + ' Tumor'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
