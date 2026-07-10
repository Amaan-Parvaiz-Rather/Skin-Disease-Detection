import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Ensure we have a model instance
# We use gemini-1.5-flash as it is fast and free for these types of lookups
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception:
    model = None

def get_disease_info(disease_name):
    """
    Calls the Gemini API to get structured information about a disease.
    Returns a dictionary with overview, symptoms, precautions, and when_to_see_doctor.
    """
    
    # Fallback response in case API fails or key is invalid
    fallback_response = {
        "disease_name": disease_name,
        "overview": f"{disease_name} is a skin condition that requires medical evaluation.",
        "symptoms": "Symptoms can vary. Please consult a dermatologist for accurate assessment.",
        "precautions": [
            "Do not scratch or irritate the affected area.",
            "Keep the area clean and dry.",
            "Avoid home remedies without consulting a doctor."
        ],
        "when_to_see_doctor": "Please consult a healthcare professional immediately for an accurate diagnosis."
    }

    if not model or not GEMINI_API_KEY or GEMINI_API_KEY == "your_key_here":
        print("⚠️ Gemini API Key missing or invalid. Returning fallback data.")
        return fallback_response

    prompt = f"""
    The patient has been diagnosed with "{disease_name}" using an AI skin disease detection system.
    Respond ONLY in valid JSON format with these exact keys, no markdown blocks, no extra text:
    {{
      "disease_name": "{disease_name.split(' Photos')[0]}",
      "overview": "2–3 sentence plain-English overview of what this condition is",
      "symptoms": "2 sentence description of common symptoms",
      "precautions": ["Precaution 1", "Precaution 2", "Precaution 3"],
      "when_to_see_doctor": "1 sentence about when to seek professional medical help"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean up any potential markdown formatting the model might return despite instructions
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        data = json.loads(text.strip())
        
        # Ensure precautions is a list
        if not isinstance(data.get("precautions"), list):
            data["precautions"] = fallback_response["precautions"]
            
        return data
        
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return fallback_response

if __name__ == "__main__":
    # Test script
    print("Testing Gemini API for Eczema...")
    info = get_disease_info("Eczema Photos")
    print(json.dumps(info, indent=2))
