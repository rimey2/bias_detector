import streamlit as st
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Détecteur et Correcteur de Biais",
    layout="wide"
)

@st.cache_resource


def load_detection_model():
    """Charge le modèle de détection de biais"""
    tokenizer = AutoTokenizer.from_pretrained("C:/Users/aboky/Documents/ESGI/NLP/projet/tokeniser_biais")
    model = AutoModelForSequenceClassification.from_pretrained("C:/Users/aboky/Documents/ESGI/NLP/projet/model_biais")
    return model, tokenizer

@st.cache_resource

def load_correction_model():
    """Charge le modèle de correction"""
    model_name = "gpt2"  # 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def detect_bias(text, model, tokenizer):
    """Détecte les biais dans le texte"""
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()
    
    label_map = {0: 'stereotype', 1: 'anti-stereotype', 2: 'unrelated'}
    
    # Déterminer le type de biais (à adapter selon votre modèle)
    bias_types = {
        'stereotype': {
            'race': ['ethnicity', 'nationality', 'color'],
            'religion': ['faith', 'belief', 'worship'],
            'gender': ['sex', 'male', 'female'],
            'profession': ['job', 'career', 'occupation']
        }
    }
    
    # Analyse simple du type de biais (à améliorer selon vos besoins)
    detected_type = None
    text_lower = text.lower()
    for bias_type, keywords in bias_types['stereotype'].items():
        if any(keyword in text_lower for keyword in keywords):
            detected_type = bias_type
            break
    
    return {
        'label': label_map[predicted_class],
        'confidence': confidence,
        'bias_type': detected_type if label_map[predicted_class] == 'stereotype' else None
    }

def load_correction_model():
    """Charge le modèle de correction"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # Ajouter les tokens spéciaux si nécessaire
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def correct_bias(text, bias_type, model, tokenizer):
    """Corrige le biais dans le texte en utilisant directement le modèle"""
    prompt = f"Original: {text}\nCorrected (remove {bias_type} bias):"
    
    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Générer le texte
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            do_sample=True
        )
        
        # Décoder la sortie
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire la partie corrigée
        correction = generated_text.split("Corrected")[1].strip()
        return correction if correction else "Impossible de générer une correction."
        
    except Exception as e:
        return f"Erreur lors de la génération: {str(e)}"







def main():
    st.title("📝 Bias Dectector App")

    url = "https://www.linkedin.com/in/rimey-aboky-25603a20b/"
    mygithub = "https://github.com/rimey2/bias_detector.git"

    st.subheader(
        "This application allows you to detect small biais in texts"
    )
    st.sidebar.write("[Author : Rimey ABOKY](%s)" % url)
    st.sidebar.write("[Github : rimey2](%s)" % mygithub)
    st.sidebar.markdown(
        "**To get started, follow the instructions below :** \n"
        "1. Copy and paste or type your texts in the rectangle area, \n"
        "1. Analyze your text with the button , \n"
        "1. Check results on the right side of your screen, \n"

    )
    
    # Description de l'application
    st.write("""
    We get rid of all your biased texts !!
    """)
    
    # Chargement des modèles
    try:
        detection_model, detection_tokenizer = load_detection_model()
        correction_model, correction_tokenizer = load_correction_model()
        
        # Interface principale
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Text")
            user_text = st.text_area(
                "Type your text here:",
                height=200,
                key="input_text"
            )
        
        if st.button("Analyze", type="primary"):
            if user_text:
                with st.spinner("Analyzing ..."):
                    # Détection de biais
                    result = detect_bias(user_text, detection_model, detection_tokenizer)
                    
                    with col2:
                        st.subheader("Analysis Results")
                        
                        if result['label'] == 'stereotype':
                            st.error("⚠️ Bias detected!")
                            st.write(f"Type of bias : {result['bias_type']}")
                            st.write(f"Confidence: {result['confidence']:.2%}")
                            
                            if st.button("Corriger le texte"):
                                with st.spinner("Generating  correction..."):
                                    correction = correct_bias(
                                        user_text,
                                        result['bias_type'],
                                        correction_model,
                                        correction_tokenizer
                                    )
                                    st.success("Version corrigée:")
                                    st.write(correction)
                                    
                        elif result['label'] == 'anti-stereotype':
                            st.success("✅ Unbiased text")
                            st.write(f"Confidence: {result['confidence']:.2%}")
                        else:
                            st.info("ℹ️ Neutral text")
                            st.write(f"Confidence: {result['confidence']:.2%}")
                        
                        # Afficher les détails techniques
                        with st.expander("Show technicals details"):
                            st.json(result)
            else:
                st.error("⚠️ Please type a texte to analyze")
                
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        st.write("Assurez-vous que les modèles sont correctement sauvegardés dans les dossiers appropriés.")

if __name__ == "__main__":
    main()