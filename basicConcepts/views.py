import os
from django.shortcuts import render
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    'aggression': './Models/aggression_parsed_dataset_svm_model.joblib',
    'attack': './Models/attack_parsed_dataset_svm_model.joblib',
    'other': './Models/kaggle_parsed_dataset_svm_model.joblib',
    'toxicity': './Models/toxicity_parsed_dataset_svm_model.joblib',
    'racism': './Models/twitter_racism_parsed_dataset_svm_model.joblib',
    'sexism': './Models/twitter_sexism_parsed_dataset_svm_model.joblib',
}

VECTORIZER_PATHS = {
    'aggression': './Models/aggression_parsed_dataset_svm_vectorizer.joblib',
    'attack': './Models/attack_parsed_dataset_svm_vectorizer.joblib',
    'other': './Models/kaggle_parsed_dataset_svm_vectorizer.joblib',
    'toxicity': './Models/toxicity_parsed_dataset_svm_vectorizer.joblib',
    'racism': './Models/twitter_racism_parsed_dataset_svm_vectorizer.joblib',
    'sexism': './Models/twitter_sexism_parsed_dataset_svm_vectorizer.joblib',

}

models = {}

for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        models[model_name] = load(model_path)
    else:
        print(f"Model path not found: {model_path}")

def Welcome(request):
    return render(request, 'index.html')

def form_view(request):
    return render(request, 'form.html')

def formInfo(request):
    name = request.GET['username']
    message = request.GET['message']

    y_pred = "Not Cyberbullying"
    for dataset_name, model in models.items():
        vectorizer = load(VECTORIZER_PATHS[dataset_name])
        user_input_vectorized = vectorizer.transform([message])

        prediction = model.predict(user_input_vectorized)

        if prediction == 1:
            y_pred = f"Cyberbullying detected (Type: {dataset_name})"
            break  
        
    print(f"Debug: y_pred - {y_pred}")

    data = {'name': name, 'message': message, 'result': y_pred}
    return render(request, 'result.html', data)
