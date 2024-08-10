import joblib

def save_model(model, filename='model.joblib'):
    joblib.dump(model, filename)

def load_model(filename='model.joblib'):
    return joblib.load(filename)

if __name__ == '__main__':
    # Assuming the model was trained in part 2
    from model_building import build_and_evaluate_model
    model = build_and_evaluate_model()
    save_model(model)
    loaded_model = load_model()
