# End-to-End Machine Learning Pipeline

This repository contains a machine learning pipeline from data cleaning to model deployment. It includes data preprocessing, model training, API development with FastAPI, and a user interface using Gradio, all deployed on Hugging Face Spaces.

## Contents

- **Part 1: Data Cleaning**
- **Part 2: Model Building**
- **Part 3: Model Saving and Loading**
- **Part 4: FastAPI Endpoint**
- **Part 5: Gradio UI Deployment**

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

## Requirements

- Python 3.7+
- Libraries: `pandas`, `scikit-learn`, `joblib`, `fastapi`, `uvicorn`, `gradio`
- An account on [Hugging Face](https://huggingface.co/)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `.\env\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Cleaning

Clean the dataset, handle missing values, encode categorical variables, and standardize features.

Run the script:

```bash
python part1/data_cleaning.py
```

### 2. Model Building

Train and evaluate a logistic regression model.

Run the script:

```bash
python part2/model_building.py
```

### 3. Model Saving and Loading

Save and load the trained model using joblib.

Run the script:

```bash
python part3/model_io.py
```

### 4. FastAPI Endpoint

Create an API endpoint to serve the model for predictions.

Run the server:

```bash
uvicorn part4.api:app --reload
```

### 5. Gradio UI Deployment

Create a simple UI for the model using Gradio.

Run the Gradio app:

```bash
python part5/app.py
```

## Deployment on Hugging Face Spaces

1. Create a Space on Hugging Face and select the "Gradio" template.
2. Upload `app.py`, `requirements.txt`, and any necessary files.
3. Deploy the application.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
```

### Key Changes

- Simplified and focused on key instructions for running the project.
- Maintained essential project information without excessive detail.
- Streamlined sections for ease of understanding and quick setup.

Feel free to adjust the repository name and URL to match your project. This version provides a clear and concise guide for anyone looking to run the project or understand its structure.
