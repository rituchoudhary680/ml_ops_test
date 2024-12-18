# ML Ops Test

## Objective
Predict trends based on given weather and energy data, and deploy the model using Docker, Poetry, and FastAPI.

## Instructions
1. **Data Preprocessing**: Review and preprocess the provided datasets.
2. **Model Training**: Train a machine learning model to predict energy consumption trends.
3. **Model Evaluation**: Evaluate the model's performance.
4. **Deployment**: Deploy the model using FastAPI and Docker.
5. **Submission**: Provide the Docker image and source code.

## Guiding/Generic Questions that might be asked during the follow up interview
1. **Data Preprocessing**
- What steps will you take to handle missing values in the datasets?
- How will you deal with any outliers in the data?
- What feature engineering techniques can you apply to improve model performance?
2. **Model Training**
- Which machine learning algorithms will you consider for this task and why?
- How will you split the data into training and testing sets?
- What metrics will you use to evaluate the model’s performance?
3. **Model Evaluation**
- How will you ensure that your model is not overfitting or underfitting?
- What techniques will you use for hyperparameter tuning?
- How will you interpret the results of your model evaluation?
4. **Deployment**
- How will you containerize your application using Docker?
- What steps will you take to ensure that your FastAPI application is scalable and efficient?
- How will you set up and test the API endpoints for your model?
5. **General**
- What challenges do you anticipate in this project, and how will you address them?
- How will you document your code and processes to ensure clarity and reproducibility?
- What additional tools or libraries might be useful for this project, and why?

## Steps
1. Clone the repository.
2. Install dependencies using Poetry.
3. Run the preprocessing script.
4. Train and evaluate the model.
5. Deploy the model using Docker.

## Commands
```bash
# Install dependencies
poetry install

# Run preprocessing
poetry run python src/preprocessing.py

# Train model
poetry run python src/model_training.py

# Evaluate model
poetry run python src/model_evaluation.py

# Build Docker image [Image being written successfully]
docker build -t ml_ops_test .

# Run Docker container [this command is not running, docker server is somehow creating problem]
docker run -p 8000:8000 ml_ops_test

# Run fastapi server locally
uvicorn src.api:app --reload
