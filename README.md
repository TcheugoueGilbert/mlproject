# Student Performance Indicator

This project analyzes how students' performance (test scores) is affected by various factors such as gender, ethnicity, parental level of education, lunch, and test preparation course. It includes data exploration, preprocessing, model training, and evaluation.

## Project Structure

```
.
├── app.py
├── requirements.txt
├── setup.py
├── README.md
├── artifacts/
│   ├── data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── test.csv
│   └── train.csv
├── catboost_info/
├── logs/
├── mlproject.egg-info/
├── notebook/
├── src/
│   ├── logger.py
│   └── pipeline/
│       └── train_pipeline.py
└── templates/
```

## Features

- **Exploratory Data Analysis (EDA):** Understand the dataset and visualize relationships.
- **Data Preprocessing:** Clean and prepare data for modeling.
- **Model Training:** Train multiple machine learning models and evaluate their performance.
- **Logging:** Track experiments and errors using a custom logging utility ([src/logger.py](src/logger.py)).
- **Artifacts:** Store trained models and preprocessors for reuse.

## Getting Started

### Prerequisites

- Python 3.7+
- See [requirements.txt](requirements.txt) for dependencies.

### Installation

1. Clone the repository:
    ```sh
    git clone <repo-url>
    cd mlproject
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. **Data Exploration:**  
   Explore the data using the notebooks in [notebook/](notebook/).

2. **Train Models:**  
   Run the training pipeline:
   ```sh
   python src/pipeline/train_pipeline.py
   ```

3. **Logs:**  
   Check the [logs/](logs/) directory for log files generated during training.

4. **Artifacts:**  
   Trained models and preprocessors are saved in [artifacts/](artifacts/).

### Project Lifecycle

- Understanding the Problem Statement
- Data Collection
- Data Checks
- Exploratory Data Analysis
- Data Pre-Processing
- Model Training
- Model Selection

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE) (add a LICENSE file if needed)

## Acknowledgements

- Inspired by student performance datasets and ML best practices.