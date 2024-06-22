import os
from pathlib import Path

project_name = "DiamondPricePrediction"

listOfFiles = [
    ".github/workflows/ci.yaml",
    "src/__init__.py",
    f"src/component/__init__.py",
    f"src/component/data_ingestion.py",
    f"src/component/data_transformation.py",
    f"src/component/model_evaluation.py",
    f"src/component/model_trainer.py",
    f"src/logFile/loggingInfo.py",
    f"src/logFile/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/pipeline/training_pipeline.py",
    f"src/pipeline/prediction_pipeline.py",
    f"src/utils/__init__.py",
    f"src/exception/exception.pytests/__init__.py",
    f"src/exception/__init__.py",
    f"src/exception/exception.py",
    f"test/__init__.py",
    f"test/unit_test.py",
    f"test/integrated_test.py",
    f"experiment/experiment.ipynb",
    f"templates/index.html",
    f"templates/form.html",
    f"templates/prediction.html",
    "app.py",
    "setup.py",
    ".env",
    "tox.ini",
    "README.md",
    "Dockerfile",
    "docker_compose.yaml",
    "requirements.txt"
    ]

for file in listOfFiles:
    filePath = Path(file)
    fileDir, fileEndPath = os.path.split(filePath)
    
    if (fileDir != ""):
        os.makedirs(fileDir, exist_ok=True)
        
    if (not os.path.exists(filePath)):
        with open(filePath, "w") as f:
            pass
        
    else:
        print("file already exists")