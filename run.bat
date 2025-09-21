@echo off
echo 🚀 Starting training...
python src/train.py

echo 📊 Evaluating best model...
python src/evaluate.py

echo ✅ All steps completed successfully.
echo 👉 To serve the model locally, run:
echo    uvicorn app.main:app --reload --port 8000
pause