@echo off
cd /d "%~dp0src"

echo ============================================
echo  STEP 1/2: Training the GNN (GraphSAGE)
echo  ~30 seconds. Please wait...
echo ============================================
python train.py --model sage --epochs 5 --batch_size 512

echo.
echo ============================================
echo  STEP 2/2: Running baseline comparison
echo  (RandomForest + GradientBoosting)
echo ============================================
python baseline.py

echo.
echo ============================================
echo  ALL DONE. Scroll up to see the results.
echo  Trained model saved in: outputs\sage_model.pt
echo  Train/test data saved in: data\processed\
echo ============================================
pause
