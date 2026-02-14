# For training use the command:
python script.py --mode train --dataset data_1
# Continue training from saved weights:
python script.py --mode train --dataset data_1 --weights best_model_data_1.pkl
# Evaluate a trained model:
python script.py --mode test --dataset data_1 --weights best_model_data_1.pkl


