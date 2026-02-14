# For training use the command:
'''bash
python script.py --mode train --dataset data_1
'''
# Continue training from saved weights:
'''bash
python script.py --mode train --dataset data_1 --weights best_model_data_1.pkl
'''
# Evaluate a trained model:
'''bash
python script.py --mode test --dataset data_1 --weights best_model_data_1.pkl
'''


