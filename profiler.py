import time
import os
import glob
import cv2
import math
import random
import my_framework as nn

# --- Configuration ---



def profile_model(model, input_size=(3, 32, 32)):
    """
    Profiler for custom frameworks using ctypes/raw arrays.
    Calculates Params, MACs, and FLOPs.
    """
    total_params = 0
    total_macs = 0
    c, h, w = input_size

    print(f"\n{'Layer':<15} | {'Output Shape':<15} | {'Params':<10} | {'MACs':<10}")
    print("-" * 65)

    # We iterate through the layers defined in the CNN class
    for name, layer in model.__dict__.items():
        layer_params = 0
        layer_macs = 0
        shape_str = ""

        # Use class names to identify layers
        layer_type = layer.__class__.__name__

        if "Conv2d" in layer_type:
                    params = layer.parameters()
                    layer_params = sum(len(p.data) for p in params)
                    
                    # --- FIXED PROFILER DYNAMICS ---
                    k = layer.k
                    p = layer.p
                    s = layer.s
                    out_c = layer.out_c
                    in_c = c 
                    
                    # Use the actual output dimension formula
                    h = (h + 2 * p - k) // s + 1
                    w = (w + 2 * p - k) // s + 1
                    
                    layer_macs = (h * w) * (k * k * in_c * out_c)
                    c = out_c
                    shape_str = f"({c}, {h}, {w})"

        elif "MaxPool2d" in layer_type:
            h //= 2
            w //= 2
            shape_str = f"({c}, {h}, {w})"
            layer_macs = 0

        elif "Linear" in layer_type:
            params = layer.parameters()
            layer_params = sum(len(p.data) for p in params)
            
            # Linear weights: in_features * out_features
            # out_features is the length of the bias
            out_f = len(params[1].data)
            in_f = len(params[0].data) // out_f
            
            layer_macs = in_f * out_f
            shape_str = f"({out_f})"

        elif "ReLU" in layer_type:
            continue # Skip activations for the table

        if shape_str:
            total_params += layer_params
            total_macs += layer_macs
            print(f"{name:<15} | {shape_str:<15} | {layer_params:<10,} | {layer_macs:<10,}")

    print("-" * 65)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print(f"TOTAL MACs:       {total_macs:,}")
    print(f"TOTAL FLOPs:      {total_macs * 2:,}")
    print("-" * 65)