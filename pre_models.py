# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:59:10 2022

@author: Mengqi Liu
"""
import pytorch_forecasting as pf

def get_model(args, training):
    if args.model_name == "deepar":
        model = pf.models.deepar.DeepAR.from_dataset(dataset=training, 
                                                    learning_rate=args.learning_rate,
                                                    cell_type=args.cell_type, 
                                                    hidden_size=args.hidden_size, 
                                                    rnn_layers=args.rnn_layers, 
                                                    dropout=0.1,
                                                    loss=pf.metrics.NormalDistributionLoss(),
                                                    log_interval=args.log_interval,
                                                    log_val_interval=args.log_val_interval,
                                                    # reduce_on_plateau_patience=3,
                                                    )
    else:
        model = None
        print("Invalid Model Name!")
    return model