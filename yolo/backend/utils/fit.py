# -*- coding: utf-8 -*-
import os
import time

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


def train(model,
         loss_func,
         train_batch_gen,
         valid_batch_gen,
         learning_rate = 1e-4,
         nb_epoch = 300,
         saved_weights_name = 'best_weights.h5'):
    """A function that performs training on a general keras model.

    # Args
        model : keras.models.Model instance
        loss_func : function
            refer to https://keras.io/losses/

        train_batch_gen : keras.utils.Sequence instance
        valid_batch_gen : keras.utils.Sequence instance
        learning_rate : float
        saved_weights_name : str
    """
    # 1. create optimizer
    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    # 2. create loss function
    model.compile(loss=loss_func,
                  optimizer=optimizer)

    # 4. training
    train_start = time.time()
    model.fit_generator(generator = train_batch_gen,
                        steps_per_epoch  = len(train_batch_gen), 
                        epochs           = nb_epoch,
                        validation_data  = valid_batch_gen,
                        validation_steps = len(valid_batch_gen),
                        callbacks        = _create_callbacks(saved_weights_name),                        
                        verbose          = 1,
                        workers          = 3,
                        max_queue_size   = 8)
    
    _print_time(time.time()-train_start)

def _print_time(process_time):
    if process_time < 60:
        print("{:d}-seconds to train".format(int(process_time)))
    else:
        print("{:d}-mins to train".format(int(process_time/60)))

def _create_callbacks(saved_weights_name):
    # Make a few callbacks
    early_stop = EarlyStopping(monitor='val_loss', 
                       min_delta=0.001, 
                       patience=10, 
                       mode='min', 
                       verbose=1)
    checkpoint = ModelCheckpoint(saved_weights_name, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)
#     tb_counter  = len([log for log in os.listdir(os.path.expanduser('~/logs/')) if 'yolo' in log]) + 1
#     tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/') + 'yolo' + '_' + str(tb_counter), 
#                               histogram_freq=0, 
#                               write_graph=True, 
#                               write_images=False)
    callbacks = [early_stop, checkpoint]
    return callbacks
