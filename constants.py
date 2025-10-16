"""
Constants and configuration values for AR training.
"""

# Numerical stability
EPSILON = 1e-8

# Default model parameters
DEFAULT_GRID_SIZE = 64
DEFAULT_N_MODES = [8, 8]
DEFAULT_HIDDEN_CHANNELS = 16
DEFAULT_N_LAYERS = 4

# Default training parameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_CLIP = 1.0

# Default temporal parameters
DEFAULT_T_IN = 10
DEFAULT_T_OUT = 5
DEFAULT_STEP_SIZE = 1
DEFAULT_TEACHER_FORCING_INITIAL = 1.0

# Default loss parameters
DEFAULT_ALPHA_TEMPORAL = 0.5
DEFAULT_LP_NORM_ORDER = 2
DEFAULT_SPATIAL_DIM = 2

# Scheduler parameters
DEFAULT_SCHEDULER_FACTOR = 0.7
DEFAULT_SCHEDULER_PATIENCE = 8

# Visualization parameters
DEFAULT_VIZ_FREQ = 20
DEFAULT_N_VIZ_SAMPLES = 4
DEFAULT_N_LONG_TERM_SAMPLES = 20

# Data split ratios
TRAIN_VAL_SPLIT = 0.9

# File extensions
CHECKPOINT_EXTENSION = '.pth'
CHECKPOINT_EXTENSION_ALT = '.pt'

# Checkpoint keys
CHECKPOINT_KEYS = {
    'epoch': 'epoch',
    'model_state': 'model_state_dict',
    'loss_state': 'loss_state_dict',
    'optimizer_state': 'optimizer_state_dict',
    'scheduler_state': 'scheduler_state_dict',
    'best_val_loss': 'best_val_loss',
    'train_losses': 'train_losses',
    'val_losses': 'val_losses',
    'train_loss_components': 'train_loss_components',
    'val_loss_components': 'val_loss_components',
    'loss_weights_history': 'loss_weights_history',
    'data_mean': 'data_mean',
    'data_std': 'data_std',
    'args': 'args',
}

# Progress reporting
PRINT_FREQ_BATCHES = 500

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_EPOCHS = 15
