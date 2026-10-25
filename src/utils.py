import logging
import random
import os
import numpy as np

def setup_logger(log_file: str = "pipeline.log") -> logging.Logger:
    """Configures structured logging with console and file handlers."""
    logger = logging.getLogger("HotelNoShowPipeline")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File Handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def set_seed(seed: int = 42) -> None:
    """Sets global random seed for reproducibility across random, numpy, and python."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
