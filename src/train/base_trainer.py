"""Base trainer class for model training."""

from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer class."""
    
    def __init__(self, config):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    @abstractmethod
    def train_epoch(self, train_loader):
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
        """
        pass
    
    @abstractmethod
    def validate(self, val_loader):
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        pass
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
        """
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.train_epoch(train_loader)
            metrics = self.validate(val_loader)
            logger.info(f"Validation metrics: {metrics}")
