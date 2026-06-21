"""Dataset module for data loading and preprocessing."""


class Dataset:
    """Base dataset class."""
    
    def __init__(self, root_dir, transform=None):
        """Initialize dataset.
        
        Args:
            root_dir: Root directory of the dataset
            transform: Optional transforms to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
