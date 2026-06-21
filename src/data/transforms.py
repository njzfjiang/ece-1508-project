"""Image transformation utilities."""

from torchvision import transforms


def get_train_transforms(image_size=256):
    """Get training transforms.
    
    Args:
        image_size: Size of the image
        
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_eval_transforms(image_size=256):
    """Get evaluation transforms.
    
    Args:
        image_size: Size of the image
        
    Returns:
        Composed transforms for evaluation
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
