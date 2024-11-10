import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_imgfile(filename):
    """Checking if the file format is image format"""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_datapaths(train_dir):
    """To create dataset, we need to save paths of images"""
    image_paths = []
    assert os.path.isdir(train_dir), f"{train_dir} does not exist"

    for root, _, names in os.walk(train_dir):
        for name in names:
            if is_imgfile(name):
                path = os.path.join(root, name)
                image_paths.append(path)
    return image_paths


def get_transform(opt):
    """Transforms images to grayscale if opt.channels == 1."""
    transform = []
    
    if opt.rotate:
        transform.append(transforms.RandomRotation(0.5))
    
    transform.append(transforms.ColorJitter(brightness=opt.brightness))
    
    # Resize image to the desired size
    transform.append(transforms.Resize((opt.cropsize, opt.cropsize), interpolation=2))
    
    # Convert image to grayscale if opt.channels == 1
    if opt.channels == 1:
        transform.append(transforms.Grayscale(num_output_channels=1))  # Convert RGB to grayscale
    
    transform.append(transforms.ToTensor())
    
    # Normalize the grayscale image (1 channel) or RGB image (3 channels)
    if opt.channels == 1:
        transform.append(transforms.Normalize((0.5,), (0.5,)))  # For grayscale
    else:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # For RGB

    
    return transforms.Compose(transform)



class ImageFolder(data.Dataset):
    """Custom ImageFolder dataset for loading images"""

    def __init__(self, root, transform=None, return_paths=False):
        imgs = get_datapaths(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths

    def __getitem__(self, index):
        path = self.imgs[index]
        
        # Convert image to grayscale if opt.channels == 1
        if self.transform is not None and hasattr(self.transform, 'opt') and self.transform.opt.channels == 1:
            img = Image.open(path).convert('L')  # 'L' mode for grayscale
        else:
            img = Image.open(path).convert('RGB')  # Convert image to RGB if not grayscale
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.return_paths:
            return img, path
        else:
            return img
            
    def __len__(self):
        return len(self.imgs)

