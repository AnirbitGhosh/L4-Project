import os
import h5py
import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms

class PCamData(data_utils.Dataset):
    def __init__(self, path, mode="train", batch_size=32, n_iters=None, augment=False):
        super().__init__()
        
        self.n_iters = n_iters
        self.batch_size = batch_size
        
        assert mode in ["train", "test", "validation"]
        base_name = "{}/camelyonpatch_level_2_split_{}_{}.h5" 
        
        print('\n')
        print("# "*50)
        print(f"Loading {mode} dataset...")
        
        #Open files
        h5x = h5py.File(os.path.join(path, base_name.format(mode, mode, 'x')), 'r')
        h5y = h5py.File(os.path.join(path, base_name.format(mode, mode, 'y')), 'r')
        
        # Read to numpy arrays
        self.X = np.array(h5x.get('x'))
        self.y = np.array(h5y.get('y'))
        
        print(f"Loaded {mode} dataset with {len(self.X)} samples")
        print("# "*50)
        
        if augment:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ColorJitter(brightness=.5, saturation=.25, hue=.1, contrast=.5),
                                                 transforms.RandomAffine(10, (0.05, 0.05), fillcolor=(255, 255, 255)),
                                                 transforms.RandomHorizontalFlip(.5),
                                                 transforms.RandomVerticalFlip(.5),
                                                 transforms.ToTensor()])
        else: 
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.ToTensor()])
        
    def __getitem__(self, item):
        idx = item % self.__len__()
        _slice = slice(idx*self.batch_size, (idx+1) * self.batch_size)
        images  = self._transform(self.X[_slice])
        labels = torch.tensor(self.y[_slice].astype(np.float32)).view(-1,1)
        return {'images' : images, 'labels' : labels}
    
    def _transform(self, images):
        tensors = []
        for image in images:
            tensors.append(self.transform(image))
        return torch.stack(tensors)
    
    def __len__(self):
        return len(self.X)//self.batch_size
        