import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ColorizationDataset(BaseDataset):
    """This dataset class can load a set of natural images in RGB, and convert RGB format into (L, ab) pairs in Lab color space.

    This dataset is required by pix2pix-based colorization model ('--model colorization')
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        By default, the number of channels for input image  is 1 (L) and
        the number of channels for output image is 2 (ab). The direction is from A to B
        """
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')      # 覆盖并设置原有的参数，返回一个“传参集合”: parser。
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir, opt.max_dataset_size))    # 从dataset中读取dataset目录
        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')    # 确认参数信息
        self.transform = get_transform(self.opt, convert=False) #将图像转换为PyTorch Tensor

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - the L channel of an image
            B (tensor) - - the ab channels of the same image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        #__getitem__在这里的作用在于，将每个A、B数据对，转化为PyTorch Tensor，最终放到Dict中去

        path = self.AB_paths[index]                 # 根据index进入每个子目录，注意这个子目录应当这么写！
                                                    # 类中的临时变量，懂的都懂

        im = Image.open(path).convert('RGB')        # 将子目录的图片转换为PIL库的Image类型（这个我们不用，我们直接有numpy）
        im = self.transform(im)                     # 使用get_transform方法，将图像直接转换为PyTorch Tensor
        im = np.array(im)                           # 这个操作很骚  
        lab = color.rgb2lab(im).astype(np.float32)  # 将RGB三通道转化为CIELAB颜色格式通道数
        lab_t = transforms.ToTensor()(lab)          # 将CIELAB格式的文件，格式抓话为PyTorch Tensor
        A = lab_t[[0], ...] / 50.0 - 1.0            
        B = lab_t[[1, 2], ...] / 110.0
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}   #封装入特定格式的Dict中去

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
