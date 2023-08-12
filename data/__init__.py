"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

#class MyDatasetDataloader():
#        def __init__(self, opt):
#        self.opt = opt
#        self.img_dir = os.path.join(opt.dataroot, 'images')
#        self.anno_dir = os.path.join(opt.dataroot, 'annotations')
#        self.img_names = sorted(os.listdir(self.img_dir))
#        self.anno_dict = {}
#        for img_name in self.img_names:
#            anno_path = os.path.join(self.anno_dir, os.path.splitext(img_name)[0] + '.txt')
#            with open(anno_path, 'r') as f:
#                lines = f.readlines()
#                boxes = []
#                for line in lines:
#                    box = list(map(int, line.strip().split(',')))
#                    boxes.append(box)
#            self.anno_dict[img_name] = boxes
#
#    def __len__(self):
#        return len(self.img_names)
#
#    def __getitem__(self, index):
#        img_path = os.path.join(self.img_dir, self.img_names[index])
#        img = cv2.imread(img_path)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        boxes = self.anno_dict[self.img_names[index]]
#        return {'img': img, 'boxes': boxes}
#
#def My_createdataset(opt):
#    """Create a dataset given the option.
#    This function wraps the class CustomDataset.
#    This is the main interface between this package and 'train.py'/'test.py'
#    Example:
#        >>> from data import create_dataset
#        >>> dataset = create_dataset(opt)
#    """
#    dataset = CustomDataset(opt)
#    return dataset
def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            drop_last=True,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
