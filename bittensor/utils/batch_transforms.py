import torch


class ToTensor:
    """Applies the :class:`~torchvision.transforms.ToTensor` transform to a batch of images.
    """

    def __init__(self):
        self.max = 255

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be tensorized.
        Returns:
            Tensor: Tensorized Tensor.
        """
        return tensor.float().div_(self.max)


class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self,
                 mean,
                 std,
                 inplace=False,
                 dtype=torch.float,
                 device=None):
        
        if not device:
            device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.mean = torch.as_tensor(mean, dtype=dtype,
                                    device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :,
                                                                    None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class RandomHorizontalFlip:
    """Applies the :class:`~torchvision.transforms.RandomHorizontalFlip` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        p (float): probability of an image being flipped.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, p=0.5, inplace=False):
        self.p = p
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be flipped.
        Returns:
            Tensor: Randomly flipped Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        flipped = torch.rand(tensor.size(0)) < self.p
        tensor[flipped] = torch.flip(tensor[flipped], [3])
        return tensor


class RandomCrop:
    """Applies the :class:`~torchvision.transforms.RandomCrop` transform to a batch of images.
    Args:
        size (int): Desired output size of the crop.
        padding (int, optional): Optional padding on each border of the image. 
            Default is None, i.e no padding.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, size, padding=None, dtype=torch.float, device='cpu'):
        self.size = size
        self.padding = padding
        self.dtype = dtype
        self.device = device

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be cropped.
        Returns:
            Tensor: Randomly cropped Tensor.
        """
        if self.padding is not None:
            padded = torch.zeros(
                (tensor.size(0), tensor.size(1), tensor.size(2) +
                 self.padding * 2, tensor.size(3) + self.padding * 2),
                dtype=self.dtype,
                device=self.device)
            padded[:, :, self.padding:-self.padding,
                   self.padding:-self.padding] = tensor
        else:
            padded = tensor

        w, h = padded.size(2), padded.size(3)
        th, tw = self.size, self.size
        if w == tw and h == th:
            i, j = 0, 0
        else:
            i = torch.randint(0,
                              h - th + 1, (tensor.size(0),),
                              device=self.device)
            j = torch.randint(0,
                              w - tw + 1, (tensor.size(0),),
                              device=self.device)

        rows = torch.arange(th, dtype=torch.long, device=self.device) + i[:,
                                                                          None]
        columns = torch.arange(tw, dtype=torch.long,
                               device=self.device) + j[:, None]
        padded = padded.permute(1, 0, 2, 3)
        padded = padded[:,
                        torch.arange(tensor.size(0))[:, None, None],
                        rows[:, torch.arange(th)[:, None]], columns[:, None]]
        return padded.permute(1, 0, 2, 3)
