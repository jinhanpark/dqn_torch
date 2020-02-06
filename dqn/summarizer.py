from torch.utils.tensorboard import SummaryWriter


class Summarizer:
    
    def __init__(self):
        self.writer = SummaryWriter()
        
    def summarize_scalars(self, scalar_dict, n_iter):
        for key in scalar_dict:
            self.writer.add_scalar(key, scalar_dict[key], n_iter)