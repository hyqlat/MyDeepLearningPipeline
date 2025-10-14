import argparse

class MyOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        self.parser.add_argument('--dataset_type', type=str, default='RegressionDataset')
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--valid_batch_size', type=int, default=1)
        self.parser.add_argument('--test_batch_size', type=int, default=1)
        self.parser.add_argument('--model_type', type=str, default='RegressionModel')
        self.parser.add_argument('--loss_type', type=str, default='CE', choices=['MSE', 'CE'])
        self.parser.add_argument('--lr', type=float, default=0.01)
        self.parser.add_argument('--total_epoch', type=int, default=50)
        self.parser.add_argument('--has_valid', action='store_true')
        self.parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
        self.parser.add_argument('--is_load', action='store_true')
        self.parser.add_argument('--load_filename', type=str, default='latest')
        self.parser.add_argument('--result_dir', type=str, default='results')

        self.parser.add_argument('--classname_path', type=str, default='../datasets/fruits100/train/classname.txt')
   
   
    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        return self.opt
