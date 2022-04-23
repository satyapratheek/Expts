import errno 
import os 
import shutil
import sys
from venv import create

def to_np(var):
    """Exports torch.Tensor to numpy array
    """
    return var.detach().cpu().numpy()

def create_folder(folder_path):
    """Creates a folder if it does not exist
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise

def clear_folder(folder_path):
    """Clear all contents recursively if the folder exists
       and create the folder if it has been accidentally deleted
    """
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)

class StdOut(object):
    """Redirects stdout to file, and prints to console."""
    
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush ()