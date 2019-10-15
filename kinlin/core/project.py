from typing import List
from .utils import *
from .experiment import Experiment

class Project:
    def __init__(self, name: str = 'Untitled Project', notes: str = '', path: Union[Path, str] = None):

        # general properties
        self.id: str = generate_random_id()
        self.name: str = name
        self.fname: str = generate_fname(name)
        self.notes: str = notes
        self.path = Path(__file__).resolve() / self.fname
        create_dir_for(self.path, self.name)

        # experiments
        self.experiments: List[Experiment] = []

    def new_experiment(self, name: str):

        # new_exp = Experiment(project=self)
        # setup experiment?
        # self.experiments += new_exp
        pass

    def continue_experiment(self):
        pass

    def open_experiment_folder(self):
        open_folder(self.path)

