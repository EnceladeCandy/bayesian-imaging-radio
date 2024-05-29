import torch 
import numpy as np
from score_models import ScoreModel
import sys
sys.path.append("../src/inference")
from forward_model import complex_to_real, score_likelihood
from posterior_sampling import euler_sampler, pc_sampler

device = "cuda" if torch.cuda.is_available() else "cpu"



def main(args):
    ...



if __name__ == "__main__":
    ...