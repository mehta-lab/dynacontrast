# bchhun, {2020-02-21}

from pipeline.patch_VAE import extract_patches, build_trajectories
from multiprocessing import Pool, Queue, Process
import os
from run_preproc import sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST
import numpy as np

# ESS from hulk

# SITES = ['B4-Site_0', 'B4-Site_1',  'B4-Site_2',  'B4-Site_3',  'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
#          'B5-Site_0', 'B5-Site_1',  'B5-Site_2',  'B5-Site_3',  'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
#          'C3-Site_0', 'C3-Site_1',  'C3-Site_2',  'C3-Site_3',  'C3-Site_4', 'C3-Site_5', 'C3-Site_6', 'C3-Site_7', 'C3-Site_8',
#          'C4-Site_0', 'C4-Site_1',  'C4-Site_2',  'C4-Site_3',  'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
#          'C5-Site_0', 'C5-Site_1',  'C5-Site_2',  'C5-Site_3',  'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8']

#SITES_ctrl = ['C5-Site_0', 'C5-Site_4']
#SITES_GBM = ['B2-Site_0', 'B2-Site_4']
#SITES_IL17 = ['B4-Site_0', 'B4-Site_4']
#SITES_IFbeta=['B5-Site_0', 'B5-Site_4']
#ITES_fast = ['C5-Site_0', 'C5-Site_4']

RAW_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
RAW_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/raw'
RAW_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'
INTERMEDIATE_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/supp'

#TARGET = '/gpfs/CompMicro/Projects/learningCellState/microglia/segmentation_experiments/expt_001'
TARGET = '/data_sm/home/michaelwu/VALIDATION'

class Worker(Process):
    def __init__(self, inputs, gpuid=0):
        super().__init__()
        self.gpuid=gpuid
        self.inputs=inputs

    def run(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)

        #extract_patches(self.inputs)
        build_trajectories(self.inputs)


def main():

    # loads 'Site.npy',
    #       '_NNProbabilities.npy',
    #       '/Site-supps/Site/cell_positions.pkl',
    #       '/Site-supps/site/cell_pixel_assignments.pkl',

    # generates 'stacks_%d.pkl' % timepoint

    # prints: "writing time %d"
    n_gpu = 1
    for sites, inputs, outputs in zip([sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST],
                            [RAW_NOVEMBER, RAW_JANUARY, RAW_JANUARY_FAST],
                            [INTERMEDIATE_NOVEMBER, INTERMEDIATE_JANUARY, INTERMEDIATE_JANUARY_FAST]):
        segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site)) and \
                                     os.path.exists(os.path.join(inputs, "%s_NNProbabilities.npy" % site))]
        sep = np.linspace(0, len(segment_sites), n_gpu+1).astype(int)
        
        process = []
        for i in range(n_gpu):
            _sites = segment_sites[sep[i]:sep[i+1]]
            args = (inputs, outputs, TARGET, _sites)
            p = Worker(args, gpuid=i)
            p.start()

    # *** NOT USED WITH VAE ***
    # *** USED IN POST-PCA TRAJ MATCHING ***
    # loads 'cell_positions.pkl', 'cell_pixel_assignments.pkl'
    # generates 'cell_traj.pkl'

if __name__ == '__main__':
    main()