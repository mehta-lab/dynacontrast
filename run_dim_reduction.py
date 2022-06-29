import os
import argparse
from analysis.dim_reduction import dim_reduction
from utils.config_reader import YamlReader

def main(dataset_dirs, weights_dirs, label_cols, split, fraction):
    input_batch = []
    # if args.reduction:
    for weights_dir in weights_dirs:
        model_name = os.path.basename(weights_dir)
        embed_dirs = []
        for dataset_dir in dataset_dirs:
            embed_dir = os.path.join(dataset_dir, model_name)
            # embed_dir = os.path.join(dataset_dir, model_name + '_pool_norm')
            # embed_dir = dataset_dir
            embed_dirs.append(embed_dir)
        input_batch.append(embed_dirs)
    # method = 'pca'
    method = 'umap'
    fit_model = True
    # fit_model = False

    # conditions = [
    #               'CM mock', 'CM infected',
    #               'kidney tissue',
    #               'A549', 'A549', 'A549', 'A549',
    #               # 'A549 Mock 24h 60X', 'A549 Mock 48h 60X', 'A549 RSV 24h 60X', 'A549 RSV 48h 60X',
    #               #   'A549 MOCK IFNA 48 40X', 'A549 RSV IFNA 24 40X', 'A549 RSV IFNA 48 40X', 'A549 RSV IFNL 24 40X',
    #               # 'HEK Mock rep0',  'HEK Mock_rep1', 'HEK MOI0.25 rep0',  'HEK MOI0.25_rep1',  'HEK MOI1_rep0',  'HEK MOI1_rep1',  'HEK MOI2_rep0', 'HEK MOI2_rep1',
    #              ]
    # label_col = ['organelle-level ground truth']
    # label_col = 'protein-complex-level ground truth'

    for embed_dirs, weights_dir in zip(input_batch, weights_dirs):
        # try:
        dim_reduction(input_dirs=embed_dirs,
                      output_dirs=embed_dirs,
                      weights_dir=weights_dir,
                      method=method,
                      fit_model=fit_model,
                      label_cols=label_cols,
                      split=split,
                      fraction=fraction,
                      )
        # except Exception as ex:
        #     print('Dimension reduction failed for {}'.format(weights_dir), ex)
        #     continue
def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='path to yaml configuration file'
    )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)
    main(config.dim_reduction.raw_dirs,
         config.dim_reduction.weights_dirs,
         config.dim_reduction.label_cols,
         config.dim_reduction.split,
         config.dim_reduction.fraction)