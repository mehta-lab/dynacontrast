import yaml
import logging

# to add a new configuration parameter, simply add the string to the appropriate set here

PREPROCESS = {
    "image_dirs",
    "target_dirs",
    "channels",
    "fov",
    "pos_dir",
    "multipage",
    "z_slice",
}

PATCH = {
    "raw_dirs",
    "supp_dirs",
    "channels",
    "fov",
    "num_cpus",
    "crop_size",
    "save_fig",
    "reload",
    "skip_boundary",
    "min_length",
    "track_dim",
}

INFERENCE = {
    "raw_dirs",
    "supp_dirs",
    "val_dirs",
    "network",
    "network_width",
    "weights",
    "save_output",
    "gpu_ids",
    "fov",
    "channels",
    "channel_mean",
    "channel_std",
    "num_classes",
    "crop_size",
    "batch_size",
    "num_pred_rnd",
    "seg_val_cat",
    "num_workers",
    "normalization",
    "model",
}

SEGMENTATION = {
    "raw_dirs",
    "supp_dirs",
    "val_dirs",
    "model",
    "weights",
    "save_output",
    "gpu_ids",
    "num_workers",
    "fov",
    "channels",
    "channel_mean",
    "channel_std",
    "num_classes",
    "crop_size",
    "batch_size",
    "num_pred_rnd",
    "seg_val_cat",
}

DIM_REDUCTION = {
    "input_dirs",
    "output_dirs",
    "file_name_prefixes",
    "weights_dirs",
    "fit_model",
    "conditions",
}

TRAINING = {
    "raw_dirs",
    "supp_dirs",
    "weights_dirs",
    "network",
    "network_width",
    "num_inputs",
    "num_hiddens",
    "num_residual_hiddens",
    "num_residual_layers",
    "num_embeddings",
    "weight_matching",
    "margin",
    "w_a",
    "w_t",
    "w_n",
    "channels",
    "channel_mean",
    "channel_std",
    "commitment_cost",
    "n_epochs",
    "learn_rate",
    "batch_size",
    "val_split_ratio",
    "shuffle_data",
    "transform",
    "patience",
    "n_pos_samples",
    "num_workers",
    "gpu_id",
    "start_model_path",
    "retrain",
    "start_epoch",
    "earlystop_metric",
    "model_name",
    "use_mask",
    "normalization",
    "loss",
    "temperature",
    "augmentations",
}

##TODO: add checks for config fields
class Object:
    pass


class Struct(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)([self._wrap(v) for v in value])
        else:
            return Struct(value) if isinstance(value, dict) else value


class YamlReader(Struct):
    def __init__(self):
        self.config = None

        # easy way to assign attributes to each category
        # self.files = Object()
        # self.preprocess = Object()
        # self.patch = Object()
        # self.inference = Object()
        # self.segmentation = Object()
        # self.dim_reduction = Object()
        # self.training = Object()

    def read_config(self, yml_config):
        with open(yml_config, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            super(YamlReader, self).__init__(self.config)
            # # self._parse_files()
            # self._parse_preprocessing()
            # self._parse_patch()
            # self._parse_inference()
            # self._parse_segmentation()
            # self._parse_dim_reduction()
            # self._parse_training()

    # def _parse_preprocessing(self):
    #     for key, value in self.config['preprocess'].items():
    #         if key in PREPROCESS:
    #             setattr(self.preprocess, key, value)
    #         else:
    #             log.warning(f"yaml PREPROCESS config field {key} is not recognized")
    #
    # def _parse_patch(self):
    #     for key, value in self.config['patch'].items():
    #         if key in PATCH:
    #             setattr(self.patch, key, value)
    #         else:
    #             log.warning(f"yaml PATCH config field {key} is not recognized")
    #
    # def _parse_inference(self):
    #     for key, value in self.config['inference'].items():
    #         if key in INFERENCE:
    #             setattr(self.inference, key, value)
    #         else:
    #             log.warning(f"yaml INFERENCE config field {key} is not recognized")
    #
    # def _parse_segmentation(self):
    #     for key, value in self.config['segmentation'].items():
    #         if key in SEGMENTATION:
    #             setattr(self.segmentation, key, value)
    #         else:
    #             log.warning(f"yaml SEGMENTATION config field {key} is not recognized")
    #
    # def _parse_dim_reduction(self):
    #     for key, value in self.config['dim_reduction'].items():
    #         if key in DIM_REDUCTION:
    #             setattr(self.dim_reduction, key, value)
    #         else:
    #             log.warning(f"yaml DIM REDUCTION config field {key} is not recognized")
    #
    # def _parse_training(self):
    #     for key, value in self.config['training'].items():
    #         if key in TRAINING:
    #             setattr(self.training, key, value)
    #             if isinstance(value, dict):
    #                 for key_sub, value_sub in value.items():
    #                     setattr(self.training, key, value)
    #
    #
    #         else:
    #             log.warning(f"yaml TRAINING config field {key} is not recognized")
