import os
import time
import torch
import argparse
import sys
import pprint

import json
from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        self.parser.add_argument("--dset_name", type=str, default="tvr", choices=["tvr", "didemo"])
        self.parser.add_argument("--eval_split_name", type=str, default="val",
                                 help="should match keys in video_duration_idx_path, must set for VCMR")
        self.parser.add_argument("--data_ratio", type=float, default=1.0,
                                 help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                      "Use small portion for debug purposes. Note this is different from --debug, "
                                      "which works by breaking the loops, typically they are not used together.")
        self.parser.add_argument("--debug", action="store_true",
                                 help="debug (fast) mode, break all loops, do not load all data into memory.")
        self.parser.add_argument("--disable_eval", action="store_true",
                                 help="disable eval")
        self.parser.add_argument("--results_root", type=str, default="results")
        self.parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        self.parser.add_argument("--seed", type=int, default=2018, help="random seed")
        self.parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        self.parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU ids to run the job")
        self.parser.add_argument("--num_workers", type=int, default=8,
                                 help="num subprocesses used to load the data, 0: use main process")

        # training config
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_warmup_proportion", type=float, default=0.01,
                                 help="Proportion of training to perform linear learning rate warmup for. "
                                      "E.g., 0.1 = 10% of training.")
        self.parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
        self.parser.add_argument("--n_epoch", type=int, default=50, help="number of epochs to run")
        self.parser.add_argument("--max_es_cnt", type=int, default=3,
                                 help="number of epochs to early stop, use -1 to disable early stop")
        self.parser.add_argument("--stop_task", type=str, default="VCMR", choices=["VCMR", "SVMR", "VR"],
                                 help="Use metric associated with stop_task for early stop")
        self.parser.add_argument("--eval_tasks_at_training", type=str, nargs="+",
                                 default=["VCMR", "SVMR", "VR"], choices=["VCMR", "SVMR", "VR"],
                                 help="evaluate and report  numbers for tasks specified here.")
        self.parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        self.parser.add_argument("--eval_query_bsz", type=int, default=8,
                                 help="mini-batch size at inference, for query")
        self.parser.add_argument("--no_eval_untrained", action="store_true", help="Evaluate on un-trained model")
        self.parser.add_argument("--grad_clip", type=float, default=-1, help="perform gradient clip, -1: disable")
        self.parser.add_argument("--eval_epoch_num", type=int, default=1, help="eval_epoch_num")

        # Data config
        self.parser.add_argument("--max_ctx_len", type=int, default=100,
                                 help="max number of snippets, 100 for tvr clip_length=1.5, only 109/21825 > 100")
        self.parser.add_argument("--max_desc_len", type=int, default=30, help="max number of query token")
        self.parser.add_argument("--clip_length", type=float, default=1.5,
                                 help="each video will be uniformly segmented into small clips")
        self.parser.add_argument("--ctx_mode", type=str, default="visual_sub",
                                 help="adopted modality list for each clip")
        self.parser.add_argument("--dataset_config", type=str,help="data config")


        # Model config

        self.parser.add_argument("--visual_dim", type=int,default=4352,help="visual modality feature dimension")
        self.parser.add_argument("--text_dim", type=int, default=768, help="textual modality feature dimension")
        self.parser.add_argument("--query_dim", type=int, default=768, help="query feature dimension")
        self.parser.add_argument("--hidden_dim", type=int, default=768, help="joint dimension")
        self.parser.add_argument("--no_output_moe_weight",action="store_true",
                                 help="whether NOT to use query dependent fusion")
        self.parser.add_argument("--model_config", type=str, help="model config")


        ## Train config
        self.parser.add_argument("--lw_st_ed", type=float, default=0.01, help="weight for moment cross-entropy loss")
        self.parser.add_argument("--lw_video_ce", type=float, default=0.05, help="weight for video cross-entropy loss")
        self.parser.add_argument("--lr_mul", type=float, default=1, help="Learning rate multiplier for backbone module")
        self.parser.add_argument("--use_extend_pool", type=int, default=1000,
                                 help="use_extend_pool")
        self.parser.add_argument("--neg_video_num",type=int,default=3,
                                 help="sample the number of negative video, "
                                      "if neg_video_num=0, then disable shared normalization training objective")
        self.parser.add_argument("--encoder_pretrain_ckpt_filepath", type=str,
                                 default="None",
                                 help="first_stage_pretrain checkpoint")
        self.parser.add_argument("--use_interal_vr_scores", action="store_true",
                                 help="whether to interal_vr_scores, true only for general similarity measure function")

        ## Eval config
        self.parser.add_argument("--similarity_measure",
                                 type=str, choices=["general", "exclusive","disjoint"],
                                 default="general",help="similarity_measure_function")

        # post processing
        self.parser.add_argument("--min_pred_l", type=int, default=0,
                                 help="constrain the [st, ed] with ed - st >= 1"
                                      "(1 clips with length 1.5 each, 1.5 secs in total"
                                      "this is the min length for proposal-based method)")
        self.parser.add_argument("--max_pred_l", type=int, default=24,
                                 help="constrain the [st, ed] pairs with ed - st <= 24, 36 secs in total"
                                      "(24 clips with length 1.5 each, "
                                      "this is the max length for proposal-based method)")
        self.parser.add_argument("--max_before_nms", type=int, default=200)
        self.parser.add_argument("--max_vcmr_video", type=int, default=10,
                                 help="ranking in top-max_vcmr_video")
        self.parser.add_argument("--nms_thd", type=float, default=-1,
                                 help="additionally use non-maximum suppression "
                                      "(or non-minimum suppression for distance)"
                                      "to post-processing the predictions. "
                                      "-1: do not use nms. 0.7 for tvr")

        # can use config files
        self.parser.add_argument('--config', help='JSON config files')


    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        # print("------------ Options -------------\n{}\n-------------------"
        #       .format({str(k): str(v) for k, v in sorted(args.items())}))
        print("------------ Options -------------\n{}\n-------------------"
              .format(pprint.pformat({str(k): str(v) for k, v in sorted(args.items())}, indent=4)))


        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)


    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = parse_with_config(self.parser)

        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            #opt.disable_eval = True

        if isinstance(self, TestOptions):

            # modify model_dir to absolute path
            opt.model_dir = os.path.join("results", opt.model_dir)

            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "nms_thd", "debug", "dataset_config", "model_config","device",
                               "eval_split_name", "eval_query_bsz", "eval_context_bsz", "device_ids",
                               "max_vcmr_video","max_pred_l", "min_pred_l", "external_inference_vr_res_path"]:
                    setattr(opt, arg, saved_options[arg])
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([opt.dset_name, opt.exp_id,
                                                     time.strftime("%Y_%m_%d_%H_%M_%S")]))
            mkdirp(opt.results_dir)
            # save a copy of current code
            code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["condor","data","results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"],)

        self.display_save(opt)


        assert opt.stop_task in opt.eval_tasks_at_training
        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda:%d" % opt.device_ids[0] if opt.device >= 0 else "cpu")

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""
    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")
        self.parser.add_argument("--tasks", type=str, nargs="+",
                                 choices=["VCMR", "SVMR", "VR"], default=["VCMR", "SVMR", "VR"],
                                 help="Which tasks to run."
                                      "VCMR: Video Corpus Moment Retrieval;"
                                      "SVMR: Single Video Moment Retrieval;"
                                      "VR: regular Video Retrieval. (will be performed automatically with VCMR)")

if __name__ == '__main__':
    print(__file__)
    print(os.path.realpath(__file__))
    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(code_dir)