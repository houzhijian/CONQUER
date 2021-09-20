"""
Data Loader
"""

import logging
import torch
from torch.utils.data import Dataset
import math
import os
import random
import lmdb
import io
import numpy as np
from utils.basic_utils import load_jsonl, l2_normalize_np_array, load_json
import msgpack
import msgpack_numpy
logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    """
    Args:
        dset_name, str, ["tvr"]
    Return:
        a dict: {
            "model_inputs": {
                "query"
                    "feat": torch.tensor, (max_desc_len, D_q)
                    "feat_mask": torch.tensor, (max_desc_len)
                    "feat_pos_id": torch.tensor, (max_desc_len)
                    "feat_token_id": torch.tensor, (max_desc_len)
                "visual"
                    "feat": torch.tensor, (max_ctx_len, D_video)
                    "feat_mask": torch.tensor, (max_ctx_len)
                    "feat_pos_id": torch.tensor, (max_ctx_len)
                    "feat_token_id": torch.tensor, (max_ctx_len)
                "sub" (optional)
                "st_ed_indices": torch.LongTensor, (2, )
            }
        }
    """
    def __init__(self, config, max_ctx_len=100, max_desc_len=30, clip_length=1.5,ctx_mode="visual_sub",
                 is_eval = False, mode = "train",
                 neg_video_num=3, data_ratio=1,
                 use_extend_pool=500, inference_top_k=10):


        self.dset_name = config.dset_name
        self.root_path = config.root_path

        self.desc_bert_path = os.path.join(self.root_path,config.desc_bert_path)
        self.vid_feat_path = os.path.join(self.root_path,config.vid_feat_path)

        self.ctx_mode = ctx_mode
        self.use_sub = "sub" in self.ctx_mode

        if self.use_sub:
            self.sub_bert_path = os.path.join(self.root_path, config.sub_bert_path)

        self.max_ctx_len = max_ctx_len
        self.max_desc_len = max_desc_len
        self.clip_length = clip_length

        self.neg_video_num = neg_video_num
        self.is_eval = is_eval

        self.mode = mode
        if mode == "train":
            self.data_path = os.path.join(self.root_path, config.train_data_path)
        elif mode == "val":
            self.data_path = os.path.join(self.root_path,config.eval_data_path)
        elif mode == "test_public":
            self.data_path = os.path.join(self.root_path,config.test_data_path)

        if mode == "train":
            self.first_VR_ranklist_path = os.path.join(self.root_path,config.train_first_VR_ranklist_path)
        elif mode == "val":
            self.first_VR_ranklist_path = os.path.join(self.root_path,config.eval_first_VR_ranklist_path_hero)
        elif mode == "test_public":
            self.first_VR_ranklist_path = os.path.join(self.root_path,config.test_public_first_VR_ranklist_path_hero)


        print(self.first_VR_ranklist_path)
        self.mode = mode

        # prepare desc data
        self.query_data = load_jsonl(self.data_path)

        self.first_VR_ranklist_pool_env = lmdb.open(self.first_VR_ranklist_path,
                                       readonly=True, create=False,
                                       max_readers=4096 * 8,
                                       readahead=False)

        self.first_VR_ranklist_pool_txn = self.first_VR_ranklist_pool_env.begin(buffers=True)

        self.data_ratio = data_ratio
        if self.data_ratio != 1:
            n_examples = int(len(self.query_data) * self.data_ratio)
            self.query_data = self.query_data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(self.data_ratio * 100, n_examples))


        self.desc_bert_env = lmdb.open(self.desc_bert_path,
                                       readonly=True, create=False,
                                       max_readers=4096 * 8,
                                       readahead=False)
        self.desc_bert_txn = self.desc_bert_env.begin(buffers=True)

        self.vid_feat_env = lmdb.open(self.vid_feat_path,
                                      readonly=True, create=False,
                                      max_readers=4096 * 8,
                                      readahead=False)
        self.vid_feat_txn = self.vid_feat_env.begin(buffers=True)

        if self.use_sub:
            self.sub_bert_env = lmdb.open(self.sub_bert_path,
                                          readonly=True, create=False,
                                          max_readers=4096 * 8,
                                          readahead=False)
            self.sub_bert_txn = self.sub_bert_env.begin(buffers=True)


        self.inference_top_k = inference_top_k
        video_data = load_json(os.path.join(self.root_path,config.video_duration_idx_path))[mode]

        self.video_data = [{"vid_name": k, "duration": v[0]} for k, v in video_data.items()]
        self.video2idx = {k: v[1] for k, v in video_data.items()}
        self.idx2video = {v[1]:k for k, v in video_data.items()}
        self.use_extend_pool = use_extend_pool

        self.normalize_vfeat = True
        self.normalize_tfeat = False

        self.visual_token_id = 0
        self.text_token_id = 1

    def __len__(self):
        return len(self.query_data)

    def pad_feature(self, feature, max_ctx_len):
        """
            Args:
                feature: original feature without padding
                max_ctx_len: the maximum length of video clips (or query token)

            Returns:
                 feat_pad : padded feature
                 feat_mask : feature mask
        """
        N_clip, feat_dim = feature.shape

        feat_pad = torch.zeros((max_ctx_len, feat_dim))
        feat_mask = torch.zeros(max_ctx_len, dtype=torch.long)
        feat_pad[:N_clip, :] = torch.from_numpy(feature)
        feat_mask[:N_clip] = 1

        return feat_pad , feat_mask

    def get_query_feat_by_desc_id(self, desc_id, token_id=1):
        """
            Args:
                desc_id: unique query description id
                token_id: specify modality embedding
            Returns:
                a dict for query: {
                    "feat": torch.tensor, (max_desc_len, D_q)
                    "feat_mask": torch.tensor, (max_desc_len)
                    "feat_pos_id": torch.tensor, (max_desc_len)
                    "feat_token_id": torch.tensor, (max_desc_len)
                }
        """
        dump = self.desc_bert_txn.get(str(desc_id).encode())
        with io.BytesIO(dump) as reader:
            feat_dump = np.load(reader, allow_pickle=True)
            query_feat = feat_dump['features'][:self.max_desc_len]

        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)

        feat_pad, feat_mask = \
            self.pad_feature(query_feat, self.max_desc_len)

        temp_model_inputs = dict()
        temp_model_inputs["feat"] = feat_pad
        temp_model_inputs["feat_mask"] = feat_mask
        temp_model_inputs["feat_pos_id"] = torch.arange(self.max_desc_len, dtype=torch.long)
        temp_model_inputs["feat_token_id"] = torch.full((self.max_desc_len,), token_id, dtype=torch.long)

        return temp_model_inputs

    def get_visual_feat_from_storage(self,vid_name):
        """
            Args:
                vid_name: unique video description id
            Returns:
                visual_feat: torch.tensor, (max_ctx_len, D_v)
                Use ResNet + SlowFast , D_v = 2048 + 2304 = 4352
        """
        dump = self.vid_feat_txn.get(vid_name.encode())
        img_dump = {k: np.copy(v) for k, v in msgpack_numpy.loads(dump, raw=False).items()}
        visual_feat = img_dump['features'][:self.max_ctx_len]

        if self.normalize_vfeat:
            visual_feat = l2_normalize_np_array(visual_feat)

        return visual_feat

    def get_sub_feat_from_storage(self,vid_name):
        """
            Args:
                vid_name: unique video description id
            Returns:
                visual_feat: torch.tensor, (max_ctx_len, D_s)
                Use RoBERTa, D_s =768
        """
        dump = self.sub_bert_txn.get(vid_name.encode())
        with io.BytesIO(dump) as reader:
            feat_dump = np.load(reader, allow_pickle=True)
            sub_feat = feat_dump["features"][:self.max_ctx_len]

        if self.normalize_tfeat:
            sub_feat = l2_normalize_np_array(sub_feat)

        return sub_feat

    def __getitem__(self, index):

        raw_data = self.query_data[index]

        # initialize with basic data
        meta = dict(
            desc_id=raw_data["desc_id"],
            desc=raw_data["desc"],
            vid_name=raw_data["vid_name"] if "vid_name" in raw_data else None,
            ts=raw_data["ts"] if "ts" in raw_data else None,
        )

        # If mode is test_public, no ground-truth video_id is provided. So use a fixed dummy ground-truth video_id
        if self.mode =="test_public":
            meta["vid_name"] = "castle_s01e01_seg02_clip_20"


        model_inputs = dict()
        ## query information
        model_inputs["query"] = self.get_query_feat_by_desc_id(meta["desc_id"],
                                                               token_id=self.text_token_id)

        ## get first stage VR search engine ranklist
        _external_inference_vr_res = msgpack.loads(self.first_VR_ranklist_pool_txn.get(str(meta["desc_id"]).encode()))


        if not self.is_eval:
            ##get the rank location of the ground-truth video for the first VR search engine
            location = 100
            for idx, item in enumerate(_external_inference_vr_res):
                if meta["vid_name"] == self.idx2video[item[0]]:
                    location = idx
                    break

            ##check all the location is below 100 when mode is train
            if self.mode =="train":
                assert  0<=location<100

            ##get the ranklist without the ground-truth video
            negative_video_pool_list = [self.idx2video[item[0]] for item in _external_inference_vr_res if meta["vid_name"] != self.idx2video[item[0]] ]

            ##sample neg_video_num negative videos for shared normalization
            sampled_negative_video_pool = random.sample(negative_video_pool_list[:location+self.use_extend_pool],
                                                            k=self.neg_video_num)
            ##the complete sampled video list , [pos, neg1, neg2, ...]
            total_vid_name_list = [meta["vid_name"],] + sampled_negative_video_pool

            self.shared_video_num = 1 + self.neg_video_num

        else:
            ##during eval, use top-k videos recommended by the first VR search engine
            inference_video_list = [ self.idx2video[item[0]] for item in _external_inference_vr_res[:self.inference_top_k]]
            inference_video_scores = [ item[1] for item in _external_inference_vr_res[:self.inference_top_k]]
            model_inputs["inference_vr_scores"] = torch.FloatTensor(inference_video_scores)
            total_vid_name_list = [meta["vid_name"],] + inference_video_list
            self.shared_video_num = 1 + self.inference_top_k

        # sampled neg_video_num negative videos or top-k videos
        meta["sample_vid_name_list"] = total_vid_name_list[1:]

        """ 
            a dict for visual modality: {
                "feat": torch.tensor, (shared_video_num, max_ctx_len, D_v)
                "feat_mask": torch.tensor, (shared_video_num, max_ctx_len)
                "feat_pos_id": torch.tensor, (shared_video_num, max_ctx_len)
                "feat_token_id": torch.tensor, (shared_video_num, max_ctx_len)
            }
        """
        groundtruth_visual_feat = self.get_visual_feat_from_storage(meta["vid_name"])
        ctx_l, feat_dim = groundtruth_visual_feat.shape

        visual_feat_pad = torch.zeros((self.shared_video_num, self.max_ctx_len, feat_dim))
        visual_feat_mask = torch.zeros((self.shared_video_num, self.max_ctx_len), dtype=torch.long)
        visual_feat_pos_id = \
            torch.repeat_interleave(torch.arange(self.max_ctx_len, dtype=torch.long).unsqueeze(0),
                                    self.shared_video_num, dim=0)
        visual_feat_token_id = torch.full((self.shared_video_num, self.max_ctx_len), self.visual_token_id,
                                          dtype=torch.long)

        for index, video_name in enumerate(total_vid_name_list,start=0):
            visual_feat = self.get_visual_feat_from_storage(video_name)

            feat_pad, feat_mask = \
                self.pad_feature(visual_feat, self.max_ctx_len)

            visual_feat_pad[index] = feat_pad
            visual_feat_mask[index] = feat_mask

        temp_model_inputs = dict()
        temp_model_inputs["feat"] = visual_feat_pad
        temp_model_inputs["feat_mask"] = visual_feat_mask
        temp_model_inputs["feat_pos_id"] = visual_feat_pos_id
        temp_model_inputs["feat_token_id"] = visual_feat_token_id

        model_inputs["visual"] = temp_model_inputs

        """ 
              a dict for sub modality: {
                  "feat": torch.tensor, (shared_video_num, max_ctx_len, D_t)
                  "feat_mask": torch.tensor, (shared_video_num, max_ctx_len)
                  "feat_pos_id": torch.tensor, (shared_video_num, max_ctx_len)
                  "feat_token_id": torch.tensor, (shared_video_num, max_ctx_len)
              }
        """
        if self.use_sub:
            groundtruth_sub_feat = self.get_sub_feat_from_storage(meta["vid_name"])

            _ , feat_dim = groundtruth_sub_feat.shape

            sub_feat_pad = torch.zeros((self.shared_video_num, self.max_ctx_len, feat_dim))
            sub_feat_mask = torch.zeros((self.shared_video_num, self.max_ctx_len), dtype=torch.long)
            sub_feat_pos_id = \
                torch.repeat_interleave(torch.arange(self.max_ctx_len, dtype=torch.long).unsqueeze(0),
                                        self.shared_video_num, dim=0)
            sub_feat_token_id = torch.full((self.shared_video_num, self.max_ctx_len), self.text_token_id, dtype=torch.long)

            for index, video_name in enumerate(total_vid_name_list, start=0):
                sub_feat = self.get_sub_feat_from_storage(video_name)

                feat_pad, feat_mask = \
                    self.pad_feature(sub_feat, self.max_ctx_len)

                sub_feat_pad[index] = feat_pad
                sub_feat_mask[index] = feat_mask

            temp_model_inputs = dict()
            temp_model_inputs["feat"] = sub_feat_pad
            temp_model_inputs["feat_mask"] = sub_feat_mask
            temp_model_inputs["feat_pos_id"] = sub_feat_pos_id
            temp_model_inputs["feat_token_id"] = sub_feat_token_id

            model_inputs["sub"] = temp_model_inputs

        if not self.is_eval:
            model_inputs["st_ed_indices"] = self.get_st_ed_label(meta["ts"],
                                                                 max_idx=ctx_l - 1)

        return dict(meta=meta, model_inputs=model_inputs)

    def get_st_ed_label(self, ts, max_idx):
        """
        Args:
            ts: [st (float), ed (float)] in seconds, ed > st
            max_idx: length of the video

        Returns:
            [st_idx, ed_idx]: int,
            ed_idx >= st_idx
            st_idx, ed_idx both belong to [0, max_idx-1]

        Given ts = [3.2, 7.6], st_idx = 2, ed_idx = 6,
        clips should be indexed as [2: 6), the translated back ts should be [3:9].
        # TODO which one is better, [2: 5] or [2: 6)
        """
        st_idx = min(math.floor(ts[0] / self.clip_length), max_idx)
        ed_idx = min(math.ceil(ts[1] / self.clip_length) - 1, max_idx)  # st_idx could be the same as ed_idx
        assert 0 <= st_idx <= ed_idx <= max_idx, (ts, st_idx, ed_idx, max_idx)
        return torch.LongTensor([st_idx, ed_idx])


