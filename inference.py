import os
import pprint
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config.config import TestOptions
from model.conquer import CONQUER
from data_loader.second_stage_start_end_dataset import StartEndDataset as StartEndEvalDataset
from utils.inference_utils  import \
    get_submission_top_n, post_processing_vcmr_nms
from utils.basic_utils import save_json , load_config
from utils.tensor_utils import find_max_triples_from_upper_triangle_product
from standalone_eval.eval import eval_retrieval
from utils.model_utils import move_cuda , start_end_collate
from utils.model_utils import VERY_NEGATIVE_NUMBER
import logging
from time import time

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def generate_min_max_length_mask(array_shape, min_l, max_l):
    """ The last two dimension denotes matrix of upper-triangle with upper-right corner masked,
    below is the case for 4x4.
    [[0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]

    Args:
        array_shape: np.shape??? The last two dimensions should be the same
        min_l: int, minimum length of predicted span
        max_l: int, maximum length of predicted span

    Returns:

    """
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape, dtype=np.float32)  # (1, ..., 1, L, L)
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask  # with valid bit to be 1


def get_svmr_res_from_st_ed_probs_disjoint(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx,
                                  clip_length, min_pred_l, max_pred_l, max_before_nms):
    """
    Args:
        svmr_gt_st_probs: np.ndarray (N_queries, L, L), value range [0, 1]
        svmr_gt_ed_probs:
        query_metas:
        video2idx:
        clip_length: float, how long each clip is in seconds
        min_pred_l: int, minimum number of clips
        max_pred_l: int, maximum number of clips
        max_before_nms: get top-max_before_nms predictions for each query

    Returns:

    """
    svmr_res = []
    query_vid_names = [e["vid_name"] for e in query_metas]

    # masking very long ones! Since most are relatively short.
    # disjoint : b_i + e_i
    _st_ed_scores = np.expand_dims(svmr_gt_st_probs,axis=2) + np.expand_dims(svmr_gt_ed_probs,axis=1)

    _N_q = _st_ed_scores.shape[0]

    _valid_prob_mask = np.logical_not(generate_min_max_length_mask(
        _st_ed_scores.shape, min_l=min_pred_l, max_l=max_pred_l).astype(bool))

    valid_prob_mask = np.tile(_valid_prob_mask,(_N_q, 1, 1))

    # invalid location will become VERY_NEGATIVE_NUMBER!
    _st_ed_scores[valid_prob_mask] = VERY_NEGATIVE_NUMBER

    batched_sorted_triples = find_max_triples_from_upper_triangle_product(
        _st_ed_scores, top_n=max_before_nms, prob_thd=None)
    for i, q_vid_name in tqdm(enumerate(query_vid_names),
                              desc="[SVMR] Loop over queries to generate predictions",
                              total=len(query_vid_names)):  # i is query_id
        q_m = query_metas[i]
        video_idx = video2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        _sorted_triples[:, 1] += 1  # as we redefined ed_idx, which is inside the moment.
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * clip_length
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = [[video_idx, ] + row for row in _sorted_triples.tolist()]
        cur_query_pred = dict(
            desc_id=q_m["desc_id"],
            desc=q_m["desc"],
            predictions=cur_ranked_predictions
        )
        svmr_res.append(cur_query_pred)
    return svmr_res


def get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs, query_metas, video2idx,
                                  clip_length, min_pred_l, max_pred_l, max_before_nms):
    """
    Args:
        svmr_gt_st_probs: np.ndarray (N_queries, L, L), value range [0, 1]
        svmr_gt_ed_probs:
        query_metas:
        video2idx:
        clip_length: float, how long each clip is in seconds
        min_pred_l: int, minimum number of clips
        max_pred_l: int, maximum number of clips
        max_before_nms: get top-max_before_nms predictions for each query

    Returns:

    """
    svmr_res = []
    query_vid_names = [e["vid_name"] for e in query_metas]

    # masking very long ones! Since most are relatively short.
    # general/exclusive :  \hat{b_i} * \hat{e_i}
    st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs, svmr_gt_ed_probs)  # (N, L, L)

    valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape, min_l=min_pred_l, max_l=max_pred_l)
    st_ed_prob_product *= valid_prob_mask  # invalid location will become zero!

    batched_sorted_triples = find_max_triples_from_upper_triangle_product(
        st_ed_prob_product, top_n=max_before_nms, prob_thd=None)
    for i, q_vid_name in tqdm(enumerate(query_vid_names),
                              desc="[SVMR] Loop over queries to generate predictions",
                              total=len(query_vid_names)):  # i is query_id
        q_m = query_metas[i]
        video_idx = video2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        _sorted_triples[:, 1] += 1  # as we redefined ed_idx, which is inside the moment.
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * clip_length
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = [[video_idx, ] + row for row in _sorted_triples.tolist()]
        cur_query_pred = dict(
            desc_id=q_m["desc_id"],
            desc=q_m["desc"],
            predictions=cur_ranked_predictions
        )
        svmr_res.append(cur_query_pred)
    return svmr_res



def compute_query2ctx_info(model, eval_dataset, opt,
                           max_before_nms=200, max_n_videos=100, tasks=("SVMR",)):
    """
    Use val set to do evaluation, remember to run with torch.no_grad().
     model : CONQUER
     eval_dataset :
     opt :
     max_before_nms : max moment number before non-maximum suppression
     tasks: evaluation tasks

     general/exclusive function : r * \hat{b_i} + \hat{e_i}
    """
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks
    is_svmr = "SVMR" in tasks

    video2idx = eval_dataset.video2idx

    model.eval()
    query_eval_loader = DataLoader(eval_dataset,
                                   collate_fn= start_end_collate,
                                   batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=True)

    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=np.int)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)

    if is_vr :
        if opt.use_interal_vr_scores:
            sorted_q2c_indices = np.tile(np.arange(max_n_videos, dtype=np.int),n_total_query).reshape(n_total_query,max_n_videos)
            sorted_q2c_scores = np.empty((n_total_query, max_n_videos), dtype=np.float32)
        else:
            sorted_q2c_indices = np.empty((n_total_query, max_n_videos), dtype=np.int)
            sorted_q2c_scores = np.empty((n_total_query, max_n_videos), dtype=np.float32)

    if is_svmr:
        svmr_gt_st_probs = np.zeros((n_total_query, opt.max_ctx_len), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, opt.max_ctx_len), dtype=np.float32)

    query_metas = []
    for idx, batch in tqdm(
            enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

        _query_metas = batch["meta"]
        query_metas.extend(batch["meta"])

        if opt.device.type == "cuda":
            model_inputs = move_cuda(batch["model_inputs"], opt.device)
        else:
            model_inputs = batch["model_inputs"]


        video_similarity_score, begin_score_distribution, end_score_distribution = \
            model.get_pred_from_raw_query(model_inputs)

        if is_svmr:
            _svmr_st_probs = begin_score_distribution[:, 0]
            _svmr_ed_probs = end_score_distribution[:, 0]

            # normalize to get true probabilities!!!
            # the probabilities here are already (pad) masked, so only need to do softmax
            _svmr_st_probs = F.softmax(_svmr_st_probs, dim=-1)  # (_N_q, L)
            _svmr_ed_probs = F.softmax(_svmr_ed_probs, dim=-1)
            if opt.debug:
                print("svmr_st_probs: ", _svmr_st_probs)

            svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz] = \
                _svmr_st_probs.cpu().numpy()

            svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz] = \
                _svmr_ed_probs.cpu().numpy()

        _vcmr_st_prob = begin_score_distribution[:, 1:]
        _vcmr_ed_prob = end_score_distribution[:, 1:]

        if not (is_vr or is_vcmr):
            continue

        if opt.use_interal_vr_scores:
            bs = begin_score_distribution.size()[0]
            _sorted_q2c_indices = torch.arange(max_n_videos).to(begin_score_distribution.device).repeat(bs,1)
            _sorted_q2c_scores = model_inputs["inference_vr_scores"]
            if is_vr:
                sorted_q2c_scores[idx * bsz:(idx + 1) * bsz] = model_inputs["inference_vr_scores"].cpu().numpy()
        else:
            video_similarity_score = video_similarity_score[:, 1:]

            _query_context_scores = torch.softmax(video_similarity_score,dim=1)

            # Get top-max_n_videos videos for each query
            _sorted_q2c_scores, _sorted_q2c_indices = \
                torch.topk(_query_context_scores, max_n_videos, dim=1, largest=True)
            if is_vr:
                sorted_q2c_indices[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_indices.cpu().numpy()
                sorted_q2c_scores[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_scores.cpu().numpy()


        if not is_vcmr:
            continue


        # normalize to get true probabilities!!!
        # the probabilities here are already (pad) masked, so only need to do softmax
        _st_probs = F.softmax(_vcmr_st_prob, dim=-1)  # (_N_q, N_videos, L)
        _ed_probs = F.softmax(_vcmr_ed_prob, dim=-1)


        # Get VCMR results
        # compute combined scores
        row_indices = torch.arange(0, len(_st_probs), device=opt.device).unsqueeze(1)
        _st_probs = _st_probs[row_indices, _sorted_q2c_indices]  # (_N_q, max_n_videos, L)
        _ed_probs = _ed_probs[row_indices, _sorted_q2c_indices]

        # (_N_q, max_n_videos, L, L)
        # general/exclusive :  r * \hat{b_i} * \hat{e_i}
        _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs, _sorted_q2c_scores, _ed_probs)

        valid_prob_mask = generate_min_max_length_mask(
            _st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)

        _st_ed_scores *= torch.from_numpy(
            valid_prob_mask).to(_st_ed_scores.device)  # invalid location will become zero!

        _n_q  = _st_ed_scores.shape[0]

        # sort across the total_n_videos videos (by flatten from the 2nd dim)
        # the indices here are local indices, not global indices

        _flat_st_ed_scores = _st_ed_scores.reshape(_n_q, -1)  # (N_q, total_n_videos*L*L)
        _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = \
            torch.sort(_flat_st_ed_scores, dim=1, descending=True)

        # collect data
        flat_st_ed_sorted_scores[idx * bsz:(idx + 1) * bsz] = \
            _flat_st_ed_sorted_scores[:, :max_before_nms].cpu().numpy()
        flat_st_ed_scores_sorted_indices[idx * bsz:(idx + 1) * bsz] = \
            _flat_st_ed_scores_sorted_indices[:, :max_before_nms].cpu().numpy()

        if opt.debug:
            break

    # Numpy starts here!!!
    vr_res = []
    if is_vr:
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(zip(sorted_q2c_scores, sorted_q2c_indices)),
                desc="[VR] Loop over queries to generate predictions", total=n_total_query):
            cur_vr_redictions = []
            query_specific_video_metas = query_metas[i]["sample_vid_name_list"]
            for j, (v_score, v_meta_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = video2idx[query_specific_video_metas[v_meta_idx]]
                cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
            cur_query_pred = dict(
                desc_id=query_metas[i]["desc_id"],
                desc=query_metas[i]["desc"],
                predictions=cur_vr_redictions
            )
            vr_res.append(cur_query_pred)

    svmr_res = []
    if is_svmr:
        svmr_res = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs,
                                                 query_metas, video2idx,
                                                 clip_length=opt.clip_length,
                                                 min_pred_l=opt.min_pred_l,
                                                 max_pred_l=opt.max_pred_l,
                                                 max_before_nms=max_before_nms)


    vcmr_res = []
    if is_vcmr:
        for i, (_flat_st_ed_scores_sorted_indices, _flat_st_ed_sorted_scores) in tqdm(
                enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),
                desc="[VCMR] Loop over queries to generate predictions", total=n_total_query):  # i is query_idx
            # list([video_idx(int), st(float), ed(float), score(float)])
            video_meta_indices_local, pred_st_indices, pred_ed_indices = \
                np.unravel_index(_flat_st_ed_scores_sorted_indices,
                                 shape=(max_n_videos, opt.max_ctx_len, opt.max_ctx_len))
            # video_meta_indices refers to the indices among the total_n_videos
            # video_meta_indices_local refers to the indices among the top-max_n_videos
            # video_meta_indices refers to the indices in all the videos, which is the True indices
            video_meta_indices = sorted_q2c_indices[i, video_meta_indices_local]

            pred_st_in_seconds = pred_st_indices.astype(np.float32) * opt.clip_length
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * opt.clip_length + opt.clip_length
            cur_vcmr_redictions = []
            query_specific_video_metas = query_metas[i]["sample_vid_name_list"]
            for j, (v_meta_idx, v_score) in enumerate(zip(video_meta_indices, _flat_st_ed_sorted_scores)):  # videos
                video_idx = video2idx[query_specific_video_metas[v_meta_idx]]
                cur_vcmr_redictions.append(
                    [video_idx, float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j]), float(v_score)])

            cur_query_pred = dict(
                desc_id=query_metas[i]["desc_id"],
                desc=query_metas[i]["desc"],
                predictions=cur_vcmr_redictions)
            vcmr_res.append(cur_query_pred)

    res = dict(VCMR=vcmr_res, SVMR=svmr_res, VR=vr_res)
    return {k: v for k, v in res.items() if len(v) != 0}


def compute_query2ctx_info_disjoint(model, eval_dataset, opt,
                           max_before_nms=200, max_n_videos=100, tasks=("VCMR","SVMR","VR")):
    """Use val set to do evaluation, remember to run with torch.no_grad().
     model : CONQUER
     eval_dataset :
     opt :
     max_before_nms : max moment number before non-maximum suppression
     tasks: evaluation tasks

     disjoint function : b_i + e_i

    """
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks
    is_svmr = "SVMR" in tasks

    video2idx = eval_dataset.video2idx

    model.eval()
    query_eval_loader = DataLoader(eval_dataset,
                                   collate_fn= start_end_collate,
                                   batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=True)

    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty((n_total_query, max_before_nms), dtype=np.int)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms), dtype=np.float32)

    if is_vr :
        sorted_q2c_indices = np.empty((n_total_query, max_n_videos), dtype=np.int)
        sorted_q2c_scores = np.empty((n_total_query, max_n_videos), dtype=np.float32)

    if is_svmr:
        svmr_gt_st_probs = np.zeros((n_total_query, opt.max_ctx_len), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, opt.max_ctx_len), dtype=np.float32)

    query_metas = []
    for idx, batch in tqdm(
            enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):

        _query_metas = batch["meta"]
        query_metas.extend(batch["meta"])

        if opt.device.type == "cuda":
            model_inputs = move_cuda(batch["model_inputs"], opt.device)

        else:
            model_inputs = batch["model_inputs"]

        _ , begin_score_distribution, end_score_distribution = \
            model.get_pred_from_raw_query(model_inputs)

        if is_svmr:
            _svmr_st_probs = begin_score_distribution[:,0]
            _svmr_ed_probs = end_score_distribution[:,0]

            # Do not normalize to get true probabilities!!!
            # The normalized scores are not comparable when not being prior on the video similarity score

            if opt.debug:
                print("svmr_st_probs: ", _svmr_st_probs)

            svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz] = \
                _svmr_st_probs.cpu().numpy()

            svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz] = \
                _svmr_ed_probs.cpu().numpy()

        if not (is_vr or is_vcmr):
            continue

        begin_score_distribution = begin_score_distribution[:,1:]
        end_score_distribution= end_score_distribution[:,1:]

        if opt.debug:
            print("begin_score_distribution: ", begin_score_distribution, begin_score_distribution.shape)

        # Get VCMR results
        # (_N_q, total_n_videos, L, L)
        # b_i + e_i
        _st_ed_scores = torch.unsqueeze(begin_score_distribution, 3) + torch.unsqueeze(end_score_distribution, 2)

        _n_q, total_n_videos = _st_ed_scores.size()[:2]

        if opt.debug:
            print("_st_ed_scores: ",_st_ed_scores)

        ## mask the invalid location out of moment length constrain
        _valid_prob_mask = np.logical_not(generate_min_max_length_mask(
            _st_ed_scores.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l).astype(bool))

        _valid_prob_mask = torch.from_numpy(_valid_prob_mask).to(_st_ed_scores.device)

        valid_prob_mask = _valid_prob_mask.repeat(_n_q,total_n_videos,1,1)

        # invalid location will become VERY_NEGATIVE_NUMBER!
        _st_ed_scores[valid_prob_mask] = VERY_NEGATIVE_NUMBER

        if opt.debug:
            print("_st_ed_scores: ",_st_ed_scores)

        if is_vr:
            # get video-level retrieval scores
            # pick the maximum across all possible start and end pair
            # (by flatten from the 3,4nd dim)
            _flat_video_scores = _st_ed_scores.reshape(_n_q, total_n_videos, -1)  # (N_q, max_n_videos,L*L)
            _query_context_scores, _ = torch.max(_flat_video_scores, dim=-1)

            # Get top-max_n_videos videos for each query
            _sorted_q2c_scores, _sorted_q2c_indices = \
                torch.topk(_query_context_scores, max_n_videos, dim=1, largest=True)

            sorted_q2c_indices[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_indices.cpu().numpy()
            sorted_q2c_scores[idx * bsz:(idx + 1) * bsz] = _sorted_q2c_scores.cpu().numpy()

        # sort across the total_n_videos videos (by flatten from the 2nd dim)
        # the indices here are local indices, not global indices
        _flat_st_ed_scores = _st_ed_scores.reshape(_n_q, -1)  # (N_q, total_n_videos*L*L)
        _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = \
            torch.sort(_flat_st_ed_scores, dim=1, descending=True)

        # collect data
        flat_st_ed_sorted_scores[idx * bsz:(idx + 1) * bsz] = \
            _flat_st_ed_sorted_scores[:, :max_before_nms].cpu().numpy()
        flat_st_ed_scores_sorted_indices[idx * bsz:(idx + 1) * bsz] = \
            _flat_st_ed_scores_sorted_indices[:, :max_before_nms].cpu().numpy()

        if opt.debug:
            exit(1)
    # Numpy starts here!!!

    vr_res = []
    if is_vr:
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(zip(sorted_q2c_scores, sorted_q2c_indices)),
                desc="[VR] Loop over queries to generate predictions", total=n_total_query):
            cur_vr_redictions = []
            query_specific_video_metas = query_metas[i]["sample_vid_name_list"]
            for j, (v_score, v_meta_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = video2idx[query_specific_video_metas[v_meta_idx]]
                cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
            cur_query_pred = dict(
                desc_id=query_metas[i]["desc_id"],
                desc=query_metas[i]["desc"],
                predictions=cur_vr_redictions
            )
            vr_res.append(cur_query_pred)

    svmr_res = []
    if is_svmr:
        svmr_res = get_svmr_res_from_st_ed_probs_disjoint(svmr_gt_st_probs, svmr_gt_ed_probs,
                                                 query_metas, video2idx,
                                                 clip_length=opt.clip_length,
                                                 min_pred_l=opt.min_pred_l,
                                                 max_pred_l=opt.max_pred_l,
                                                 max_before_nms=max_before_nms)

    vcmr_res = []
    if is_vcmr:
        for i, (_flat_st_ed_scores_sorted_indices, _flat_st_ed_sorted_scores) in tqdm(
                enumerate(zip(flat_st_ed_scores_sorted_indices, flat_st_ed_sorted_scores)),
                desc="[VCMR] Loop over queries to generate predictions", total=n_total_query):  # i is query_idx
            # list([video_idx(int), st(float), ed(float), score(float)])
            video_meta_indices_local, pred_st_indices, pred_ed_indices = \
                np.unravel_index(_flat_st_ed_scores_sorted_indices,
                                 shape=(total_n_videos, opt.max_ctx_len, opt.max_ctx_len))
            # video_meta_indices refers to the indices among the total_n_videos
            # video_meta_indices_local refers to the indices among the top-max_n_videos
            # video_meta_indices refers to the indices in all the videos, which is the True indices
            # video_meta_indices = sorted_q2c_indices[i, video_meta_indices_local]

            pred_st_in_seconds = pred_st_indices.astype(np.float32) * opt.clip_length
            pred_ed_in_seconds = pred_ed_indices.astype(np.float32) * opt.clip_length + opt.clip_length
            cur_vcmr_redictions = []
            query_specific_video_metas = query_metas[i]["sample_vid_name_list"]
            for j, (v_meta_idx, v_score) in enumerate(zip(video_meta_indices_local, _flat_st_ed_sorted_scores)):  # videos
                video_idx = video2idx[query_specific_video_metas[v_meta_idx]]
                cur_vcmr_redictions.append(
                    [video_idx, float(pred_st_in_seconds[j]), float(pred_ed_in_seconds[j]), float(v_score)])

            cur_query_pred = dict(
                desc_id=query_metas[i]["desc_id"],
                desc=query_metas[i]["desc"],
                predictions=cur_vcmr_redictions)
            vcmr_res.append(cur_query_pred)

    res = dict(VCMR=vcmr_res, SVMR=svmr_res, VR=vr_res)
    return {k: v for k, v in res.items() if len(v) != 0}


def get_eval_res(model, eval_dataset, opt, tasks):
    """compute and save query and video proposal embeddings"""

    if opt.similarity_measure  == "disjoint": #disjoint b_i+ e_i
        eval_res = compute_query2ctx_info_disjoint(model, eval_dataset, opt,
                                          max_before_nms=opt.max_before_nms,
                                          max_n_videos=opt.max_vcmr_video,
                                          tasks=tasks)
    elif opt.similarity_measure  in  ["general" , "exclusive" ] : # r * \hat{b_i} * \hat{e_i}
        eval_res = compute_query2ctx_info(model, eval_dataset, opt,
                                          max_before_nms=opt.max_before_nms,
                                          max_n_videos=opt.max_vcmr_video,
                                          tasks=tasks)


    eval_res["video2idx"] = eval_dataset.video2idx
    return eval_res


POST_PROCESSING_MMS_FUNC = {
    "SVMR": post_processing_vcmr_nms,
    "VCMR": post_processing_vcmr_nms
}


def eval_epoch(model, eval_dataset, opt, save_submission_filename,
               tasks=("VCMR",), max_after_nms=100):
    """max_after_nms: always set to 100, since the eval script only evaluate top-100"""
    model.eval()
    logger.info("Computing scores")
    st = time()
    eval_submission_raw = get_eval_res(model, eval_dataset, opt, tasks)

    IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw, top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    if opt.eval_split_name == "val":  # since test_public has no GT
        metrics = eval_retrieval(eval_submission, eval_dataset.query_data,
                                 iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug,
                                 use_desc_type=opt.dset_name == "tvr")
        save_metrics_path = submission_path.replace(".json", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]


    if opt.nms_thd != -1:
        logger.info("Performing nms with nms_thd {}".format(opt.nms_thd))
        eval_submission_after_nms = dict(video2idx=eval_submission_raw["video2idx"])
        if "VR" in eval_submission_raw:
            eval_submission_after_nms["VR"] = eval_submission_raw["VR"]
        for k, nms_func in POST_PROCESSING_MMS_FUNC.items():
            if k in eval_submission_raw:
                eval_submission_after_nms[k] = nms_func(eval_submission_raw[k],
                                                        nms_thd=opt.nms_thd,
                                                        max_before_nms=opt.max_before_nms,
                                                        max_after_nms=max_after_nms)

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".json", "_nms_thd_{}.json".format(opt.nms_thd))
        save_json(eval_submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_retrieval(eval_submission_after_nms, eval_dataset.query_data,
                                         iou_thds=IOU_THDS, match_number=not opt.debug, verbose=opt.debug)
            save_metrics_nms_path = submission_nms_path.replace(".json", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None

    tot_time = time() - st
    logger.info(f"validation finished in {int(tot_time)} seconds")

    return metrics, metrics_nms, latest_file_paths


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    checkpoint = torch.load(opt.ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]

    model = CONQUER(loaded_model_cfg,
                    visual_dim=opt.visual_dim,
                    text_dim=opt.text_dim,
                    query_dim=opt.query_dim,
                    hidden_dim=opt.hidden_dim,
                    video_len=opt.max_ctx_len,
                    ctx_mode=opt.ctx_mode,
                    no_output_moe_weight=opt.no_output_moe_weight,
                    similarity_measure=opt.similarity_measure,
                    use_debug = opt.debug)
    model.load_state_dict(checkpoint["model"])

    logger.info("Loaded model saved at epoch {} from checkpoint: {}"
                .format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        assert len(opt.device_ids) == 1
        # if len(opt.device_ids) > 1:
        #     logger.info("Use multi GPU", opt.device_ids)
        #     model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    data_config = load_config(opt.dataset_config)

    eval_dataset = StartEndEvalDataset(
        config = data_config,
        max_ctx_len=opt.max_ctx_len,
        max_desc_len= opt.max_desc_len,
        clip_length = opt.clip_length,
        ctx_mode = opt.ctx_mode,
        mode = opt.eval_split_name,
        data_ratio = opt.data_ratio,
        is_eval = True,
        inference_top_k = opt.max_vcmr_video)

    postfix = "_hero"
    model = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_predictions_{}{}.json".format(
        opt.dset_name, opt.eval_split_name, opt.eval_id, "_".join(opt.tasks),postfix)
    print(save_submission_filename)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename,
                       tasks=opt.tasks, max_after_nms=100)
    logger.info("metrics_no_nms \n{}".format(pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms, indent=4)))


if __name__ == '__main__':
    start_inference()
