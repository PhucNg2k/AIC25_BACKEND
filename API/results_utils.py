from models.response import ImageResult
from typing import List
from models.entry_models import StageModalities
import asyncio
from search_utils import call_text_search, call_ocr_search, call_asr_search, call_image_search
from utils import normalize_score, get_weighted_union_results


MODAL_ORDER = ['text', 'img', 'ocr', 'asr', 'localized']

def reorder_modal_results(modalities: List[List[ImageResult]], record_order: List[str]) -> List[List[ImageResult]]:
    """Reorder modality results to match MODAL_ORDER"""
    final_res = []
    
    # Create a mapping from modality name to its results
    modality_map = {}
    for i, modality_name in enumerate(record_order):
        modality_map[modality_name] = modalities[i]
    
    # Reorder according to MODAL_ORDER
    for modal_name in MODAL_ORDER:
        if modal_name in modality_map:
            final_res.append(modality_map[modal_name])
    
    return final_res


def create_search_tasks(modalities: StageModalities, form, top_k: int):
    """Create async tasks for each modality search"""
    tasks = {}
    
    if modalities.text and modalities.text.value:
        tasks['text'] = asyncio.create_task(call_text_search(modalities.text.value, top_k))

    if modalities.ocr and modalities.ocr.value:
        tasks['ocr'] = asyncio.create_task(call_ocr_search(modalities.ocr.value, top_k))

    if modalities.asr and modalities.asr.value:
        tasks['asr'] = asyncio.create_task(call_asr_search(modalities.asr.value, top_k))

    if modalities.img and modalities.img.value:
        field_name = modalities.img.value
        img_file = form.get(field_name)
        if img_file is not None:
            tasks['img'] = asyncio.create_task(call_image_search(img_file, top_k))
    
    return tasks


async def process_search_results(tasks, modalities: StageModalities) ->  List[List[ImageResult]]:
    """Process search results and apply weights"""
    singe_stage_results = []
    record_order = []

    if tasks:
        results = await asyncio.gather(*tasks.values())
        for (key, res) in zip(tasks.keys(), results):
            if res:
                res = normalize_score(res)
                singe_stage_results.append(res)
                record_order.append(key)

    # Weight handling (required by contract)
    weight_list = [float(modalities.weight_dict.get(key_name, 0.0)) for key_name in record_order]
    weighted_union_results = get_weighted_union_results(singe_stage_results, weight_list, fuse=True)
    
    return weighted_union_results, record_order


def update_temporal_score(temporal_results: List[List[tuple]]) -> List[ImageResult]:
    """Convert temporal chain results back to List[ImageResult] with updated scores"""
    final_results = []
    
    for chain in temporal_results:
        # Each chain is a list of tuples (hit, new_score)
        for hit, new_score in chain:
            # Update the hit's score with the new temporal score
            hit['score'] = new_score
            final_results.append(hit)
    
    return final_results


def discard_duplicate_frame(frame_list: List[ImageResult]) -> List[ImageResult]:
    """Remove duplicate frames based on video_name and frame_idx, keeping the one with highest score"""
    if not frame_list:
        return []
    
    # Create a dictionary to track unique frames by (video_name, frame_idx)
    unique_frames = {}
    
    for frame in frame_list:
        key = (frame['video_name'], frame['frame_idx'])
        
        if key not in unique_frames:
            unique_frames[key] = frame
    
    # Return the unique frames as a list
    return list(unique_frames.values())

async def process_one_stage(modalities: StageModalities, form, top_k: int):
    """Process a single stage with all its modalities"""
    if not isinstance(modalities, StageModalities):
        return None

    # Create search tasks for all modalities
    tasks = create_search_tasks(modalities, form, top_k)
    
    # Process results and apply weights
    results, record_order = await process_search_results(tasks, modalities)

    results = reorder_modal_results(results, record_order)

    if len(results) > 1:
        temporal_chain_results = temporal_chain(results, 2) 
        res_temporal_chain = update_temporal_score(temporal_chain_results)
    else: 
        res_temporal_chain = results[0]

    return res_temporal_chain


def temporal_chain(stages: List[List[ImageResult]], window_s: float) -> List[ImageResult]:
    print("PERFORMING TEMPORAL CHAIN")
    seqs = []
    for hit in stages[0]:
        best = [(hit, hit['score'])]
        cur = hit
        ok = True

        for i in range(1, len(stages)):
            cands = [ h for h in stages[i]
                     if h['video_name'] == cur['video_name'] and abs(h['pts_time'] - cur['pts_time']) <= window_s]
            
            if not cands:
                ok = False 
                break # stop the next of current video

            nxt = max(cands, key=lambda h: h['score'] + best[-1][1]) # add last score in chain
            best.append( (nxt, best[-1][1] + nxt['score']) )
            cur = nxt
        if ok:
            seqs.append(best)

    sorted_seqs = sorted(seqs, key=lambda seq: seq[-1][1], reverse=True)

    return sorted_seqs            


def events_chain(stages: List[List[ImageResult]]) -> List[ImageResult]:
    print("\nPERFORMING EVENT CHAINING")
    seqs = []
    for hit in stages[0]:
        best = [(hit, hit['score'])]
        cur = hit
        ok = True

        for i in range(1, len(stages)):
            cands = [ h for h in stages[i]
                     if h['video_name'] == cur['video_name'] and (h['pts_time'] > cur['pts_time'])]
            
            if not cands:
                ok = False 
                break # stop the next of current video

            nxt = max(cands, key=lambda h: h['score'] + best[-1][1]) # add last score in chain
            best.append( (nxt, best[-1][1] + nxt['score']) )
            cur = nxt
        if ok:
            seqs.append(best)

    sorted_seqs = sorted(seqs, key=lambda seq: seq[-1][1], reverse=True)

    return sorted_seqs