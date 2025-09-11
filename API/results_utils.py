from models.response import ImageResult
from typing import List
from models.entry_models import StageModalities
import asyncio
from search_utils import call_text_search, call_ocr_search, call_asr_search, call_image_search
from utils import normalize_score, get_weighted_union_results


MODAL_ORDER = ['text', 'img', 'ocr', 'asr', 'localized']

# Sentinel value for atemporal results (e.g., ASR without timestamps)
ATEMPORAL_TIME = -1.0

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
        field_name = modalities.img.value # e.g., "uploaded_image"
        img_file = form.get(field_name)   # Get actual file from form data
        if img_file is not None:
            tasks['img'] = asyncio.create_task(call_image_search(img_file, top_k))
    
    return tasks


async def process_search_results(tasks, modalities: StageModalities) ->  List[List[ImageResult]]:
    """Process search results and apply weights (internal stage)
    # Let's say tasks contains:
        tasks = {
            'text_search': search_text_async("cats"),      # Takes 2 seconds
            'image_search': search_images_async("cats"),   # Takes 3 seconds  
            'video_search': search_videos_async("cats")    # Takes 1 second
        }

    """
    singe_stage_results = []
    record_order = []
    """
    What gather does:
        Concurrent execution: Runs multiple async tasks simultaneously (not sequentially)
        Wait for all: Blocks until ALL tasks complete
        Preserves order: Returns results in the same order as input tasks
        Unpacking: *tasks.values() unpacks the task collection
    """
    if tasks:
        results = await asyncio.gather(*tasks.values())
        # for each search's result
        for (key, res) in zip(tasks.keys(), results):
            # key can be 'ocr', 'asr' while res is List[ImageResult]
            if res:
                res = normalize_score(res)
                singe_stage_results.append(res)
                record_order.append(key)

    # Weight handling (required by contract)
    weight_list = [float(modalities.weight_dict.get(key_name, 0.0)) for key_name in record_order]
    weighted_union_results = get_weighted_union_results(singe_stage_results, weight_list, fuse=True)
    
    return weighted_union_results, record_order

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
    
    # Process results and apply weights for each stage
    results, record_order = await process_search_results(tasks, modalities)

    # reorder list of results
    results = reorder_modal_results(results, record_order)

    if len(results) > 1:
        temporal_chain_results = temporal_chain(results, 2) 
        res_temporal_chain = update_temporal_score(temporal_chain_results)
    else: 
        res_temporal_chain = results[0]

    return res_temporal_chain


def temporal_chain(stages: List[List[ImageResult]], window_s: float = 2) -> List[ImageResult]:
    print("PERFORMING TEMPORAL CHAIN")
    seqs = []
    for hit in stages[0]:
        # best stores each best hit in each stage
        best = [(hit, hit['score'])] # (ImageResult, ImageResult.score)
        cur = hit
        ok = True

        for i in range(1, len(stages)):

            cands = []
            for h in stages[i]:
                if h['video_name'] != cur['video_name']:
                    continue
                pts = float(h.get('pts_time', ATEMPORAL_TIME))
                cur_pts = float(cur.get('pts_time', ATEMPORAL_TIME))
                
                if pts == ATEMPORAL_TIME or cur_pts == ATEMPORAL_TIME:
                    cands.append(h)  # allow ASR/atemporal
                elif abs(pts - cur_pts) <= window_s:
                    cands.append(h)

            if not cands:
                ok = False 
                break # stop the next of current video

            nxt = max(cands, key=lambda h: h['score'] + best[-1][1]) # add last score in chain and picks the hit with max score
            best.append( (nxt, best[-1][1] + nxt['score']) ) # add it to best
            cur = nxt
        if ok:
            # add temporal chain hits across stage to seqs
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
                     if h['video_name'] == cur['video_name'] and (h['pts_time'] >= cur['pts_time'])]
            
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


def update_temporal_score(temporal_results: List[List[tuple]]) -> List[ImageResult]:
    """Convert temporal chain results back to List[ImageResult] with updated scores"""
    final_results = []
    
    for chain in temporal_results:
        # Each chain is a list of tuples (hit, new_score)
        for hit, new_score in chain:
            # hit is a ImageResult (video_id, frame_id,image_path,  pts, score)
            # Update the hit's score with the new temporal score
            hit['score'] = new_score
            final_results.append(hit)
    
    return final_results

