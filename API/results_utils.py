from models import ImageResult
from typing import List

def sort_to_stages():
    pass




def temporal_chain(stages: List[List[ImageResult]], window_s: float) -> List[ImageResult]:
    seqs = []
    for hit in stages[0]:
        best = [(hit, hit.score)]
        cur = hit
        ok = True

        for i in range(1, len(stages)):
            cands = [ h for h in stages[i]
                     if h.video_name == cur.video_name and abs(h.pts_time - cur.pts_time) <= window_s]
            
            if not cands:
                ok = False 
                break # stop the next of current video

            nxt = max(cands, key=lambda h: h.score + best[-1][1]) # add last score in chain
            best.append( (nxt, best[-1][1] + nxt.score) )
            cur = nxt
        if ok:
            seqs.append(best)

    sorted_seqs = sorted(seqs, key=lambda seq: seq[-1][1], reverse=True)

    return sorted_seqs            

