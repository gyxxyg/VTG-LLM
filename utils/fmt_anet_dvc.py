
import json
from eval_dvc_anet import eval_dvc
from eval_soda import eval_soda

data_path = "yourpath/fmt_activitynet_test_f96_result.json"
gt_file = 'yourpath/data/dense_video_captioning/anet/val_2.json'

with open(data_path, 'r') as f:
    data = json.load(f)

new_data = {}

for vid in data.keys():
    new_item = []
    for pred in data[vid]:
        new_item.append({
                            'timestamp': pred['timestamp'],
                            'sentence': pred['caption'],
                        })

    new_data[vid.replace('.mp4', '')] = new_item

with open(gt_file, 'r') as f:
    gt_data = json.load(gt_file)

gt_data = {k: v for k, v in gt_data.items() if k in pred.keys()}

pred_result = {'results': new_data}

# metrics = eval_soda(pred_result, [gt_js])
metrics = {}
metrics.update(eval_dvc(pred_result, [gt_js], 
            tious=[0.3, 0.5, 0.7, 0.9], 
            distances=[],
            max_proposals_per_video=1000, 
            verbose=False, 
            no_lang_eval=False))
print(f'Found {len(new_data)} log')
metrics = {k: v * 100 for k, v in metrics.items() if k in ['soda_c', 'METEOR', 'CIDEr']}

print(metrics)


