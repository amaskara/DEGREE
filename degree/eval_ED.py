import os, sys, json, logging, pprint, tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import GenerativeModel
from data import GenDataset, EEDataset
from utils import compute_f1
from argparse import ArgumentParser, Namespace
import ipdb

# configuration
parser = ArgumentParser()
parser.add_argument('-ced', '--ed_config', required=False)
parser.add_argument('-ed', '--ed_model', required=True)
parser.add_argument('-g', '--gold_trigger', action='store_true', default=False)
parser.add_argument('--no_dev', action='store_true', default=False)
parser.add_argument('--eval_batch_size', type=int)
parser.add_argument('--write_file', type=str)
args = parser.parse_args()




# check ed_model
assert (args.ed_config and args.ed_model) or args.gold_trigger
if args.ed_model:
    with open(args.ed_config) as fp:
        ed_config = json.load(fp)
    ed_config = Namespace(**ed_config)

if ed_config.dataset == "ace05e" or ed_config.dataset == "ace05ep":
    from template_generate_ace import eve_template_generator
    template_file = "template_generate_ace"
elif ed_config.dataset == "ere":
    from template_generate_ere import eve_template_generator
    template_file = "template_generate_ere"
# fix random seed


# set GPU device
# torch.cuda.set_device(ed_config.gpu_device)

np.random.seed(ed_config.seed)
torch.manual_seed(ed_config.seed)
torch.backends.cudnn.enabled = False

# logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]')
logger = logging.getLogger(__name__)

def get_span_idx(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return -1, -1
    else:
        if trigger_span is None:
            return candidates[0]
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))[0]

def get_span_idx_tri(pieces, token_start_idxs, span, tokenizer, trigger_span=None):
    """
    This function is how we map the generated prediction back to span prediction.

    Detailed Explanation:
        We will first split our prediction and use tokenizer to tokenize our predicted "span" into pieces. Then, we will find whether we can find a continuous span in the original "pieces" can match tokenized "span". 

    If it is an argument/relation extraction task, we will return the one which is closest to the trigger_span.
    """
    words = []
    for s in span.split(' '):
        words.extend(tokenizer.encode(s, add_special_tokens=False))
    
    candidates = []
    for i in range(len(pieces)):
        j = 0
        k = 0
        while j < len(words) and i+k < len(pieces):
            if pieces[i+k] == words[j]:
                j += 1
                k += 1
            elif tokenizer.decode(words[j]) == "":
                j += 1
            elif tokenizer.decode(pieces[i+k]) == "":
                k += 1
            else:
                break
        if j == len(words):
            candidates.append((i, i+k))
            
    candidates = [(token_start_idxs.index(c1), token_start_idxs.index(c2)) for c1, c2 in candidates if c1 in token_start_idxs and c2 in token_start_idxs]
    if len(candidates) < 1:
        return [(-1, -1)]
    else:
        if trigger_span is None:
            return candidates
        else:
            return sorted(candidates, key=lambda x: np.abs(trigger_span[0]-x[0]))

def cal_scores(gold_triggers, pred_triggers):
    assert len(gold_triggers) == len(pred_triggers)

    # tri_id
    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set([(t[0], t[1]) for t in gold_trigger])
        pred_set = set([(t[0], t[1]) for t in pred_trigger])
        gold_tri_id_num += len(gold_set)
        pred_tri_id_num += len(pred_set)
        match_tri_id_num += len(gold_set & pred_set)
    
    # tri_cls
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set(gold_trigger)
        pred_set = set(pred_trigger)
        gold_tri_cls_num += len(gold_set)
        pred_tri_cls_num += len(pred_set)
        match_tri_cls_num += len(gold_set & pred_set)
    
    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
    }
    
    return scores


# check valid styles
assert np.all([style in ['event_type_sent', 'keywords', 'template'] for style in ed_config.input_style])
assert np.all([style in ['trigger:sentence'] for style in ed_config.output_style])
              
# tokenizer
ed_tokenizer = AutoTokenizer.from_pretrained(ed_config.model_name, cache_dir=ed_config.cache_dir)
special_tokens = ['<Trigger>', '<sep>']
ed_tokenizer.add_tokens(special_tokens)

if args.eval_batch_size:
    ed_config.eval_batch_size=args.eval_batch_size

# load data
dev_set = EEDataset(ed_tokenizer, ed_config.dev_file, max_length=ed_config.max_length)
test_set = EEDataset(ed_tokenizer, ed_config.test_file, max_length=ed_config.max_length)
dev_batch_num = len(dev_set) // ed_config.eval_batch_size + (len(dev_set) % ed_config.eval_batch_size != 0)
test_batch_num = len(test_set) // ed_config.eval_batch_size + (len(test_set) % ed_config.eval_batch_size != 0)
with open(ed_config.vocab_file) as f:
    vocab = json.load(f)

# load model
logger.info(f"Loading model from {args.ed_model}")
ed_model = GenerativeModel(ed_config, ed_tokenizer)
ed_model.load_state_dict(torch.load(args.ed_model, map_location=f'cuda:{ed_config.gpu_device}'))
ed_model.cuda(device=ed_config.gpu_device)
ed_model.eval()

# eval dev set
if not args.no_dev:
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev')
    dev_gold_triggers, dev_pred_triggers = [], [], [], []
    for batch in DataLoader(dev_set, batch_size=ed_config.eval_batch_size, shuffle=False, collate_fn=dev_set.collate_fn):
        progress.update(1)
        
        # trigger predictions
        if args.gold_trigger:
            p_triggers = batch.triggers
        else:
            p_triggers = [[] for _ in range(len(batch.tokens))]
            for event_type in vocab['event_type_itos']:
                theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)
                
                inputs = []
                for tokens in batch.tokens:
                    template = theclass(ed_config.input_style, ed_config.output_style, tokens, event_type)
                    inputs.append(template.generate_input_str(''))
                
                inputs = ed_tokenizer(inputs, return_tensors='pt', padding=True, max_length=ed_config.max_length)
                enc_idxs = inputs['input_ids'].cuda()
                enc_attn = inputs['attention_mask'].cuda()
                
                outputs = ed_model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, num_beams=ed_config.beam_size, max_length=ed_config.max_output_length)
                final_outputs = [ed_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
                
                for bid, (tokens, p_text) in enumerate(zip(batch.tokens, final_outputs)):
                    template = theclass(ed_config.input_style, ed_config.output_style, tokens, event_type)
                    pred_object = template.decode(p_text)
                    triggers_ = [mention+(event_type, ) for span, _, _ in pred_object for mention in get_span_idx_tri(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, ed_tokenizer)]
                    triggers_ = [t for t in triggers_ if t[0] != -1]
                    triggers_ = list(set(triggers_))
                    p_triggers[bid].extend(triggers_)
                
            # if ed_config.ignore_first_header:
            #     for bid, wnd_id in enumerate(batch.wnd_ids):
            #         if int(wnd_id.split('-')[-1]) < 4:
            #             p_triggers[bid] = []
        
        
        dev_gold_triggers.extend(batch.triggers)
        dev_pred_triggers.extend(p_triggers)

                
    progress.close()
    
    # calculate scores
    dev_scores = cal_scores(dev_gold_triggers, dev_pred_triggers)
    
    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_id'][3] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][1], 
        dev_scores['tri_id'][4] * 100.0, dev_scores['tri_id'][2], dev_scores['tri_id'][0], dev_scores['tri_id'][5] * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        dev_scores['tri_cls'][3] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][1], 
        dev_scores['tri_cls'][4] * 100.0, dev_scores['tri_cls'][2], dev_scores['tri_cls'][0], dev_scores['tri_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
    
    
# test set
progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test')
test_gold_triggers, test_gold_roles, test_pred_triggers, test_pred_roles = [], [], [], []
write_object = []
for batch in DataLoader(test_set, batch_size=ed_config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    
    p_tri_texts = [[] for _ in range(len(batch.tokens))]
    p_arg_texts = [[] for _ in range(len(batch.tokens))]
    # trigger predictions
    if args.gold_trigger:
        p_triggers = batch.triggers
    else:
        p_triggers = [[] for _ in range(len(batch.tokens))]
        for event_type in vocab['event_type_itos']:
            theclass = getattr(sys.modules[template_file], event_type.replace(':', '_').replace('-', '_'), False)
            
            inputs = []
            for tokens in batch.tokens:
                template = theclass(ed_config.input_style, ed_config.output_style, tokens, event_type)
                inputs.append(template.generate_input_str(''))
            
            inputs = ed_tokenizer(inputs, return_tensors='pt', padding=True, max_length=ed_config.max_length)
            enc_idxs = inputs['input_ids'].cuda()
            enc_attn = inputs['attention_mask'].cuda()
            
            outputs = ed_model.model.generate(input_ids=enc_idxs, attention_mask=enc_attn, num_beams=ed_config.beam_size, max_length=ed_config.max_output_length)
            final_outputs = [ed_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
            
            for bid, (tokens, p_text) in enumerate(zip(batch.tokens, final_outputs)):
                template = theclass(ed_config.input_style, ed_config.output_style, tokens, event_type)
                pred_object = template.decode(p_text)
                triggers_ = [mention+(event_type, ) for span, _, _ in pred_object for mention in get_span_idx_tri(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, ed_tokenizer)]
                triggers_ = [t for t in triggers_ if t[0] != -1]
                triggers_ = list(set(triggers_))
                p_triggers[bid].extend(triggers_)
                p_tri_texts[bid].append(p_text)
            
        if ed_config.ignore_first_header:
            for bid, wnd_id in enumerate(batch.wnd_ids):
                if int(wnd_id.split('-')[-1]) < 4:
                    p_triggers[bid] = []
    
    test_gold_triggers.extend(batch.triggers)
    test_pred_triggers.extend(p_triggers)

    for gt, gr, pt, pr, tte, ate in zip(batch.triggers, p_triggers, p_tri_texts):
        write_object.append({
            "pred trigger text": tte,
            "pred triggers": pt,
            "gold triggers": gt,
        })
            
progress.close()

# calculate scores
test_scores = cal_scores(test_gold_triggers, test_pred_triggers, test_gold_roles, test_pred_roles)

print("---------------------------------------------------------------------")
print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
    test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
    test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))
print("---------------------------------------------------------------------")


if args.write_file:
    with open(args.write_file, 'w') as fw:
        json.dump(write_object, fw, indent=4)          
