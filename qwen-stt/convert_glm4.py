import os
import json
import click
import torch
import math
import librosa
from glob import glob
from multiprocess import Pool
from tqdm import tqdm

def new_path(f):
    f = f.replace('.mp3', '.glm4')
    splitted = f.split('/')
    base_folder = splitted[0] + '_glm4'
    splitted = '/'.join([base_folder] + splitted[1:])
    return splitted

def chunks(l, devices):
    chunk_size = len(l) // len(devices)
    remainder = len(l) % len(devices)
    start = 0
    for i in range(len(devices)):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        yield (l[start:end], devices[i])
        start = end

def loop(rows):
    rows, index = rows
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)
        
    import torch
    torch.set_grad_enabled(False)

    from glm4_audio_tokenizer import Glm4Tokenizer
    from glm4_audio_tokenizer.utils import extract_speech_token
    from torch.utils.data import Dataset, DataLoader

    class CustomDataset(Dataset):
        def __init__(self, rows):
            self.rows = rows
    
        def __len__(self):
            return len(self.rows)
    
        def __getitem__(self, idx):
            try:
                return self.rows[idx], librosa.load(self.rows[idx], sr = 16000)
            except:
                return None
    
    def collator(batch):
        batch = [b for b in batch if b is not None]
        filenames = [b[0] for b in batch]
        audio = [b[1] for b in batch]
        return {'filenames': filenames, 'audio': audio}

    model = Glm4Tokenizer().to(torch.float16).cuda()

    data = CustomDataset(rows)
    dataloader = DataLoader(
        data, batch_size=24, collate_fn = collator, 
        num_workers = 10, prefetch_factor = 5, pin_memory = True)
    
    for batch in tqdm(iter(dataloader)):
        try:
            tokens = extract_speech_token(model.whisper_model, model.feature_extractor, batch['audio'])
            for no, f in enumerate(batch['filenames']):
                splitted = new_path(f)
                os.makedirs(os.path.split(splitted)[0], exist_ok = True)
                with open(splitted, 'w') as fopen:
                    json.dump(tokens[no], fopen)
        except Exception as e:
            print(e)

@click.command()
@click.option('--path', default = '*segment/*.mp3')
@click.option('--replication', default = 1)
def main(path, replication):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    else:
        devices = [d.strip() for d in devices.split(',')]

    devices = replication * devices
    print(devices)
    
    rows = glob(path)
    print(len(rows))
    rows = [f for f in tqdm(rows) if not os.path.exists(new_path(f))]
    print(len(rows))
    
    df_split = chunks(rows, devices)
    pool = Pool(len(devices))
    pooled = pool.map(loop, df_split)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()