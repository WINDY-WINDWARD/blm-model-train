from transformers import Trainer

def collate(batch):
    maxlen = max(len(f['input_ids']) for f in batch)
    ii, ll, mm = [], [], []
    for f in batch:
        seq = list(f['input_ids'])
        lab = list(f['labels'])
        msk = [1] * len(seq)
        padlen = maxlen - len(seq)
        ii.append(seq + [pad_id] * padlen)
        ll.append(lab + [-100] * padlen)
        mm.append(msk + [0] * padlen)
    return {
        'input_ids': torch.tensor(ii),
        'labels': torch.tensor(ll),
        'attention_mask': torch.tensor(mm),
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collate,
)
result = trainer.train()
print(result.metrics)
metrics = trainer.evaluate()
print('EVAL:', metrics)
