# SuperCD

- An implementation for LREC-COLING 2024 paper [Few-shot Named Entity Recognition via Superposition Concept Discrimination](http://arxiv.org/abs/2305.11038)

## Quick links

* [Environment](#Environment)
* [Model](#model)
* [Active Learning](#active-learning)

### Environment

```bash
conda create -n supercd python=3.9.0
conda activate supercd
bash env.sh
```
### Model

The pre-trained models are in huggingface: [SIR](https://huggingface.co/jiawei1998/SIR) and [CE](https://huggingface.co/jiawei1998/CE) 

### Active Learning
You can run:
```bash
python main.py --output_dir output_dir \
--dataset ${dataset} \
--plm bert-base-uncased \
--plmpath bert-base-uncased \
--modelname tagmodel \
--per_device_train_batch_size 4 \
--do_train \
--shot 5 \
--maxshot 5 \
--save_strategy no \
--num_train_epochs 10 \
--learning_rate 1e-4 \
--warmup_ratio 0.1 \
--active supercd \
--save_total_limit 1 
```
The result will be in output_dir. You can change the `shot` for different shot and `maxshot` is the additional shot for active learning.

For different pre-trained model, you should change `plm` and `plmpath`.

For different base model, you can change `modelname` (tagmodel, structshot, proto, sdnet or container)

`num_train_epochs` is set to 50 for sdnet and 10 for other models.

`learning_rate` is set to 5e-5 for container and 1e-4 for other models.

## License

The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg