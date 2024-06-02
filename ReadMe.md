## Why Continual Pretrained?
- Tại sao cần phải Continual Pretrained? ... đơn giản là chúng ta muốn kế thừa những thành quả mà model đã được huấn luyện rất tốt trước đó, và tiếp tục huấn luyện model này trên dữ liệu cá nhân của mình để cho ra kết quả chính xác hơn.
- Tại sao không Pretrained từ đầu hoặc Fine-tuning? ... đơn giản vì Pretrained từ đầu sẽ rất tốn kém, chỉ riêng tiền thuê server huấn luyện có thể lên tới cả triệu đô với model lớn, còn nếu tiếp tục Fine-tuning thì model vẫn không hiểu được ngôn ngữ mới như tiếng Việt vì trước đây model chủ yếu được huấn luyện bằng tiếng Anh nên nó đã được học cái đó đâu, nên khi tokenizer làm việc thì sẽ phá hỏng toàn bộ những gì model đã học trước đó (T_T) "... Catastrophic forgetting (CF) is a phenomenon that occurs in machine learning when a model forgets previously learned information while acquiring new knowledge..." (https://arxiv.org/abs/2308.08747)
- Vậy sau bước Continual Pre-Training thì làm gì tiếp theo? ... sau bước Continual Pre-Training thì lại tiếp tục Continual Fine-tuning with instructions thôi. Xong bước đó thì đời sẽ tươi đẹp rồi ^^

#### (01 Jun 2024) Review: Continued Pretraining with TinyLlama 1.1B (summary)<br>
- https://lightning.ai/lightning-ai/studios/pretrain-llms-tinyllama-1-1b
- https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b

Continued pretraining is the process of continuing to update a pretrained model using new data. Take for example an LLM that is trained on news articles: There is a knowledge cutoff at the date the data for training was collected. One could add all new articles to the dataset and retrain the model from scratch, but this is very expensive. Instead, continued pretraining allows us to continue training on the recent data without discarding the previously acquired knowledge.<br>

In this Studio, we implement continued pretraining simply by loading the TinyLlama checkpoint and training on a new dataset and warming up the learning rate during the first few iterations. But in general there are some challenges involved:<br>

- Forgetting: The model could "forget" some of the knowledge it has obtained from the initial pretraining
- The performance in downstream tasks could become worse

**Prepare the dataset**<br>
TinyLlama was initially trained on 3 trillion tokens from a mix of SlimPajama and Starcoder data. In this tutorial, we choose to continue training on the OpenWebMath dataset to improve its skills in the domain of mathematics.<br><br>

Step 1: Download the data into the Studio. The math dataset is ~52 GB
```
git clone https://huggingface.co/datasets/open-web-math/open-web-math  dataset/raw/open-web-math
```

Step 2: Preprocess the dataset. Before we can consume the data in our training script, we need to tokenize it. In addition, we will use Lightning Data to optimize the dataset by creating chunks that are efficient to load...  In this case, the dataset is saved as Parquet files, and we simply tokenize the text samples with the same tokenizer that was used for training TinyLlama.<br><br>

**Warm up**<br>
we choose the fraction of warmup steps to be 1% of the training data size (~1.4K tokens), which is generally considered a good default... For the initial pretraining, TinyLlama used a minimum learning rate of 4e-5 and a maximum learning rate of 4e-4. For our experiment with OpenWebMath, we found that reducing them to 5% of the original value worked reasonably well and was enough to suppress catastrophic forgetting. (min= 0.1 x 4e-5 ; max= 0.05 x 4e-4)<br><br>

**Start the training**<br>
 You must be at least on a A10G GPU machine.<br>
 ```
litgpt pretrain --config configs/tinyllama-openwebmath.yaml --train.max_seq_length 64 --train.micro_batch_size 1
```

This will run with a reduced context size and batch size 1 to fit in memory... Let's take a quick look at the config file in configs/tinyllama-openwebmath.yaml. Here is an excerpt with the most important settings.<br>
```
train:
  global_batch_size: 512
  micro_batch_size: 4
  warmup_fraction: 0.01
  max_seq_length: 2048
  learning_rate: 0.00002
  min_lr: 0.000004
```

On a single A10G GPU, the training will take a few days, and using the full context size of 2048 tokens won't fit in the memory of an A10G. To reduce the training time down to a few hours with full context size, either switch to an A100 or H100 Studio machine, or use the Multi-Machine Training app ... <br>

```
litgpt pretrain --config configs/tinyllama-openwebmath.yaml


# Tune the micro batch size to maximize the memory usage (e.g. on H100)
litgpt pretrain --config configs/tinyllama-openwebmath.yaml --train.micro_batch_size 8
```

The training script periodically saves a checkpoint to the results folder, and at the end of training a checkpoint folder named final. <br><br>

**Results**<br>
After tuning the learning rate to the value mentioned earlier, the SlimPajama loss is increasing from 2.1 to 2.137 (a change in perplexity of 0.36), indicating a minor degradation but no catastrophic forgetting...<br>

![alt text](https://github.com/Mr-Jack-Tung/Continually-pretrained/blob/main/Screenshot%202024-06-01.png)

We see that in most tasks the metrics improved slightly, and only HellaSwag and Piqa dropped 2 points. To evaluate the effect of continued training on a math domain dataset, we also ran on the MathQA showing a slight increase in accuracy.<br>

Note that after pretraining, the model can only do next-token prediction, i.e., completing the sentence we give as the input prompt. To make it into chat model / assistant and evaluate it properly, we would have to do additional instruction finetuning and potentially further alignment but we omitted this in the interest of keeping the tutorial brief.<br><br>

**Conclusion**<br>
Continued pretraining is a cost effective method to incorporate new knowledge into a pretrained LLM without having to retrain it from scratch. Using warmup and a smart choice of learning rate schedule, we can minimize the risk that the model forgets the original data. When we did the experiments with TinyLlama and OpenWebMath in this tutorial, we saw small amount of forgetting happening when plotting the loss/perplexity over the SlimPajama dataset during training. Despite this, performance across tasks has improved by a small amount. On the other hand, a more thorough analysis of how well the model has adopted the new domain is still needed. If you are planning to train on your own data, consider creating a benchmark ahead of time based on your requirements. Finally, since we didn't include a thorough analysis on the hyperparameters, it is possible that better numbers can be achieved with an increased learning rate or different warmup schedule.<br><br>

**References**<br>
- K. Paster, M. Dos Santos, Z. Azerbayev, J. Ba, OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text, ArXiv, 2023
- P. Zhang, G. Zeng, T. Wang, W. Lu, TinyLlama: An Open-Source Small Language Model, ArXiv, 2024
- K. Gupta, B. Thérien, A. Ibrahim, M. Richter et al., Continual Pre-Training of Large Language Models: How to re-warm your model?, ICML, 2023
- Z. Ke, Y. Shao, H. Lin, T. Konishi et al., Continual Pre-training of Language Models, ICLR, 2023
- L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black et al., A framework for few-shot language model evaluation, Zenodo, 2023


#### Update 24 May 2024 - 10 PM 45
- Tham khảo OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (Support 70B+ full tuning & LoRA & Mixtral & KTO) https://github.com/OpenLLMAI/OpenRLHF
- Continue pretrain: https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_continue_pretrain_llama.sh

#### Update 11 April 2024 - 11 AM
- Update extend vocabsize with Chinese - Korean - Japanese ^^
  - "你好很高兴见到你"
  - "こんにちは、初めまして！"
  - "안녕하세요, 만나서 반갑습니다!"

#### Update 11 April 2024 - 06 AM
- Một trong những thách thức của việc Continual Pretrained đó chính là mở rộng Vocabsize. Vì khi dữ liệu mới được đưa vào để huấn luyện model sẽ gồm cả các tokens mới, mà các tokens này chưa từng tồn tại trong model trước đây (T_T)
- File name: example_extract_unicode_tokens_and_extend_vocabsize_send.py
- Kết quả: Như bạn thấy trong file chương trình hoặc khi bạn chạy file example về extend vocabsize thì lúc ban đầu tokenizer không hiển thị đúng nội dung có chưa những tokens tiếng Việt mới được nhập vào, sau khi được extend vocabsize thì tokenizer đã có thể nhận diện và hiển thị đúng nội dung phù hợp với nội dung nhập vào là ngôn ngữ mới.

#### Blogs
- Continued Pretraining with TinyLlama 1.1B (https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b)
- Tips for LLM Pretraining and Evaluating Reward Models (https://magazine.sebastianraschka.com/p/tips-for-llm-pretraining-and-evaluating-rms)
- Fine tune model – viết truyện phong cách Nam Cao (https://blog.ngxson.com/fine-tune-model-viet-truyen-phong-cach-nam-cao/)


#### Models
- Vistral-7B-Chat - Towards a State-of-the-Art Large Language Model for Vietnamese (https://huggingface.co/Viet-Mistral/Vistral-7B-Chat)
- Github: https://github.com/vilm-ai/llm-factory ; https://github.com/vilm-ai/vietcuna
- **Model Description:** Vistral-7B-chat, a multi-turn conversational large language model for Vietnamese. Vistral is extended from the Mistral 7B model using diverse data for **continual pre-training** and **instruction tuning**. In particular, our process to develop Vistral involves:
  - Extend the tokenizer of Mistral 7B to better support Vietnamese.
  - Perform continual pre-training for Mistral over a diverse dataset of Vietnamese texts that are meticulously cleaned and deduplicated.
  - Perform supervised fine-tuning for the model using diverse instruction data. We design a set of instructions to align the model with the safety criteria in Vietnam.
- Paper:
  - CulturaX: A Cleaned, Enormous, and Multilingual Dataset for Large Language Models in 167 Languages (https://arxiv.org/abs/2309.09400)
- **People:**
  - Quan Nguyen _ https://github.com/qnguyen3 ; https://www.linkedin.com/in/qnguyen3/ ; https://twitter.com/stablequan
  - Thien Huu Nguyen _ https://ix.cs.uoregon.edu/~thien/index.html
  - Minh Nguyen _ https://minhhdvn.github.io
  - Chien Nguyen _ https://chiennv2000.github.io ; https://twitter.com/chiennv2000

#### News
- Bỏ việc tại Mỹ, về làm trí tuệ nhân tạo miễn phí cho người Việt (https://thanhnien.vn/bo-viec-tai-my-ve-lam-tri-tue-nhan-tao-mien-phi-cho-nguoi-viet-185240121183106753.htm)
- Nhóm kỹ sư GenZ làm ứng dụng trí tuệ nhân tạo miễn phí cho người Việt (https://vnexpress.net/nhom-ky-su-genz-lam-ung-dung-tri-tue-nhan-tao-mien-phi-cho-nguoi-viet-4708838.html)
- Chuyên gia về ai đưa ra phác hoạ về sự phát triển của Trí tuệ nhân tạo tại Việt Nam (https://www.elleman.vn/tin-tuc/phat-trien-ai-2023-elleman)

#### Papers
- Simple and Scalable Strategies to Continually Pre-train Large Language Models (https://arxiv.org/abs/2403.08763)
- RewardBench: Evaluating Reward Models for Language Modeling (https://arxiv.org/abs/2403.13787)
- Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order (https://arxiv.org/abs/2404.00399)
- An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning (https://arxiv.org/abs/2308.08747)
- Continual Pre-Training of Large Language Models: How to (re)warm your model? (https://arxiv.org/abs/2308.04014)
- Continual Pre-training of Language Models (https://arxiv.org/abs/2302.03241)
- Adapting a Language Model While Preserving its General Knowledge (https://arxiv.org/abs/2301.08986)
- Continual Training of Language Models for Few-Shot Learning (https://arxiv.org/abs/2210.05549)
- Adapting BERT for Continual Learning of a Sequence of Aspect Sentiment Classification Tasks (https://arxiv.org/abs/2112.03271)
- CLASSIC: Continual and Contrastive Learning of Aspect Sentiment Classification Tasks (https://arxiv.org/abs/2112.02714)
- Achieving Forgetting Prevention and Knowledge Transfer in Continual Learning (https://arxiv.org/abs/2112.02706)
- Don't Stop Pretraining: Adapt Language Models to Domains and Tasks (https://arxiv.org/abs/2004.10964)

#### Github
- https://github.com/UIC-Liu-Lab/ContinualLM
- https://github.com/ZixuanKe/PyContinual
- https://github.com/xyjigsaw/LLM-Pretrain-SFT
- https://github.com/rasbt/LLMs-from-scratch
- https://github.com/google/sentencepiece
- https://github.com/kyegomez/Zeta
- https://github.com/jzhang38/TinyLlama


