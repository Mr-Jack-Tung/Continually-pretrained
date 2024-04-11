## Why Continual Pretrained?
- Tại sao cần phải Continual Pretrained? đơn giản là chúng ta muốn kế thừa những thành quả mà model đã được huấn luyện rất tốt trước đó, và tiếp tục huấn luyện model này trên dữ liệu cá nhân của mình để cho ra kết quả chính xác hơn.
- Tại sao không Fine-tuning hoặc Pretrained từ đầu? đơn giản vì Pretrained từ đầu thì rất tốn kém, tiền thuê server huấn luyện có thể lên tới cả triệu đô với model lớn, còn nếu Fine-tuning thì model vẫn không hiểu được ngôn ngữ mới như tiếng Việt vì trước đây model nó đã được học cái đó đâu, nên khi tokenizer làm việc thì sẽ phá hỏng toàn bộ những gì model đã học trước đó (T_T)

#### Update 11 April 2024:
- Một trong những thách thức của việc Continual Pretrained đó chính là mở rộng Vocabsize. Vì khi dữ liệu mới được đưa vào để huấn luyện model sẽ gồm cả các tokens mới, mà các tokens này không có trong model trước đây. Điều này sẽ làm hỏng model đã được huấn luyện trước đó (T_T)
- File name: example_extract_unicode_tokens_and_extend_vocabsize_send.py
- Kết quả: Như bạn thấy trong file chương trình hoặc khi bạn chạy file example về extend vocabsize thì lúc ban đầu tokenizer không hiển thị đúng nội dung có chưa những tokens tiếng Việt mới được nhập vào, sau khi được extend vocabsize thì tokenizer đã có thể nhận diện và hiển thị đúng nội dung phù hợp với nội dung nhập vào là ngôn ngữ mới.

#### Blogs
- Continued Pretraining with TinyLlama 1.1B (https://lightning.ai/lightning-ai/studios/continued-pretraining-with-tinyllama-1-1b)
- Tips for LLM Pretraining and Evaluating Reward Models (https://magazine.sebastianraschka.com/p/tips-for-llm-pretraining-and-evaluating-rms)


#### Models
- Vistral-7B-Chat - Towards a State-of-the-Art Large Language Model for Vietnamese (https://huggingface.co/Viet-Mistral/Vistral-7B-Chat)

#### Papers
- Simple and Scalable Strategies to Continually Pre-train Large Language Models (https://arxiv.org/abs/2403.08763)
- RewardBench: Evaluating Reward Models for Language Modeling (https://arxiv.org/abs/2403.13787)
- Aurora-M: The First Open Source Multilingual Language Model Red-teamed according to the U.S. Executive Order (https://arxiv.org/abs/2404.00399)
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


