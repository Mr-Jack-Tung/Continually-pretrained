# -*- coding: utf-8 -*-
# Author: Mr.Jack _ www.BICweb.vn
# Co-Author: ChatGPT 3.5 ^^
# Date: 11 April 2024 - 05 AM


from collections import Counter

from transformers import GPT2Tokenizer

# Load pretrained GPT-2 tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def preprocess_and_extract_tokens(data):
    # Tách từ và chuẩn hóa các tokens thành dạng Unicode
    words = []
    for text in data:
        words.extend(list(str(text))) # Tách từng chữ cái
        words.extend(str(text).split()) # Tách từng từ, không phân biệt chữ hoa chữ thường
        words.extend(str(text).lower().split()) # Chuyển về chữ thường và tách từng từ

    # Đếm tần suất xuất hiện của từng token
    token_counts = Counter(words)

    # print("\ntoken_counts:",token_counts)

    # Lọc ra các tokens mới mà không có trong bộ từ vựng hiện tại của tokenizer
    existing_tokens = set(tokenizer.get_vocab().keys())

    new_tokens = [token for token in token_counts if token not in existing_tokens]

    return new_tokens

# Dữ liệu Tiếng Việt của bạn (ví dụ)

text_data = [
    "Hello, how are you?",
    "你好很高兴见到你",
    "こんにちは、初めまして！",
    "안녕하세요, 만나서 반갑습니다!",
    "Bạn đã sẵn sàng để bắt đầu chưa ?",
    "Đây là một ví dụ về tiền xử lý dữ liệu Tiếng Việt.",
    "Chúng ta sẽ thực hiện tách từ và trích xuất các tokens mới từ dữ liệu này.",
    "Các tokens mới này là các tokens đặc thù trong ngôn ngữ Tiếng Việt.",
    "Chương trình sẽ tự động trích xuất các tokens đặc trưng và cũng tự động thêm vào từ điển."
]

print(text_data)

print("\ntokenizer.get_vocab size:", len(tokenizer.get_vocab().keys()))

# Convert new tokens to strings
# Print tokens before added to vocabulary
print("\nTokens before extending vocabulary:")
for text in text_data:
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    str_tokens_before_add = [tokenizer.decode([token], skip_special_tokens=True) for token in text_tokens]
    print(str_tokens_before_add, end=", ")

print("\n")

# Gọi hàm preprocess_and_extract_tokens để trích xuất tokens mới
new_tokens = preprocess_and_extract_tokens(text_data)

print("\nNew extract Tokens:")
print(new_tokens)

# Add new tokens to tokenizer
tokenizer.add_tokens(new_tokens)

from transformers import GPT2LMHeadModel

# Load pretrained model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Resize model's embeddings
model.resize_token_embeddings(len(tokenizer))

print("\ntokenizer.get_vocab size:", len(tokenizer.get_vocab().keys()))

# Print tokens after extending vocabulary
print("\nTokens after extending vocabulary:")
for text in text_data:
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    str_tokens_after_add = [tokenizer.decode([token], skip_special_tokens=True) for token in text_tokens]
    print(str_tokens_after_add, end=", ")


"""
['Hello, how are you?', '你好很高兴见到你', 'こんにちは、初めまして！', '안녕하세요, 만나서 반갑습니다!', 'Bạn đã sẵn sàng để bắt đầu chưa ?', 'Đây là một ví dụ về tiền xử lý dữ liệu Tiếng Việt.', 'Chúng ta sẽ thực hiện tách từ và trích xuất các tokens mới từ dữ liệu này.', 'Các tokens mới này là các tokens đặc thù trong ngôn ngữ Tiếng Việt.', 'Chương trình sẽ tự động trích xuất các tokens đặc trưng và cũng tự động thêm vào từ điển.']

tokenizer.get_vocab size: 50257

Tokens before extending vocabulary:
['Hello', ',', ' how', ' are', ' you', '?'], ['�', '�', '�', '�', '�', '�', '�', '��', '�', '�', '�', '�', '�', '�', '�', '�', '�'], ['こ', 'ん', 'に', '�', '�', 'は', '、', '�', '�', '�', '�', 'ま', 'し', 'て', '�', '�', '�'], ['�', '�', '�', '�', '�', '�', '�', '�', '�', '�', '�', '�', '�', '�', ',', ' �', '�', '�', '�', '�', '�', '�', '�', '�', ' �', '�', '�', '�', '�', '�', '�', '�', '�', '�', '�', '�', '�', '!'], ['B', '�', '�', '�', 'n', ' �', '�', 'ã', ' s', '�', '�', '�', 'n', ' s', 'à', 'ng', ' �', '�', '�', '�', '�', ' b', '�', '�', '�', 't', ' �', '�', '�', '�', '�', 'u', ' ch', '�', '�', 'a', '?'], ['�', '�', 'â', 'y', ' l', 'à', ' m', '�', '�', '�', 't', ' v', 'í', ' d', '�', '�', '�', ' v', '�', '�', '�', ' ti', '�', '�', '�', 'n', ' x', '�', '�', '�', ' l', '�', '�', ' d', '�', '�', '�', ' li', '�', '�', '�', 'u', ' Ti', '�', '�', '�', 'ng', ' Vi', '�', '�', '�', 't', '.'], ['Ch', 'ú', 'ng', ' ta', ' s', '�', '�', '�', ' th', '�', '�', '�', 'c', ' hi', '�', '�', '�', 'n', ' t', 'á', 'ch', ' t', '�', '�', '�', ' v', 'à', ' tr', 'í', 'ch', ' x', 'u', '�', '�', '�', 't', ' c', 'á', 'c', ' tokens', ' m', '�', '�', '�', 'i', ' t', '�', '�', '�', ' d', '�', '�', '�', ' li', '�', '�', '�', 'u', ' n', 'à', 'y', '.'], ['C', 'á', 'c', ' tokens', ' m', '�', '�', '�', 'i', ' n', 'à', 'y', ' l', 'à', ' c', 'á', 'c', ' tokens', ' �', '�', '�', '�', '�', 'c', ' th', '�', '�', ' tr', 'ong', ' ng', 'ô', 'n', ' ng', '�', '�', '�', ' Ti', '�', '�', '�', 'ng', ' Vi', '�', '�', '�', 't', '.'], ['Ch', '�', '�', '�', '�', 'ng', ' tr', '�', '�', 'n', 'h', ' s', '�', '�', '�', ' t', '�', '�', '�', ' �', '�', '�', '�', '�', 'ng', ' tr', 'í', 'ch', ' x', 'u', '�', '�', '�', 't', ' c', 'á', 'c', ' tokens', ' �', '�', '�', '�', '�', 'c', ' tr', '�', '�', 'ng', ' v', 'à', ' c', '�', '�', 'ng', ' t', '�', '�', '�', ' �', '�', '�', '�', '�', 'ng', ' th', 'ê', 'm', ' v', 'à', 'o', ' t', '�', '�', '�', ' �', '�', 'i', '�', '�', '�', 'n', '.'], 


New extract Tokens:
[' ', 'Hello,', 'you?', 'hello,', '你', '好', '很', '高', '兴', '见', '到', '你好很高兴见到你', 'こ', 'ん', 'に', 'ち', 'は', '、', '初', 'め', 'ま', 'し', 'て', '！', 'こんにちは、初めまして！', '안', '녕', '하', '세', '요', '만', '나', '서', '반', '갑', '습', '니', '다', '안녕하세요,', '만나서', '반갑습니다!', 'ạ', 'ẵ', 'ể', 'ắ', 'ầ', 'ư', 'Bạn', 'đã', 'sẵn', 'sàng', 'để', 'bắt', 'đầu', 'chưa', 'bạn', 'ộ', 'ụ', 'ề', 'ử', 'ữ', 'ệ', 'ế', 'Đây', 'là', 'một', 'ví', 'dụ', 'về', 'tiền', 'xử', 'lý', 'dữ', 'liệu', 'Tiếng', 'Việt.', 'đây', 'tiếng', 'việt.', 'ẽ', 'ự', 'ừ', 'ấ', 'ớ', 'Chúng', 'sẽ', 'thực', 'hiện', 'tách', 'từ', 'và', 'trích', 'xuất', 'các', 'tokens', 'mới', 'này.', 'chúng', 'ặ', 'Các', 'này', 'đặc', 'thù', 'trong', 'ngôn', 'ngữ', 'ơ', 'ũ', 'Chương', 'trình', 'tự', 'động', 'trưng', 'cũng', 'thêm', 'vào', 'điển.', 'chương']

tokenizer.get_vocab size: 50375

Tokens after extending vocabulary:
['Hello,', ' ', 'how', ' ', 'are', ' ', 'you?'], ['你好很高兴见到你'], ['こんにちは、初めまして！'], ['안녕하세요,', ' ', '만나서', ' ', '반갑습니다!'], ['Bạn', ' ', 'đã', ' ', 'sẵn', ' ', 'sàng', ' ', 'để', ' ', 'bắt', ' ', 'đầu', ' ', 'chưa', ' ', '?'], ['Đây', ' ', 'là', ' ', 'một', ' ', 'ví', ' ', 'dụ', ' ', 'về', ' ', 'tiền', ' ', 'xử', ' ', 'lý', ' ', 'dữ', ' ', 'liệu', ' ', 'Tiếng', ' ', 'Việt.'], ['Chúng', ' ', 'ta', ' ', 'sẽ', ' ', 'thực', ' ', 'hiện', ' ', 'tách', ' ', 'từ', ' ', 'và', ' ', 'trích', ' ', 'xuất', ' ', 'các', ' ', 'tokens', ' ', 'mới', ' ', 'từ', ' ', 'dữ', ' ', 'liệu', ' ', 'này.'], ['Các', ' ', 'tokens', ' ', 'mới', ' ', 'này', ' ', 'là', ' ', 'các', ' ', 'tokens', ' ', 'đặc', ' ', 'thù', ' ', 'trong', ' ', 'ngôn', ' ', 'ngữ', ' ', 'Tiếng', ' ', 'Việt.'], ['Chương', ' ', 'trình', ' ', 'sẽ', ' ', 'tự', ' ', 'động', ' ', 'trích', ' ', 'xuất', ' ', 'các', ' ', 'tokens', ' ', 'đặc', ' ', 'trưng', ' ', 'và', ' ', 'cũng', ' ', 'tự', ' ', 'động', ' ', 'thêm', ' ', 'vào', ' ', 'từ', ' ', 'điển.'], %                   

"""
