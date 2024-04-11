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

    print("\ntoken_counts:",token_counts)

    # Lọc ra các tokens mới mà không có trong bộ từ vựng hiện tại của tokenizer
    existing_tokens = set(tokenizer.get_vocab().keys())

    new_tokens = [token for token in token_counts if token not in existing_tokens]

    return new_tokens

# Dữ liệu Tiếng Việt của bạn (ví dụ)

text_data = [
    "Hello, how are you?",
    "Bạn đã sẵn sàng để bắt đầu chưa ?",
    "Đây là một ví dụ về tiền xử lý dữ liệu Tiếng Việt.",
    "Chúng ta sẽ thực hiện tách từ và trích xuất các tokens mới từ dữ liệu này.",
    "Các tokens mới này là các tokens đặc thù trong ngôn ngữ Tiếng Việt.",
    "Chương trình sẽ tự động trích xuất các tokens đặc trưng và cũng tự động thêm vào từ điển."
]

# Gọi hàm preprocess_and_extract_tokens để trích xuất tokens mới
new_tokens = preprocess_and_extract_tokens(text_data)

print("\nTokens mới được trích xuất:")
print(new_tokens)

# Convert new tokens to strings
# Print tokens before added to vocabulary
print("\nTokens before extending vocabulary:")
for text in text_data:
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    str_tokens_before_add = [tokenizer.decode([token], skip_special_tokens=True) for token in text_tokens]
    print(str_tokens_before_add, end=", ")

# Add new tokens to tokenizer
tokenizer.add_tokens(new_tokens)

from transformers import GPT2LMHeadModel

# Load pretrained model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Resize model's embeddings
model.resize_token_embeddings(len(tokenizer))

# Print tokens after extending vocabulary
print("\n\nTokens after extending vocabulary:")
for text in text_data:
    text_tokens = tokenizer.encode(text, add_special_tokens=False)
    str_tokens_after_add = [tokenizer.decode([token], skip_special_tokens=True) for token in text_tokens]
    print(str_tokens_after_add, end=", ")


"""
token_counts: Counter({' ': 70, 't': 26, 'n': 25, 'c': 15, 'h': 12, 'g': 12, 'i': 11, 'o': 9, 'đ': 8, 's': 8, 'à': 8, 'tokens': 8, 'l': 7, 'các': 7, 'e': 6, 'r': 6, 'u': 6, 'từ': 6, 'v': 5, 'ệ': 5, 'á': 5, 'y': 4, '?': 4, 'm': 4, '.': 4, 'là': 4, 'dữ': 4, 'liệu': 4, 'k': 4, 'sẽ': 4, 'và': 4, 'trích': 4, 'xuất': 4, 'mới': 4, 'đặc': 4, 'tự': 4, 'động': 4, 'a': 3, 'ư': 3, 'ộ': 3, 'í': 3, 'd': 3, 'x': 3, 'ữ': 3, 'C': 3, 'ự': 3, 'ừ': 3, 'how': 2, 'are': 2, 'you?': 2, 'ể': 2, 'đã': 2, 'sẵn': 2, 'sàng': 2, 'để': 2, 'bắt': 2, 'đầu': 2, 'chưa': 2, 'ề': 2, 'T': 2, 'ế': 2, 'V': 2, 'một': 2, 'ví': 2, 'dụ': 2, 'về': 2, 'tiền': 2, 'xử': 2, 'lý': 2, 'Tiếng': 2, 'Việt.': 2, 'tiếng': 2, 'việt.': 2, 'ẽ': 2, 'ấ': 2, 'ớ': 2, 'ta': 2, 'thực': 2, 'hiện': 2, 'tách': 2, 'này.': 2, 'ặ': 2, 'này': 2, 'thù': 2, 'trong': 2, 'ngôn': 2, 'ngữ': 2, 'trình': 2, 'trưng': 2, 'cũng': 2, 'thêm': 2, 'vào': 2, 'điển.': 2, 'H': 1, ',': 1, 'w': 1, 'Hello,': 1, 'hello,': 1, 'B': 1, 'ạ': 1, 'ã': 1, 'ẵ': 1, 'b': 1, 'ắ': 1, 'ầ': 1, 'Bạn': 1, 'bạn': 1, 'Đ': 1, 'â': 1, 'ụ': 1, 'ử': 1, 'ý': 1, 'Đây': 1, 'đây': 1, 'ú': 1, 'Chúng': 1, 'chúng': 1, 'ù': 1, 'ô': 1, 'Các': 1, 'ơ': 1, 'ì': 1, 'ũ': 1, 'ê': 1, 'Chương': 1, 'chương': 1})

Tokens mới được trích xuất:
[' ', 'Hello,', 'you?', 'hello,', 'ạ', 'ẵ', 'ể', 'ắ', 'ầ', 'ư', 'Bạn', 'đã', 'sẵn', 'sàng', 'để', 'bắt', 'đầu', 'chưa', 'bạn', 'ộ', 'ụ', 'ề', 'ử', 'ữ', 'ệ', 'ế', 'Đây', 'là', 'một', 'ví', 'dụ', 'về', 'tiền', 'xử', 'lý', 'dữ', 'liệu', 'Tiếng', 'Việt.', 'đây', 'tiếng', 'việt.', 'ẽ', 'ự', 'ừ', 'ấ', 'ớ', 'Chúng', 'sẽ', 'thực', 'hiện', 'tách', 'từ', 'và', 'trích', 'xuất', 'các', 'tokens', 'mới', 'này.', 'chúng', 'ặ', 'Các', 'này', 'đặc', 'thù', 'trong', 'ngôn', 'ngữ', 'ơ', 'ũ', 'Chương', 'trình', 'tự', 'động', 'trưng', 'cũng', 'thêm', 'vào', 'điển.', 'chương']

Tokens before extending vocabulary:
['Hello', ',', ' how', ' are', ' you', '?'], ['B', '�', '�', '�', 'n', ' �', '�', 'ã', ' s', '�', '�', '�', 'n', ' s', 'à', 'ng', ' �', '�', '�', '�', '�', ' b', '�', '�', '�', 't', ' �', '�', '�', '�', '�', 'u', ' ch', '�', '�', 'a', '?'], ['�', '�', 'â', 'y', ' l', 'à', ' m', '�', '�', '�', 't', ' v', 'í', ' d', '�', '�', '�', ' v', '�', '�', '�', ' ti', '�', '�', '�', 'n', ' x', '�', '�', '�', ' l', '�', '�', ' d', '�', '�', '�', ' li', '�', '�', '�', 'u', ' Ti', '�', '�', '�', 'ng', ' Vi', '�', '�', '�', 't', '.'], ['Ch', 'ú', 'ng', ' ta', ' s', '�', '�', '�', ' th', '�', '�', '�', 'c', ' hi', '�', '�', '�', 'n', ' t', 'á', 'ch', ' t', '�', '�', '�', ' v', 'à', ' tr', 'í', 'ch', ' x', 'u', '�', '�', '�', 't', ' c', 'á', 'c', ' tokens', ' m', '�', '�', '�', 'i', ' t', '�', '�', '�', ' d', '�', '�', '�', ' li', '�', '�', '�', 'u', ' n', 'à', 'y', '.'], ['C', 'á', 'c', ' tokens', ' m', '�', '�', '�', 'i', ' n', 'à', 'y', ' l', 'à', ' c', 'á', 'c', ' tokens', ' �', '�', '�', '�', '�', 'c', ' th', '�', '�', ' tr', 'ong', ' ng', 'ô', 'n', ' ng', '�', '�', '�', ' Ti', '�', '�', '�', 'ng', ' Vi', '�', '�', '�', 't', '.'], ['Ch', '�', '�', '�', '�', 'ng', ' tr', '�', '�', 'n', 'h', ' s', '�', '�', '�', ' t', '�', '�', '�', ' �', '�', '�', '�', '�', 'ng', ' tr', 'í', 'ch', ' x', 'u', '�', '�', '�', 't', ' c', 'á', 'c', ' tokens', ' �', '�', '�', '�', '�', 'c', ' tr', '�', '�', 'ng', ' v', 'à', ' c', '�', '�', 'ng', ' t', '�', '�', '�', ' �', '�', '�', '�', '�', 'ng', ' th', 'ê', 'm', ' v', 'à', 'o', ' t', '�', '�', '�', ' �', '�', 'i', '�', '�', '�', 'n', '.'], 

Tokens after extending vocabulary:
['Hello,', ' ', 'how', ' ', 'are', ' ', 'you?'], ['Bạn', ' ', 'đã', ' ', 'sẵn', ' ', 'sàng', ' ', 'để', ' ', 'bắt', ' ', 'đầu', ' ', 'chưa', ' ', '?'], ['Đây', ' ', 'là', ' ', 'một', ' ', 'ví', ' ', 'dụ', ' ', 'về', ' ', 'tiền', ' ', 'xử', ' ', 'lý', ' ', 'dữ', ' ', 'liệu', ' ', 'Tiếng', ' ', 'Việt.'], ['Chúng', ' ', 'ta', ' ', 'sẽ', ' ', 'thực', ' ', 'hiện', ' ', 'tách', ' ', 'từ', ' ', 'và', ' ', 'trích', ' ', 'xuất', ' ', 'các', ' ', 'tokens', ' ', 'mới', ' ', 'từ', ' ', 'dữ', ' ', 'liệu', ' ', 'này.'], ['Các', ' ', 'tokens', ' ', 'mới', ' ', 'này', ' ', 'là', ' ', 'các', ' ', 'tokens', ' ', 'đặc', ' ', 'thù', ' ', 'trong', ' ', 'ngôn', ' ', 'ngữ', ' ', 'Tiếng', ' ', 'Việt.'], ['Chương', ' ', 'trình', ' ', 'sẽ', ' ', 'tự', ' ', 'động', ' ', 'trích', ' ', 'xuất', ' ', 'các', ' ', 'tokens', ' ', 'đặc', ' ', 'trưng', ' ', 'và', ' ', 'cũng', ' ', 'tự', ' ', 'động', ' ', 'thêm', ' ', 'vào', ' ', 'từ', ' ', 'điển.'], %  

"""
