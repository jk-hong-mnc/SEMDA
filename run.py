# 라이브러리 불러오기
from py_interface import *
from ctypes import *
import threading
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np
import os

class SemanticEncoder(nn.Module):
    def __init__(self):
        super(SemanticEncoder, self).__init__()
        # 사전 학습된 ResNet50을 로드 (pretrained=True 사용)
        self.resnet50 = models.resnet50(weights=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 500)

    def forward(self, x):
        output = self.resnet50(x)
        return output

class SemanticDecoder(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(SemanticDecoder, self).__init__()
        self.Wq_view = nn.Linear(feature_dim, feature_dim, bias=False)
        self.Wk_view = nn.Linear(feature_dim, feature_dim, bias=False)
        self.Wv_view = nn.Linear(feature_dim, feature_dim, bias=False)
        self.Wq_content = nn.Linear(feature_dim, feature_dim, bias=False)
        self.Wk_content = nn.Linear(feature_dim, feature_dim, bias=False)
        self.Wv_content = nn.Linear(feature_dim, feature_dim, bias=False)
        self.fc = nn.Linear(feature_dim, num_classes)

    def compute_attention(self, F_concat, Wq, Wk, Wv):
        Q = Wq(F_concat)
        K = Wk(F_concat)
        V = Wv(F_concat)

        # 정규화 추가
        d_k = Q.shape[-1]  # feature_dim 크기
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, V)

    def forward(self, F_concat):
        O_view = self.compute_attention(F_concat, self.Wq_view, self.Wk_view, self.Wv_view)
        O_content = self.compute_attention(F_concat, self.Wq_content, self.Wk_content, self.Wv_content)
        O_combined = O_view * 0.3 + O_content * 0.7
        logits = self.fc(O_combined)
        return logits  # softmax 적용 X (CrossEntropyLoss가 내부적으로 처리)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR-10 평균 및 표준편차로 정규화
])

# encoder1 (for n0) 실행 코드: read image, write feature 
def run_encoder1(encoder1, encoder1_model, idx, device):
    with encoder1 as data:
        if data is None:
            return
        image = data.feat.image
        print(f"AI: encoder 1 received image {idx+1} from n0")
        image = np.frombuffer(image, dtype=np.uint8).reshape(32, 32, 3)
        image = transform(image)
        image = image.clone().detach().float().unsqueeze(0)
        image = image.to(device)
        feature = encoder1_model(image)
        feature_numpy = feature.detach().cpu().numpy().flatten().astype(np.float32)
        feature_c_float = (c_float * 500)(*feature_numpy.tolist())
        # print(feature_c_float[0])
        data.pred.feature = feature_c_float
        print(f"AI: encoder 1 sent feature for image {idx+1} to n0")

# encoder2 (for n1) 실행 코드: read image, write feature 
def run_encoder2(encoder2, encoder2_model, idx, device):
    with encoder2 as data:
        if data is None:
            return
        image = data.feat.image
        print(f"AI: encoder 2 received image {idx+1} from n1")
        image = np.frombuffer(image, dtype=np.uint8).reshape(32, 32, 3)
        image = transform(image)
        image = image.clone().detach().float().unsqueeze(0)
        image = image.to(device)
        feature = encoder2_model(image)
        feature_numpy = feature.detach().cpu().numpy().flatten().astype(np.float32)
        feature_c_float = (c_float * 500)(*feature_numpy.tolist())
        # print(feature_c_float[499])
        data.pred.feature = feature_c_float
        print(f"AI: encoder 2 sent feature for image {idx+1} to n1")

# decoder (for n2) 실행 코드: read feature, write predicted label
def run_decoder(decoder, decoder_model, idx, device):
    with decoder as data:
        if data is None:
            return
        feature_data = data.feat.feature
        print(f"AI: decoder received feature for image {idx+1} from n2")
        # print(feature_data[0])
        # print(feature_data[999])
        torch.set_printoptions(precision=18)
        features = torch.tensor(feature_data, dtype=torch.float32).unsqueeze(0)
        # print(features[0][0])
        # print(features[0][999])
        # print(features.shape)
        features = features.to(device)
        logits = decoder_model(features)
        _, predicted = torch.max(logits, 1)
        data.pred.pred_label[0] = int(predicted[0])
        print(f"AI: decoder sent predicted label for image {idx+1} to n2")

# encoder1, encoder2, decoder가 동시에 실행되도록 멀티스레딩
def process_image(encoder1, encoder1_model, encoder2, encoder2_model, decoder, decoder_model, idx, device):
    thread1 = threading.Thread(target=run_encoder1, args=(encoder1, encoder1_model, idx, device))
    thread2 = threading.Thread(target=run_encoder2, args=(encoder2, encoder2_model, idx, device))
    thread3 = threading.Thread(target=run_decoder, args=(decoder, decoder_model, idx, device))
    
    # 스레드 시작
    thread1.start()
    thread2.start()
    thread3.start()
    
    # 모든 스레드가 끝날 때까지 대기 -> 하나의 encoding, decoding 라운드가 다 끝나야 다음 라운드로 넘어가도록 하기 위함
    thread1.join()
    thread2.join()
    thread3.join()

# 인코더 데이터 구조 (ns-3와 동일)
class Encoder_input_image(Structure):
    _pack_ = 1
    _fields_ = [
        ('image', c_uint8 * 3072)  # 32x32x3 RGB image (3072 bytes)
    ]

class Encoder_output_feature(Structure):
    _pack_ = 1
    _fields_ = [
        ('feature', c_float * 500)  # 1000-byte feature vector
    ]

class Encoder_target(Structure):
    _pack_ = 1
    _fields_ = [
        ('target', c_uint8 * 1)  # Target values (not used for encoding)
    ]

# 디코더 데이터 구조 (ns-3와 동일)
class Decoder_input_feature(Structure):
    _pack_ = 1
    _fields_ = [
        ('feature', c_float * 1000)  # Feature length (1000 bytes)
    ]

class Decoder_output_feature(Structure):
    _pack_ = 1
    _fields_ = [
        ('pred_label', c_uint8 * 1)  # Predicted label
    ]

class Decoder_target(Structure):
    _pack_ = 1
    _fields_ = [
        ('target', c_uint8 * 1)  # Target values (not used for decoder)
    ]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images with encoder1, encoder2, and decoder.")
    parser.add_argument('--image_number', type=int, required=True, help="Number of images to process")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    image_number = args.image_number # 전송할 이미지 개수: python 실행시 입력해줘야함

    # 모델 초기화
    encoder1_model = SemanticEncoder()
    encoder2_model = SemanticEncoder()
    decoder_model = SemanticDecoder(feature_dim=1000, num_classes=10)

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    encoder1_model.to(device)
    encoder2_model.to(device)
    decoder_model.to(device)

    model_dir = "/home/ns3ai/ns-allinone-3.35/ns-3.35/scratch/SEMDA/weight/"
    encoder1_model.load_state_dict(torch.load(model_dir + "encoder1.txt", map_location=torch.device('cpu'), weights_only=True))
    encoder2_model.load_state_dict(torch.load(model_dir + "encoder2.txt", map_location=torch.device('cpu'), weights_only=True))
    decoder_model.load_state_dict(torch.load(model_dir + "decoder.txt", map_location=torch.device('cpu'), weights_only=True))

    encoder1_model.eval()
    encoder2_model.eval()
    decoder_model.eval()

    # encoder1 생성
    encoder1_mempool_key = 1001
    encoder1_mem_size = 30000
    exp = Experiment(encoder1_mempool_key, encoder1_mem_size, 'SEMDA', '../../') # 실험 환경 설정 - 한 번만 해줘도 됨
    exp.reset()
    encoder1 = Ns3AIDL(encoder1_mempool_key, Encoder_input_image, Encoder_output_feature, Encoder_target)

    # encoder2 생성
    encoder2_mempool_key = 1002
    encoder2 = Ns3AIDL(encoder2_mempool_key, Encoder_input_image, Encoder_output_feature, Encoder_target)

    # decoder 생성
    decoder_mempool_key = 2001
    decoder = Ns3AIDL(decoder_mempool_key, Decoder_input_feature, Decoder_output_feature, Decoder_target)

    ns3Settings = {'image_number': image_number}
    ns3_process = exp.run(setting=ns3Settings, show_output=True) # ns-3 실행: (예시: ./waf run "scratch/SEMDA/sim.cc --image_number=100")

    # 매 이미지에 대해 SEMDA 동작 수행
    for i in range(image_number):
        process_image(encoder1, encoder1_model, encoder2, encoder2_model, decoder, decoder_model, i, device)

    ns3_process.wait() # ns-3 프로세스 종료 대기
    del exp # 실험 환경 clear