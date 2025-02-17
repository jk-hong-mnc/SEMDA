// 헤더 파일 불러오기
#include "ns3/core-module.h"
#include "ns3/ns3-ai-module.h"
#include "ns3/log.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"


using namespace ns3;
NS_LOG_COMPONENT_DEFINE("SEMDA");

// 인코더를 위한 교환 데이터 구조 정의
struct Encoder_input_image
{
    uint8_t image[3072]; // ns-3에서 python으로 보내는 32*32*3 크기의 이미지 (1픽셀당 1바이트)
} Packed;

struct Encoder_output_feature
{
    float feature[500]; // python에서 ns-3로 보내는 2000 바이트 크기의 feature
} Packed;

struct Encoder_target
{
    uint8_t target; // 안 씀
} Packed;

int LabelBuffer[10000];

// 미리 binary화시킨 CIFAR-10 데이터셋에서 각 이미지를 하나씩 불러오는 함수: 한 이미지의 크기는 32*32*3 = 3072 bytes
bool LoadCifar10Image(const std::string &filename, Encoder_input_image &img, int imageIndex, int *LabelBuffer)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cout << "Failed to open file " << std::endl;
        return false;
    }
    uint8_t label;
    file.seekg(imageIndex * 3073, std::ios::beg);
    file.read(reinterpret_cast<char *>(&label), 1);
    LabelBuffer[imageIndex] = int(label);
    // std::cout << LabelBuffer[imageIndex] << std::endl;
    file.read(reinterpret_cast<char *>(img.image), 3072);
    file.close();
    return true;
}

// 인코더 정의
class Encoder : public Ns3AIDL<Encoder_input_image, Encoder_output_feature, Encoder_target>
{
public:
    Encoder(uint16_t id); // 인코더 클래스 자체
    void SendImage(const std::string &imagePath, int imageIndex, int sender, int *LabelBuffer); // ns-3 -> python으로 이미지 전송 함수
    void ReceiveFeature(float *featureBuffer, int sender); // python -> ns-3로 feature 전송 함수
};

// 인코더 클래스 정의: Ns3AIDL의 argument 3개가 해당 클래스의 feature, pred, target임! 우리는 feature를 image를 보낼 때 쓰고, pred를 추출 feature를 보낼 때 씀
Encoder::Encoder(uint16_t id) : Ns3AIDL<Encoder_input_image, Encoder_output_feature, Encoder_target>(id)
{
    SetCond(2, 0);
}

void Encoder::SendImage(const std::string &imagePath, int imageIndex, int sender, int *LabelBuffer)
{
    Encoder_input_image img;
    if (!LoadCifar10Image(imagePath, img, imageIndex, LabelBuffer))
    {
        NS_LOG_ERROR("Failed to load image from " << imagePath);
        return;
    }

    std::cout << "ns-3: n" << sender << " sent image to encoder " << sender+1 << std::endl;
    auto imgShared = FeatureSetterCond(); // 이미지를 write하기 위해 shared memory 접근
    std::memcpy(imgShared->image, img.image, sizeof(img.image)); // 이미지 write
    SetCompleted(); // shared memory 접근 해제
}

void Encoder::ReceiveFeature(float *featureBuffer, int sender)
{
    auto feat = PredictedGetterCond(); // 추출 feature를 read하기 위해 shared memory 접근
    std::memcpy(featureBuffer, feat->feature, 2000); // 추출 feature read
    GetCompleted(); // shared memory 접근 해제
    std::cout << "ns-3: n" << sender << " received feature from encoder " << sender+1 << std::endl;
}

// 디코더를 위한 교환 데이터 구조 정의
struct Decoder_input_feature
{
    float feature[1000]; // ns-3에서 python으로 보내는 4000 바이트 크기의 feature (n0가 보낸 2000 바이트 + n1가 보낸 2000 바이트)
} Packed;

struct Decoder_pred
{
    uint8_t pred_label[1]; // Python에서 ns-3로 보내는 최종 예측 라벨 (1바이트)
} Packed;

struct Decoder_target
{
    uint8_t target[1]; // 안 씀
} Packed;

// 디코더 정의
class Decoder : public Ns3AIDL<Decoder_input_feature, Decoder_pred, Decoder_target>
{
public:
    int n0_feature = 0; // n2가 n0의 feature를 수신했는지 나타내는 변수
    int n1_feature = 0; // n2가 n1의 feature를 수신했는지 나타내는 변수
    int decoding_done_n0 = 1; // n2가 n0의 feature를 처리하여 디코딩을 완료했는지 나타내는 변수
    int decoding_done_n1 = 1; // n2가 n1의 feature를 처리하여 디코딩을 완료했는지 나타내는 변수
    Decoder(uint16_t id); // 디코더 클래스 자체
    void StorePacket(Ptr<Socket> socket, float *receivedFeature); // n2가 n0, n1으로부터 feature 패킷을 받으면 자신의 버퍼에 저장하는 함수
    void SendFeature(float *receivedFeature); // n0, n1의 feature 패킷을 모두 받으면 n2가 자신의 버퍼에 있는 feature를 python에 전송하는 함수
    void ReceivePrediction(int *predicted_label, int imageIndex); // python으로부터 최종 예측 라벨을 n2가 받아오는 함수
};

// 디코더 클래스 정의: Ns3AIDL의 argument 3개가 해당 클래스의 feature, pred, target임! 우리는 feature를 feature를 보낼 때 쓰고, pred를 최종 예측 라벨을 보낼 때 씀
Decoder::Decoder(uint16_t id) : Ns3AIDL<Decoder_input_feature, Decoder_pred, Decoder_target>(id)
{
    SetCond(2, 0);
}

float receivedFeature[1000]; // n2가 n0,n1으로부터 전송받은 feature를 저장하는 버퍼 (총 2000바이트 크기)
Decoder *decoderPtr; // 디코더 객체 포인터
Encoder *encoderPtr1; // 인코더1 객체 포인터 (n0)
Encoder *encoderPtr2; // 인코더2 객체 포인터 (n1)

void Decoder::StorePacket(Ptr<Socket> socket, float *receivedFeature)
{
    Ptr<Packet> packet;
    uint32_t offset = 0; // receivedFeature라는 버퍼의 어느 인덱스부터 저장할지 결정하는 변수 (n0가 보낸 패킷을 받으면 0, n1이 보낸 패킷을 받으면 1000)
    
    while ((packet = socket->Recv()))
    {
        Ipv4Header ipv4Header;
        packet->PeekHeader(ipv4Header);
        Ipv4Address senderIp = ipv4Header.GetSource(); // 패킷으로부터 sender IP 획득

        if (senderIp == Ipv4Address("10.1.1.1")) // n0가 보낸 패킷
        {
            offset = 0; // offset 설정
            decoderPtr->n0_feature = 1; // n2가 n0의 feature를 수신했는지 나타내는 변수
            NS_LOG_UNCOND("ns-3: n2 received packet of size " << packet->GetSize() << " bytes from n0");
        }
        else if (senderIp == Ipv4Address("10.1.1.2")) // n1이 보낸 패킷
        {
            offset = 500; // offset 설정
            decoderPtr->n1_feature = 1; // n2가 n1의 feature를 수신했는지 나타내는 변수
            NS_LOG_UNCOND("ns-3: n2 received packet of size " << packet->GetSize() << " bytes from n1");
        }
        else
        {
            NS_LOG_ERROR("Unexpected sender IP address: " << senderIp);
            continue; 
        }

        packet->RemoveHeader(ipv4Header); // IP 헤더 제거
        
        uint8_t buffer[2000]; // 패킷 데이터를 임시로 저장할 버퍼
        packet->CopyData(buffer, 2000); // 패킷 데이터를 임시 버퍼에 복사

        // uint8_t 데이터를 float로 변환하여 receivedFeature 버퍼에 저장
        for (size_t i = 0; i < 500; ++i) // 1000 bytes를 4바이트씩 float로 변환
        {
            float value;
            memcpy(&value, buffer + (i * 4), sizeof(float)); // 4바이트씩 복사하여 float로 변환
            receivedFeature[offset + i] = value; // 변환된 값을 저장
        }
    }
}

void Decoder::SendFeature(float *receivedFeature)
{
    auto featShared = FeatureSetterCond();  // feature를 write하기 위해 shared memory 접근
    std::memcpy(featShared->feature, receivedFeature, 4000);  // feature write (2000바이트)
    SetCompleted();  // shared memory 접근 해제
    // NS_LOG_UNCOND ("C++ - Version after write process: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_id));
}

int predicted_label[10000];

void Decoder::ReceivePrediction(int *predicted_label, int imageIndex)
{
    auto prediction = PredictedGetterCond();  // 예측 라벨을 read하기 위해 shared memory 접근
    predicted_label[imageIndex] = int(prediction->pred_label[0]); // 예측 라벨 read (1바이트)
    std::cout << "ns-3: n2 received predicted label from decoder" << std::endl;
    std::cout << std::endl;
    GetCompleted(); // shared memory 접근 해제
}

void ReceivePacket(Ptr<Socket> socket)
{
    decoderPtr->StorePacket(socket, receivedFeature); // n2가 패킷을 받으면 버퍼에 저장
}

EventId Wait_feature_at_n2;

// n2가 n0, n1의 feature를 기다리는 함수, 1ms마다 실행되도록 스케줄링
void Wait_Feature_at_n2(int imageIndex, int image_number)
{
    // std::cout << "n2 alive" << std::endl;
    if (decoderPtr->n0_feature == 1 && decoderPtr->n1_feature == 1) // n0, n1의 feature를 모두 받았을 때 n2는 python 호출
    {
        Simulator::Cancel(Wait_feature_at_n2); // 스케줄링되어 있던 대기 함수 취소
        decoderPtr->SendFeature(receivedFeature); // python으로 2000바이트 크기의 feature 전송
        decoderPtr->ReceivePrediction(predicted_label, imageIndex); // python으로부터 최종 예측 라벨 획득
        // 디코더 관련 변수 설정
        decoderPtr->n0_feature = 0;
        decoderPtr->n1_feature = 0;
        decoderPtr->decoding_done_n0 = 1;
        decoderPtr->decoding_done_n1 = 1;
        if (imageIndex + 1 < image_number) // 아직 처리할 이미지가 남았다면
        {
            Wait_feature_at_n2 = Simulator::Schedule(MilliSeconds(1), &Wait_Feature_at_n2, imageIndex + 1, image_number); // 그 다음 이미지에 대한 대기 함수 스케줄링
        }
    }
    else // 아직 n0, n1의 feature를 모두 받지 못했다면
    {
        Wait_feature_at_n2 = Simulator::Schedule(MilliSeconds(1), &Wait_Feature_at_n2, imageIndex, image_number); // 해당 이미지에 대한 대기 함수 스케줄링
    }
}

EventId Wait_decoder_at_n0;
EventId Wait_decoder_at_n1;

// n0가 n2의 feature를 기다리는 함수, 1ms마다 실행되도록 스케줄링
void Wait_Decoder_at_n0(std::string imagePath, int imageIndex, int image_number, float *featureBuffer, Ptr<Socket> source, int *LabelBuffer)
{   
    // std::cout << "n0 alive" << std::endl;
    if (decoderPtr->decoding_done_n0 == 1) // n2의 디코딩이 끝났을 때 n0가 python 호출
    {
        Simulator::Cancel(Wait_decoder_at_n0); // 스케줄링되어 있던 대기 함수 취소
        decoderPtr->decoding_done_n0 = 0; // n0를 위한 디코딩이 안 되었으므로 0으로 설정

        encoderPtr1->SendImage(imagePath, imageIndex, 0, LabelBuffer); // python으로 이미지 전송
        encoderPtr1->ReceiveFeature(featureBuffer, 0); // python으로부터 추출 feature 획득

        int featureSize = 500;
        uint8_t* featureData = new uint8_t[featureSize * sizeof(float)];
        memcpy(featureData, featureBuffer, featureSize * (sizeof(float)));

        // 패킷 생성 및 전송
        Ptr<Packet> pkt = Create<Packet>(featureData, featureSize * (sizeof(float)));
        Ipv4Header ipv4Header;
        Ipv4Address srcIP = Ipv4Address("10.1.1.1");
        Ipv4Address dstIP = Ipv4Address("10.1.1.3");
        ipv4Header.SetSource(srcIP);
        ipv4Header.SetDestination(dstIP);
        pkt->AddHeader(ipv4Header);

        std::cout << "ns-3: n0 sent packet of size " << pkt->GetSize() << " bytes to n2" << std::endl;
        source->Send(pkt);
        
        if (imageIndex + 1 < image_number) // 아직 처리할 이미지가 남았다면
        {
            Wait_decoder_at_n0 = Simulator::Schedule(MilliSeconds(1), &Wait_Decoder_at_n0, imagePath, imageIndex+1, image_number, featureBuffer, source, LabelBuffer); // 그 다음 이미지에 대한 대기 함수 스케줄링
        }
    }
    else // 아직 n2의 디코딩이 끝나지 않았다면
    {
        Wait_decoder_at_n0 = Simulator::Schedule(MilliSeconds(1), &Wait_Decoder_at_n0, imagePath, imageIndex, image_number, featureBuffer, source, LabelBuffer); // 해당 이미지에 대한 대기 함수 스케줄링
    }
}

// n1가 n2의 feature를 기다리는 함수, 1ms마다 실행되도록 스케줄링
void Wait_Decoder_at_n1(std::string imagePath, int imageIndex, int image_number, float *featureBuffer, Ptr<Socket> source,  int *LabelBuffer)
{   
    // std::cout << "n1 alive" << std::endl;
    if (decoderPtr->decoding_done_n1 == 1)
    {
        Simulator::Cancel(Wait_decoder_at_n1);
        decoderPtr->decoding_done_n1 = 0;

        encoderPtr2->SendImage(imagePath, imageIndex, 1, LabelBuffer);
        encoderPtr2->ReceiveFeature(featureBuffer, 1);

        int featureSize = 500;
        uint8_t* featureData = new uint8_t[featureSize * sizeof(float)];
        memcpy(featureData, featureBuffer, featureSize * (sizeof(float)));

        // 패킷 생성 및 전송
        Ptr<Packet> pkt = Create<Packet>(featureData, featureSize * (sizeof(float)));
        Ipv4Header ipv4Header;
        Ipv4Address srcIP = Ipv4Address("10.1.1.2");
        Ipv4Address dstIP = Ipv4Address("10.1.1.3");
        ipv4Header.SetSource(srcIP);
        ipv4Header.SetDestination(dstIP);
        pkt->AddHeader(ipv4Header);

        std::cout << "ns-3: n1 sent packet of size " << pkt->GetSize() << " bytes to n2" << std::endl;
        source->Send(pkt);
        
        if (imageIndex + 1 < image_number)
        {
            Wait_decoder_at_n1 = Simulator::Schedule(MilliSeconds(1), &Wait_Decoder_at_n1, imagePath, imageIndex+1, image_number, featureBuffer, source, LabelBuffer);
        }
    }
    else
    {
        Wait_decoder_at_n1 = Simulator::Schedule(MilliSeconds(1), &Wait_Decoder_at_n1, imagePath, imageIndex, image_number, featureBuffer, source, LabelBuffer);
    }
}

// 메인 시뮬레이션 코드
int main(int argc, char *argv[])
{
    std::string imagePath_n0 = "/home/ns3/ns-allinone-3.35/ns-3.35/scratch/SEMDA/data/test_dataset.bin"; // n0 이미지 데이터
    std::string imagePath_n1 = "/home/ns3/ns-allinone-3.35/ns-3.35/scratch/SEMDA/data/test_dataset_transformed.bin"; // n1 이미지 데이터
    int imageIndex = 0; // 이미지 인덱스 카운터
    int image_number = 1; // 전송할 총 이미지 수

    CommandLine cmd;
    cmd.AddValue ("image_number","Number of images to transmit", image_number);
    cmd.Parse (argc, argv);

    Encoder encoder1(1001); // encoder1 (n0) 객체 생성 - memory ID: 1001
    Encoder encoder2(1002); // encoder2 (n1) 객체 생성 - memory ID: 1002
    Decoder decoder(2001); // decoder (n2) 객체 생성 - memory ID: 2001

    encoderPtr1 = &encoder1; // encoder1 포인터
    encoderPtr2 = &encoder2; // encoder2 포인터
    decoderPtr = &decoder; // decoder 포인터

    // 토폴로지 생성
    NodeContainer nodes; 
    nodes.Create(3);

    CsmaHelper csma;
    csma.SetChannelAttribute("DataRate", StringValue("1Gbps"));
    csma.SetChannelAttribute("Delay", TimeValue(MilliSeconds(0.1)));

    NetDeviceContainer devices = csma.Install(nodes);

    InternetStackHelper stack;
    stack.Install(nodes);

    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // n2의 sink 소켓 생성 후 ReceivePacket callback 함수 연결하여 수신 패킷 처리하도록 설정
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> sink = Socket::CreateSocket(nodes.Get(2), tid);
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), 8080);
    sink->Bind(local);
    sink->SetRecvCallback(MakeCallback(&ReceivePacket));

    Ptr<Socket> source_n0 = Socket::CreateSocket(nodes.Get(0), tid); // n0에 소켓 생성
    Ptr<Socket> source_n1 = Socket::CreateSocket(nodes.Get(1), tid); // n1에 소켓 생성
    InetSocketAddress remote = InetSocketAddress(interfaces.GetAddress(2), 8080); // n2에 소켓 생성

    source_n0->Connect(remote); // n0-n2 연결
    source_n1->Connect(remote); // n1-n2 연결

    float featureBuffer_n0[500]; // n0의 feature 버퍼: encoder1으로부터 받은 추출 feature를 저장하여 추후 n2로 보낼 패킷 전송에 활용함
    float featureBuffer_n1[500]; // n1의 feature 버퍼: encoder2으로부터 받은 추출 feature를 저장하여 추후 n2로 보낼 패킷 전송에 활용함
    // int LabelBuffer[10000];

    // std::cout << "n0 IP: " << interfaces.GetAddress(0) << std::endl;
    // std::cout << "n1 IP: " << interfaces.GetAddress(1) << std::endl;

    // encoder1, encoder2, decoder 대기함수 실행하여 실질적 SEMDA 동작 수행
    Wait_decoder_at_n0 = Simulator::Schedule(MilliSeconds(1), &Wait_Decoder_at_n0, imagePath_n0, imageIndex, image_number, featureBuffer_n0, source_n0, LabelBuffer);
    Wait_decoder_at_n1 = Simulator::Schedule(MilliSeconds(1), &Wait_Decoder_at_n1, imagePath_n1, imageIndex, image_number, featureBuffer_n1, source_n1, LabelBuffer);
    Wait_feature_at_n2 = Simulator::Schedule(MilliSeconds(1), &Wait_Feature_at_n2, imageIndex, image_number);

    Simulator::Run();
    Simulator::Destroy();

    // 최종 예측 라벨 출력
    std::cout << "=============== Predicted labels at decoder for " << image_number << " images ===============" << std::endl;
    std::cout << "[";
    for (int i = 0; i < image_number; i++)
    {
        std::cout << predicted_label[i];
        if (i < image_number - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]\n" << std::endl;

}
