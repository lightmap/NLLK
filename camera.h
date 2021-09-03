#pragma once
#include <algorithm>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
	#include<winsock2.h>
	#pragma comment(lib,"ws2_32.lib") //Winsock Library
	#define close closesocket
	#define socklen_t int
	#define ssize_t int
#endif

#include <sys/types.h>

#ifdef _linux
	#include <unistd.h>// close()?
	#include <netinet/in.h>// ip structs
	#include <sys/socket.h>
	#include <arpa/inet.h>
#endif

//std::string string_format(const std::string fmt,...){
//    int size = ((int)fmt.size())*2+50;
//    std::string str;
//    va_list ap;
//    while(1){
//        str.resize(size);
//        va_start(ap, fmt);
//        int n = vsnprintf((char*)str.data(), size, fmt.c_str(), ap);
//        va_end(ap);
//        if(n>-1 && n < size){
//            str.resize(n);
//            return str;
//        }
//        if(n>-1)
//            size=n+1;
//        else
//            size*=2;
//    }
//    return str;
//}

typedef int8_t  i8_t;// 1byte [-128 .. 127] -(2^7) .. 2^7-1
typedef uint8_t u8_t;// 1byte [0 .. 255] 2^8-1
typedef int16_t   i16_t;// 2byte [-32,768 .. 32,767] -(2^15) .. 2^15-1
typedef uint16_t  u16_t;// 2byte [0 .. 65,535] 2^16-1
typedef int32_t  i32_t;// 4byte [-2,147,483,648 .. 2,147,483,647] -(2^31) .. 2^31-1
typedef uint32_t  u32_t;// 4byte [0 .. 4,294,967,295] (2<<32)-1 == 2^32-1 == 1024^3*4 - 4Gb?
typedef int64_t   i64_t;// 8byte [-9,223,372,036,854,775,808 .. 9,223,372,036,854,775,807] -(2^63) .. 2^63-1
typedef uint64_t  u64_t;

const u32_t COMMAND_HEADER = 0x45534522u;
#define CAPABILITIES_OFFSET	(0xF0000000u)

#pragma pack(push)
#pragma pack(1)

struct RtpHeader {
	union {
		struct {
			u8_t
				csrcCount : 4,
				extension : 1,
				padding : 1,
				version : 2;
		};
		u8_t b1 = 0;
	};
	union {
		struct {
			u8_t
				payloadType : 7,
				marker : 1;
		};
		u8_t b2 = 0;
	};
	u16_t sequenceNumber = 0;
	u32_t timestamp = 0;
	u32_t ssrc = 0;
};

struct JpegHeader {
	u8_t typeSpecific = 0;
	u8_t offset_1 = 0;
	u8_t offset_2 = 0;
	u8_t offset_3 = 0;
	u8_t type = 0;
	u8_t quantization = 0;
	u8_t width8 = 0;
	u8_t height8 = 0;

	u32_t width() const {
		return width8 * 8;
	}
	u32_t height() const {
		return height8 * 8;
	}
	u32_t offset() const {
		return (offset_3 << 0) | (offset_2 << 8) | (offset_1 << 16);
	}
};

struct QuantizationHeader {
	u8_t mbz;
	u8_t precision;
	u16_t length;
};

#define HEADERS_SIZE (sizeof(RtpHeader) + sizeof(JpegHeader)/* + sizeof(QuantizationHeader)*/)

struct VideoFrameHeader {
//	DateTime timestamp;
	u32_t sequenceNumber;
	u32_t width;
	u32_t height;
	u32_t dataSize;
};
#pragma pack(pop)

struct WSASession{
	WSASession(){
	#ifdef _WIN32
        int ret = WSAStartup(MAKEWORD(2, 2), &data);
        if (ret != 0) std::cout << "WSAStartup Failed : " << WSAGetLastError() << std::endl;
	#endif
	}
    ~WSASession(){
	#ifdef _WIN32
		WSACleanup();
	#endif
	}
	#ifdef _WIN32
		WSAData data;
	#endif
};

const unsigned char packet_start[] = {
  0x22, 0x45, 0x53, 0x45, 0x10, 0x02, 0x20, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x14, 0x06, 0x00, 0xf0,
  0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
};

const unsigned char packet_stop[] = {
    0x22, 0x45, 0x53, 0x45, 0x10, 0x02, 0x20, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x14, 0x06, 0x00, 0xf0,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
  };

char packet_gain[] = {
  0x22, 0x45, 0x53, 0x45, 0x41, 0x02, 0x22, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x20, 0x08, 0x00, 0xf0,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
};

char packet_gain1[] = {
  0x22, 0x45, 0x53, 0x45, 0x45, 0x02, 0x22, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x20, 0x08, 0x00, 0xf0,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
};

char packet_gain2[] = {
  0x22, 0x45, 0x53, 0x45, 0x46, 0x02, 0x20, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x20, 0x08, 0x00, 0xf0,
  0x00, 0x00, 0x00, 0x00, 0x96, 0x00, 0x00, 0x82,
  0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
};


struct UDPSocket{
	UDPSocket(){
		sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == -1)
        {
            std::cout << "Error opening socket : "/* << WSAGetLastError()strerror()*/ << std::endl;
        }

	}
    ~UDPSocket(){close(sock);}
	//std::string input(buffer);
	//std::reverse(std::begin(input), std::end(input));
	//Socket.SendTo(add, input.c_str(), input.size());
	void SendTo(const std::string &address, unsigned short port, const char *buffer, int len, int flags = 0){
		sockaddr_in add;
		add.sin_family = AF_INET;
        std::cout << "SendTo " << address.c_str() << std::endl << "[";
        for (int i = 0; i < len; i++) std::cout << buffer[i];
        std::cout << "] (" << len << ")" << std::endl;
        //if (inet_addr(address.c_str()) == INADDR_NONE || inet_addr(address.c_str()) == INADDR_ANY) signal_handler(2);
		add.sin_addr.s_addr = inet_addr(address.c_str());
		add.sin_port = htons(port);
		int ret = sendto(sock, buffer, len, flags, (sockaddr *)(&add), sizeof(add));
        if (ret < 0)  std::cout << "sendto failed : " /*<< WSAGetLastError()*/ << std::endl;
	}
	//void SendTo(sockaddr_in &address, const char *buffer, int len, int flags = 0){
	//	int ret = sendto(sock, buffer, len, flags, (sockaddr *)(&address), sizeof(address));
	//	if (ret < 0)
	//		cout << "sendto failed : " << WSAGetLastError() << endl;
	//}
	u32_t RecvFrom(char *buffer, int len, int flags = 0){
		sockaddr_in from;
        socklen_t size = sizeof(from);
		//cout << "wait recive buf size=" << len << "\n";
        ssize_t ret = recvfrom(sock, buffer, len, flags, (sockaddr *)(&from), &size);
		//cout << "ret=" << ret << ", recived from=" << size << " adr=" << inet_ntoa(from.sin_addr) << endl;
		//for (int i = 0; i < size; i++) cout << buffer[i];
		//cout << endl;
		if (ret < 0)
            std::cout << "recvfrom failed : " /*<< WSAGetLastError()*/ << std::endl;
		// make the buffer zero terminated
		//buffer[ret] = 0;
		return ret;
	}
	void Bind(unsigned short port){
		sockaddr_in add;
		add.sin_family = AF_INET;
		add.sin_addr.s_addr = htonl(INADDR_ANY);
		add.sin_port = htons(port);

		int ret = ::bind(sock, (sockaddr *)(&add), sizeof(add));
		if (ret < 0)
            std::cout << "Bind failed : " /*<< WSAGetLastError()*/ << std::endl;

		int buf_size = (1 << 20) * 50;
		int res = setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&buf_size, 4);
        std::cout << "setsockopt : res = " << res << " sizeof(buf_size) = " << sizeof(buf_size) << " buf_size = " << buf_size << std::endl;
	}
    /*SOCKET*/int sock;
};

#define PORT 50604
#define CMD_PORT 50700
#define IP "192.168.1.34"
#define BUFFER_SIZE (1<<14)

#define HEADER			(0x00)//4
#define NUMBER			(0x04)//2
#define OPERATION		(0x06)//6
#define ADDRESS			(0x0C)//8
#define DATA			(0x14)//8
#define CRC_			(0x1C)//4
const u8_t ReadRequest = 0x02;

constexpr u32_t IMG_SZ = 2048 * 2048;// *12;// 5065984u;

#define mmax(a,b)(((a)>(b))?(a):(b))

struct Camera {

	const u32_t frame_w = 2048;
	const u32_t frame_h = 2048;
	const u32_t frame_size = frame_w * frame_h;

	char buffer[BUFFER_SIZE];
	char *image;
	char cmd[32];

	u32_t collected;

	WSASession session;
	UDPSocket sock;

	Camera(bool skip = false) {
		if (skip == false) {
			//try {
			std::cout << "frame width = " << frame_w << std::endl;
			sock.Bind(PORT);

			memset(cmd, 0, 32);
			*((u32_t*)&cmd[HEADER]) = COMMAND_HEADER;
			*((u8_t*)&cmd[OPERATION]) &= 0x0F;
			*((u8_t*)&cmd[OPERATION]) |= (2 << 4);
			//*((u16_t*)&cmd[NUMBER]) = 0;// num; ?
			*((u8_t*)&cmd[OPERATION]) &= 0xF0;
			*((u8_t*)&cmd[OPERATION]) |= (ReadRequest & 0x0F);
			*((u32_t*)&cmd[ADDRESS]) = 0;// addr; ?
			//*((u32_t*)&cmd[DATA]) = 0;//

			sock.SendTo(IP, CMD_PORT, (const char*)packet_start, 32);
			collected = 0;
			int cnt = 0;
			image = new char[IMG_SZ];
			memset(image, 0, IMG_SZ);
			int packets = 0;
			while (1) {
				u16_t q_off = 0;

				const u32_t recieved = sock.RecvFrom(buffer, sizeof(buffer));
				if (recieved < (1 << 13)) { std::cout << "init: unexpected size " << recieved << "\n"; continue; /*signal_handler(100);*/ }

				const JpegHeader *jpg = (JpegHeader*)(&buffer[0] + sizeof(RtpHeader));
				const RtpHeader *rtp = (RtpHeader*)(&buffer[0]);
				u32_t off = jpg->offset();
				if (off == 0) {
					const QuantizationHeader *Q = (QuantizationHeader*)(&buffer[0] + HEADERS_SIZE);
					//cout << string_format("len=%X mbz=%i pres=%i",Q->length, Q->mbz,Q->precision).c_str() << endl;
					q_off = sizeof(*Q) + ((Q->length & 0xFF00) >> 8) | ((Q->length & 0x00FF) << 8);
					std::cout << "q_off = " << q_off << std::endl;
					std::cout << "framesize = " << collected << " == " << frame_size << " packets = " << packets << std::endl;//5074812 == 5065984
					packets = 0;
					q_off = 0;
					if (!saveFrame())
						break;

					//continue;
				}
				u32_t data_size = (recieved - HEADERS_SIZE) - q_off;
				if (off + data_size >= IMG_SZ)// { cout << (off + data_size - IMG_SZ) << endl; data_size = std::max(IMG_SZ - off, 0u); signal_handler(200); }
				{
					data_size = static_cast<u32_t> (mmax(static_cast<i64_t>(IMG_SZ) - static_cast<i64_t>(off), i64_t(0)));
					//cout << "out of image" << std::endl;
				}

				memcpy(image + off, buffer + q_off + HEADERS_SIZE, data_size);
				collected += data_size;
				packets++;
			}
			//}ERROR_CATCHER
		}
		//signal_handler(2);
	}

	bool saveFrame() {
		static size_t cnt = 0;
		
		collected = 0;
		if (cnt++ >= 2) return false;
		FILE *in = fopen(string_format("img_%i", cnt).c_str(), "wb");
		fwrite(image, 1, IMG_SZ, in);
		fclose(in);

		return true;
	}

	void setGain(byte val) {
		std::cout << "set gain = " << (int)val << std::endl;
		packet_gain2[20] = val;
		sock.SendTo(IP, CMD_PORT, (const char*)packet_gain2, 32);
		//const u32_t recieved = sock.RecvFrom(buffer, sizeof(buffer));
		//std::cout << "RecvFrom = " << recieved << std::endl;
	}

	u32_t Grab(char *image_buf) {
		collected = 0;
        bool start = false;
        //std::cout << "-------- start grabbing" << std::endl;
		while (1) {
			u16_t q_off = 0;
			
			const u32_t recieved = sock.RecvFrom(buffer, sizeof(buffer));
            if (recieved < (1 << 13)) { std::cout << "grab: unexpected size " << recieved << "\n"; /*signal_handler(100);*/ }
			
			const JpegHeader *jpg = (JpegHeader*)(&buffer[0] + sizeof(RtpHeader));
			const u32_t off = jpg->offset();
			
			if (off == 0) {
				const QuantizationHeader *Q = (QuantizationHeader*)(&buffer[0] + HEADERS_SIZE);
				//cout << string_format("len=%X mbz=%i pres=%i",Q->length, Q->mbz,Q->precision).c_str() << endl;
				q_off = sizeof(*Q) + ((Q->length & 0xFF00) >> 8) | ((Q->length & 0x00FF) << 8);
				q_off = 0;
                if(start)break;
                start = true;
                //std::cout << "q_off = " << q_off << std::endl;
			}
			
			u32_t data_size = (recieved - HEADERS_SIZE) - q_off;
			if (off + data_size >= IMG_SZ)// { cout << (off + data_size - IMG_SZ) << endl; data_size = std::max(IMG_SZ - off, 0u); signal_handler(200); }
			{
				//data_size = static_cast<u32_t> (std::max(static_cast<u32_t> (IMG_SZ - off), 0u));
                data_size = static_cast<u32_t> (mmax(static_cast<i64_t> (IMG_SZ) - static_cast<i64_t> (off), (i64_t)0));
				//cout << "out of image" << std::endl;
			} 

            if(start)memcpy(image_buf + off, buffer + q_off + HEADERS_SIZE, data_size);
			collected += data_size;
			
            //if (!off) break;
		}
        //std::cout << "-------- end" << std::endl;
		return collected;
	}
	~Camera() {
		sock.SendTo(IP, CMD_PORT, (const char*)packet_stop, 32);
	}// signal_handler(3);
};

