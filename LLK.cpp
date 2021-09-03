#include <iostream>

#ifdef _WIN32
	#include <conio.h>
#else
	#include <curses.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>  // cv::Canny()
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>//bg
#include <iostream>
#include <numeric>
//#define NO_MIN_MAX 
//#include <debugapi.h>

#include "yolo.h"

#define DBOUT( s )            \
{                             \
   std::wostringstream os_;    \
   os_ << s;                   \
	std::cout << s << "\n"; \
}
   ///OutputDebugString( os_.str().c_str() );  
int thresh = 235;
cv::Mat src, thresh_img, gray;

#include <cstdarg>//va_start
std::string string_format(const std::string fmt,...){
    int size = ((int)fmt.size())*2+50;
    std::string str;
    va_list ap;
    while(1){
        str.resize(size);
        va_start(ap, fmt);
        int n = vsnprintf((char*)str.data(), size, fmt.c_str(), ap);
        va_end(ap);
        if(n>-1 && n < size){
            str.resize(n);
            return str;
        }
        if(n>-1)
            size=n+1;
        else
            size*=2;
    }
    return str;
}

const char* get_time_name() {
	time_t rawtime;
	struct tm *timeinfo;
	static char buffer[128];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	//strftime(buffer, sizeof(buffer), "%d-%m-%Y %I:%M:%S", timeinfo);
	strftime(buffer, sizeof(buffer), "%Y_%m_%d_%I_%M_%S\0", timeinfo);
	return buffer;
}
//#include <windows.h>
//std::vector<std::string> get_all_files_names_within_folder(std::string folder, bool only_folders = false)
//{
//	std::vector<std::string> names;
//	std::string search_path = folder + "/*";
//	WIN32_FIND_DATAA fd;
//	HANDLE hFind = ::FindFirstFileA(search_path.c_str(), &fd);
//	if (hFind != INVALID_HANDLE_VALUE) {
//		do {
//			// read all (real) files in current folder, delete '!' read other 2 default folder . and ..
//			//if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)){
//			if ((only_folders == false && !(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) ||
//				(only_folders == true && (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fd.cFileName[0] != '.')) {
//				names.push_back(fd.cFileName);
//			}
//		} while (::FindNextFileA(hFind, &fd));
//		::FindClose(hFind);
//	}
//	return names;
//}

struct textData {
	FILE *in;
	textData() {
		in = fopen(string_format("%s.txt", get_time_name()).c_str(), "wt");
		if (in == nullptr) { return; }
	}
	void print(const char *str) {
		if (in == nullptr)return;
		fprintf(in, "%s\n", str);
	}
	~textData() { if (in)fclose(in); }
};

textData TD;

// make even > 3
int ks(int a) {
	if (a < 3) return 3;
	return (a & ~1) + 1;
}

void get_markers(int, void*) {
	cv::Mat src_clone = src.clone();
	//cv::cvtColor(src, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	gray = src.clone();
	cv::GaussianBlur(gray, gray, cv::Size(ks(5), ks(5)), 0);// remove noise

	cv::threshold(gray, thresh_img, std::min(thresh, 255), 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(thresh_img, contours, cv::RetrievalModes::RETR_LIST, cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);
	cv::drawContours(src_clone, contours, -1, cv::Scalar(255, 0, 0), 1);

	cv::RNG rng(12345);
	char sbuf[255];
	int cnt = 0;
	for (int i = 0; i < contours.size(); i++) {
		if (contours[i].size() < 4) continue;
		cv::Rect rec = boundingRect(contours[i]);
		//if (std::abs(rec.width - rec.height) > std::min(rec.width, rec.height)) {
		//	continue;
		//}
		float area = cv::contourArea(contours[i]);
		if (area > 10000 || area < 50) continue;
		cnt++;
		cv::Point cp = cv::Point(rec.x + rec.width / 2, rec.y + rec.height / 2);
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255), 64);
		snprintf(sbuf, sizeof(sbuf), "%i", cnt);
		putText(src_clone, sbuf, cp, cv::FONT_HERSHEY_PLAIN, 3, color, 4);
	}

	cv::imshow("ship", src_clone);
}

typedef unsigned char byte;

void cross_corelate(byte *x, byte *y, byte n, byte *r) {
	// mean
	int mx = 0;
	int my = 0;
	for (int i = 0; i < n; i++) {
		mx += x[i];
		my += y[i];
	}
	mx /= n;
	my /= n;
	// denominatior
	float sx = 0;
	float sy = 0;
	for (int i = 0; i < n; i++) {
		sx += (x[i] - mx) * (x[i] - mx);
		sy += (y[i] - my) * (y[i] - my);
	}
	float denom = sqrt(sx*sy);
	// correlation
	int maxdelay = n/2;
	int cnt = 0;
	for (int delay = -maxdelay; delay < maxdelay; delay++) {
		float sxy = 0;
		for (int i = 0; i < n; i++) {
			int j = i + delay;
			if (j < 0 || j >= n)
				continue;
			else
				sxy += (x[i] - mx) * (y[j] - my);
			/* Or should it be (?)
			if (j < 0 || j >= n)
			   sxy += (x[i] - mx) * (-my);
			else
			   sxy += (x[i] - mx) * (y[j] - my);
			*/
		}
		r[cnt++] = (sxy / denom)*255;
	}
}

void corelation(byte *x, byte *y, byte n, byte *r) {
	float sx=0, sy=0, sxy=0;
	float sqx = 0, sqy = 0;
	for (int i = 0; i < n; i++) {
		sx += x[i];
		sy += y[i];
		sxy += x[i] * y[i];
		sqx += x[i] * x[i];
		sqy += y[i] * y[i];
	}
	float fn = n;
	float corr = (fn*sxy - sx * sy) / sqrt((fn*sqx - sx * sx) * (fn*sqy - sy * sy));
	//float corr = (sx * sy) / sqrt((sx * sx) * (sy * sy));
	corr = corr < 0 ? 0 : corr;
	//r[0] = corr * 255;
	corr *= ((float)x[n / 2]/255);
	r[0] = (corr>0.85)?corr * 255: corr*32;
	//r[0] += x[0] / 10;
}

void Detect(cv::Mat &gray) {
	cv::Mat grayT;
	cv::transpose(gray, grayT);
	cv::Mat rez = grayT.clone();
	//int patt_sz = 40;
	byte b = 0;
	byte k = 255;
	int patt_sz = 11;
	byte patt[] = { b,b,b,k,k,k,k,k,b,b,b};// 11px
	//byte patt[] = { b,b,b,b,b,k,k,k,k,k, k,k, k,k,k,k,k,b,b,b,b,b };// 22px
	//byte patt[] = { b,b,b,b,b,b,b,b,b,b, k,k,k,k,k,k,k,k,k,k, k,k,k,k,k,k,k,k,k,k, b,b,b,b,b,b,b,b,b,b };// 40px
	for (int h = 0; h < grayT.rows; h++) {
		byte *pix = grayT.ptr<byte>(h);
		byte *r_pix = rez.ptr<byte>(h);
		for (int w = 0; w < grayT.cols - patt_sz; w++) {
			//cross_corelate(pix+w, patt, patt_sz, r_pix+w); 
			corelation(pix + w, patt, patt_sz, r_pix + w);
			//if (w < 40)r_pix[w] = patt[w];
		}
	}
	cv::transpose(rez, gray);

	cv::imshow("correlation1D", gray);
	cv::imwrite("correlation1D.png", gray);
	//cv::threshold(gray, gray, std::min(170, 255), 255, cv::THRESH_BINARY);
}

#include <thread>
bool thread_busy = false;
bool use_detector = false;
bool use_yolo = false;
bool recording = false;
bool use_equalize = false;
bool record_raw = true;
bool use_experts_cam = false;

cv::Mat barr;
cv::Mat barr2;
cv::Mat barr3;
cv::Mat barr4;
int tx, ty;
float tcorr;

void templateMatch(cv::Mat &gray,const cv::Mat &templ) {
    thread_busy = true;
	cv::Mat out = gray.clone();
	tcorr = 0;
	
	bool needswap = gray.size().height < templ.size().height || gray.size().width < templ.size().width;
	if (needswap)return;

	cv::matchTemplate(gray, templ, out, cv::TemplateMatchModes::TM_CCOEFF_NORMED);//TM_CCORR_NORMED);//
	
    //cv::imshow("correlation2D", out);
    //gray = out.clone();
    //return;
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    cv::Point matchLoc;
    int cnt = 0;
    //char sbuf[255];
    std::string sbuf;
    do {
        cv::minMaxLoc(out, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        matchLoc = maxLoc;

        byte Iw = gray.at<byte>(cv::Point(matchLoc.x+templ.cols/2, matchLoc.y + templ.rows/2));
        byte Ib = gray.at<byte>(cv::Point(matchLoc.x+templ.cols/2, matchLoc.y + templ.rows/10));

        /// Fill the detected location with a rectangle of zero
        cv::rectangle(out, cv::Point(matchLoc.x - barr.cols * 2, matchLoc.y - barr.rows * 2), cv::Point(matchLoc.x + barr.cols * 2, matchLoc.y + barr.rows * 2), cv::Scalar::all(0), -1);

        float ncorr = maxVal * (float)Iw / 255;
        ncorr *= 1.0f - (float)Ib / 255;

        if (ncorr > 0.20) {
			if (matchLoc.x + templ.cols > gray.cols || matchLoc.y + templ.rows > gray.rows) {
				continue;
			}
            cv::rectangle(gray, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(128), 1, 8);
            //sbuf = string_format("%i (%.2f) [%i]", cnt, ncorr, Ib);
			////////sbuf = string_format("%i (%.2f) [%i %i]", cnt, ncorr, matchLoc.x, matchLoc.y);
   ////////         putText(gray, sbuf, matchLoc, cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar::all(128), 4);
			tx = matchLoc.x;
			ty = matchLoc.y;
			tcorr = ncorr;
            cnt++;
            //std::cout << "point correlation=" << maxVal << std::endl;
            if (cnt == 3)break;
        }
        return;
    } while (maxVal > 0.50);

	//cv::imshow("template", gray); //gray = out.clone();
    //out *= 255;
    //cv::imwrite("correlation2D.png", out);
    //cv::imwrite("labels2D.png", gray);
    thread_busy = false;
    //gray = gray;
    //return out;//cv::imshow("frame", out);//gray);
}

cv::Mat Rotate(cv::Mat img, double angle) {
	double offsetX, offsetY;
	double width = img.size().width;
	double height = img.size().height;
	cv::Point2d center = cv::Point2d(width / 2, height / 2);
	cv::Rect bounds = cv::RotatedRect(center, img.size(), angle).boundingRect();
	cv::Mat resized = cv::Mat::zeros(bounds.size(), img.type());
	offsetX = (bounds.width - width) / 2;
	offsetY = (bounds.height - height) / 2;
	cv::Rect roi = cv::Rect(offsetX, offsetY, width, height);
	img.copyTo(resized(roi));
	center += cv::Point2d(offsetX, offsetY);
	cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::warpAffine(resized, resized, M, resized.size());
	return resized;
}

void contrastStrech(cv::Mat &img){
    double pmin,pmax;
    cv::minMaxLoc(img, &pmin, &pmax);
    for(int j=0;j<img.rows;j++){
        uchar *dataj = img.ptr<uchar>(j);
        for(int i=0;i<img.cols;i++){
            dataj[i]=(dataj[i]-pmin)*255/(pmax-pmin);
        }
    }
}

cv::Mat equalizeIntensity(const cv::Mat& inputImage)
{
	if (inputImage.channels() >= 3) {
		cv::Mat ycrcb;

		//cv::cvtColor(inputImage, ycrcb, cv::COLOR_BGR2Lab);//cv::COLOR_BGR2YCrCb);
		ycrcb = inputImage;

		std::vector<cv::Mat> channels;
		cv::split(ycrcb, channels);

		cv::equalizeHist(channels[0], channels[0]);

		cv::Mat result;
		cv::merge(channels, ycrcb);

		result = ycrcb;
		//cv::cvtColor(ycrcb, result, cv::COLOR_Lab2BGR);// cv::COLOR_YCrCb2BGR);

		return result;
	}
	return cv::Mat();
}
#include "camera.h"

Camera cam(use_experts_cam==false);
std::mutex lock;
std::thread job;
YOLO yo;
cv::VideoWriter *writer = nullptr;
cv::VideoCapture cap;

void fixBox(cv::Rect &r, cv::Mat mat) {
	if (r.x + yo.roi_1.width >= mat.cols)r.width = mat.cols - r.x;
	if (r.y + yo.roi_1.height >= mat.rows)r.height = mat.rows - r.y;
	if (r.x < 0)r.x = 0;
	if (r.y < 0)r.y = 0;
}

struct BGSub {
	cv::Ptr<cv::BackgroundSubtractor> pBackSub;
	cv::Mat fgMask;
	BGSub() {
		//pBackSub = cv::createBackgroundSubtractorMOG2();
		pBackSub = cv::createBackgroundSubtractorKNN();
	}
	void update(cv::Mat frame) {
		if (frame.empty())return;
		pBackSub->apply(frame, fgMask);
	}
};

cv::Rect2f load(int n) {
	FILE *in = fopen(string_format("imgs\\%i.txt", n).c_str(), "rt");
	int tmp;
	float x, y, w, h;
	fscanf(in, "%i %f %f %f %f\n", &tmp, &x, &y, &w, &h);
	fclose(in);
	return cv::Rect2f(x, y, w, h);
}
void testRect(int i, const cv::Mat &img) {
	cv::Rect2f r = load(i);
	int w = img.size().width;
	int h = img.size().height;
	int hw = r.width*w / 2;
	int hh = r.height*h / 2;
	cv::Rect R = cv::Rect(r.x*w - hw, r.y*h - hh, r.width*w, r.height*h);
	cv::rectangle(img, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
	cv::imwrite(string_format("imgs\\%i_test.png", i), img);
}

cv::Point2f Convert(const cv::Point2f & p, const cv::Mat & t)
{
	float x = p.x*t.at<double>(0, 0) + p.y*t.at<double>(0, 1) + t.at<double>(0, 2);
	float y = p.x*t.at<double>(1, 0) + p.y*t.at<double>(1, 1) + t.at<double>(1, 2);
	return cv::Point2f(x, y);
}
int mx, my;
void mouse_callback(int  event, int  x, int  y, int  flag, void *param)
{
	if (event == cv::EVENT_MOUSEMOVE) {
		//cout << "(" << x << ", " << y << ")" << endl;
		mx = x;
		my = y;

	}
}
cv::Rect RotateRect(cv::Rect r, cv::Size sz, float angle) {
	std::vector<cv::Point2f> pts(4), pts2(4);
	pts[0] = cv::Point(r.x,			  r.y);
	pts[1] = cv::Point(r.x + r.width, r.y);
	pts[2] = cv::Point(r.x,			  r.y + r.height);
	pts[3] = cv::Point(r.x + r.width, r.y + r.height);
	//cv::Mat RM = cv::getRotationMatrix2D(cv::Point2f(sz.width, sz.height) / 2, angle, 1);

	cv::Point2f center(0,0);
	cv::Mat rot = cv::getRotationMatrix2D(center,-angle, 1.0);
	if (angle == 90) {
		rot.at<double>(0, 2) =  sz.height;
		rot.at<double>(1, 2) = 0;
	}
	if (angle == 180) {
		rot.at<double>(0, 2) = sz.width;
		rot.at<double>(1, 2) = sz.height;
	}
	if (angle == 270) {
		rot.at<double>(0, 2) = 0;
		rot.at<double>(1, 2) = sz.width;
	}
	//rot.at<double>(1, 2) = -(sz.width - sz.height);
	//cv::Rect bbox = cv::RotatedRect(center, cv::Size2f(r.width, r.height), angle).boundingRect();
	//return bbox;
	//rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	//rot.at<double>(1, 2) += bbox.h2.0 - center.y;
	//
	cv::transform(pts, pts2, rot);
	//pts2[0] = Convert(pts[0], rot);
	//pts2[1] = Convert(pts[1], rot);
	//pts2[2] = Convert(pts[2], rot);
	//pts2[3] = Convert(pts[3], rot);
	return cv::boundingRect(pts2);
}


enum ROT_CODE {
	R0 = 0,
	R90 = 1,
	R180 = 2,
	R270 = 3,
};
void WriteRect_YOLO(cv::Rect r, cv::Size s, int f_cnt, const cv::Mat &img, ROT_CODE angle_code) {
	cv::Mat rimg;
	cv::Rect R;
	switch (angle_code) {
		case R0: {rimg = img.clone();R = r;break;}
		case R90: {
			cv::rotate(img, rimg, cv::ROTATE_90_CLOCKWISE);
			R = RotateRect(r, img.size(), 90);break;
		}
		case R180: {cv::rotate(img, rimg, cv::ROTATE_180);R = RotateRect(r, img.size(), 180);break;}
		case R270: {cv::rotate(img, rimg, cv::ROTATE_90_COUNTERCLOCKWISE);R = RotateRect(r, img.size(), 270);break;}
	}
	r = R;
	std::string txt = string_format("imgs\\%i.txt", f_cnt);
	{
		FILE *in = fopen(txt.c_str(), "wt");
		float pw = s.width;
		float ph = s.height;

		//float a = float(r.x) / w, b = float(r.y) / h, c = float(r.width) / w, d = float(r.height) / h; RotateRect

		//if (a > 1.1f) { DebugBreak(); }
		//if (b > 1.1f) { DebugBreak(); }
		//if (c > 1.1f) { DebugBreak(); }
		//if (d > 1.1f) { DebugBreak(); }
		float dw = 1. / pw;
		float dh = 1. / ph;
		float xmin = min(r.x, r.x + r.width);
		float xmax = max(r.x, r.x + r.width);
		float ymin = min(r.y, r.y + r.height);
		float ymax = max(r.y, r.y + r.height);
		float box[4] = { xmin, xmax, ymin, ymax };
		float x = (box[0] + box[1]) / 2.0 - 1;
		float y = (box[2] + box[3]) / 2.0 - 1;
		float w = box[1] - box[0];
		float h = box[3] - box[2];
		x = x * dw;
		w = w * dw;
		y = y * dh;
		h = h * dh;

		x = x < 0.0f ? 0.0f : x;
		y = y < 0.0f ? 0.0f : y;
		x = x > 1.0f ? 1.0f : x;
		y = y > 1.0f ? 1.0f : y;

		w = w < 0.0f ? 0.0f : w;
		h = h < 0.0f ? 0.0f : h;
		w = w > 1.0f ? 1.0f : w;
		h = h > 1.0f ? 1.0f : h;
		fprintf(in, "0 %f %f %f %f", x, y, w, h);
		fclose(in);
	}

	cv::rectangle(rimg, R.tl(), R.br(), cv::Scalar(0, 0, 255), 2);
	cv::imwrite(string_format("imgs\\%i.png", f_cnt), rimg);

	testRect(f_cnt, rimg);

}

byte gain = 0;
void cam_loop(){

	int w = cam.frame_w, h = cam.frame_h;
    cv::Mat frame = cv::Mat(h, w, CV_8U);

	if (use_experts_cam == false) {
		//cap.open("1000_2021_08_19_02_51_00_raw.avi", cv::CAP_FFMPEG);
		cap.open(1);
		if (!cap.isOpened()) {
			std::cerr << "ERROR: Can't initialize camera capture" << std::endl;
			std::cin.get();
			exit(1);
		}
	}

    cv::Mat frame_n;
    cv::Mat frame_r;
	int frame_cnt = 0;
	cv::namedWindow("frame", cv::WINDOW_FULLSCREEN);// cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
	std::string sbuf;
	int scale = 53;// 8 * 1;// 8;
	yo.init(false, true);//-------------------------------------
	int cfames = 0;
	int f_cnt = 1;
	
	cv::setMouseCallback("frame", mouse_callback);
	cam.setGain(128);
	for(;;){

        if(use_experts_cam) 
			cam.Grab((char*)frame.data);
		else {
			cap >> frame;
			if (frame.empty()) break;// end of video file
			//if (frame.channels() != 1) cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
		}

		if (frame.empty()) frame = cv::Mat(1458, 1088, CV_8U);

        cv::resize(frame, frame_n, cv::Size(1458,1088));
		//frame_n = frame.clone();
        ////contrastStrech(frame);//cv::Mat frame_n = frame.clone();
//        if(!thread_busy){
//            job = std::thread(templateMatch, frame);
//            job.detach();
//        }
		if (use_equalize) {
			if (frame_n.channels() != 1)
				frame_n = equalizeIntensity(frame_n);
			else
				cv::equalizeHist(frame_n, frame_n);
		}

		frame_r = frame_n.clone();
		if(frame_n.channels() == 1) cv::cvtColor(frame_n, frame_n, cv::COLOR_GRAY2BGR);

		putText(frame_n, string_format("%i %i", mx,my), cv::Point(mx, my), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 4);

		if (tcorr > 0.01f) {
			sbuf = string_format("(%.2f) [%i %i]", tcorr, tx, ty);
			putText(frame_n, sbuf, cv::Point(tx,ty), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0,0,255), 4);

			int i = 8 + scale;//8==1,
			float sx = barr.cols*i;
			float sy = barr.rows*i;
			sx *= 0.125;
			sy *= 0.125;
			cv::rectangle(frame_n, cv::Point(tx,ty), cv::Point(tx + sx, ty + sy), cv::Scalar(255, 0, 255), 1, 8);

			TD.print(string_format("%f %i %i", tcorr, tx, ty).c_str());
		}
		if(use_yolo) yo.detect(frame_n);

		if(use_detector){//gray
			cv::Mat t;
			int i = 8+scale;//8==1,
			float sx = barr.cols*i;
			float sy = barr.rows*i;
			sx*=0.125;
			sy*=0.125;
			//t = barr.clone();
			cv::resize(barr, t, cv::Size(int(sx), int(sy)));
			if (use_yolo) {
				int tx1=0, ty1=0, tx2=0, ty2=0, tx3=0, ty3=0;
				fixBox(yo.roi_1, frame_r);
				cv::Mat f1 = cv::Mat(frame_r, yo.roi_1);
				templateMatch(f1, t);
				cv::rectangle(frame_n, yo.roi_1, cv::Scalar(255, 255, 0), 1);
				if (tcorr > 0.01f) {
					tx1 = yo.roi_1.x + tx;
					ty1 = yo.roi_1.y + ty;
					sbuf = string_format("1)(%.2f) [%i %i]", tcorr, tx1, ty1);
					cv::rectangle(frame_n, cv::Rect(tx1, ty1, sx, sy), cv::Scalar(255, 255, 0), 1);
					putText(frame_n, sbuf, cv::Point(tx1, ty1), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 4);
				}

				fixBox(yo.roi_2, frame_r);
				f1 = cv::Mat(frame_r, yo.roi_2);
				templateMatch(f1, t);
				cv::rectangle(frame_n, yo.roi_2, cv::Scalar(255, 255, 0), 1);
				if (tcorr > 0.01f) {
					tx2 = yo.roi_2.x + tx;
					ty2 = yo.roi_2.y + ty;
					sbuf = string_format("2)(%.2f) [%i %i]", tcorr, tx2, ty2);
					cv::rectangle(frame_n, cv::Rect(tx2, ty2, sx, sy), cv::Scalar(255, 255, 0), 1);
					putText(frame_n, sbuf, cv::Point(tx2, ty2), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 4);
				}

				fixBox(yo.roi_3, frame_r);
				f1 = cv::Mat(frame_r, yo.roi_3);
				templateMatch(f1, t);
				cv::rectangle(frame_n, yo.roi_3, cv::Scalar(255, 255, 0), 1);
				if (tcorr > 0.01f) {
					tx3 = yo.roi_3.x + tx;
					ty3 = yo.roi_3.y + ty;
					sbuf = string_format("3)(%.2f) [%i %i]", tcorr, tx3, ty3);
					cv::rectangle(frame_n, cv::Rect(tx3, ty3, sx, sy), cv::Scalar(255, 255, 0), 1);
					putText(frame_n, sbuf, cv::Point(tx3, ty3), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 4);
				}
				TD.print(string_format("1) %i %i 2) %i %i 3) %i %i", tx1, ty1, tx2, ty2, tx3, ty3).c_str());
			} else {
				templateMatch(frame_r, t);
			}
			putText(frame_n, string_format("x%.2f (%i) [%i]frames", i*0.125, scale, cfames), cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(128), 1);
			cfames++;
			cv::rectangle(frame_n, cv::Rect(50,50, sx, sy), cv::Scalar(255, 255, 0), 1);
		}
        ////if(use_detector){
        ////    for(int i=1;i<50;i++){
        ////        cv::Mat f = frame_n.clone();
        ////        cv::Mat t;
        ////        float sx = barr.cols*i;
        ////        float sy = barr.rows*i;
        ////        sx*=0.125;
        ////        sy*=0.125;
        ////        if(int(sx)<1 || int(sy)<1)continue;
        ////        std::cout << " x=" << int(sx) << " y=" <<int(sy) <<std::endl;
        ////        cv::resize(barr, t, cv::Size(int(sx),int(sy)));
        ////        templateMatch(f, t);
        ////        putText(f, string_format("x%.2f",i*0.125), cv::Point(100,100), cv::FONT_HERSHEY_PLAIN, 6, cv::Scalar::all(128), 4);
        ////        cv::imwrite(string_format("temp_%i.png",i),f);
        ////    }
        ////    exit(1);
            //templateMatch(frame_n, barr2);
            //templateMatch(frame_n, barr3);
            //templateMatch(frame_n, barr4);
        ////}
        //cv::medianBlur(frame_n, frame, 3);
        //cv::GaussianBlur(frame_n, frame, cv::Size(3,3),3);
		if (recording) {
			if (record_raw) {
				cv::cvtColor(frame, frame_r, cv::COLOR_GRAY2BGR);
				writer->write(frame_r);
			} else {
				writer->write(frame_n);
			}
			frame_cnt++;
			std::string label = cv::format("recording frame %i", frame_cnt);
			putText(frame_n, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		}
        cv::imshow("frame", frame_n);

		int key = cv::waitKey(1);
		if (key == 27) { break; }
		if (key == 'd') use_detector = !use_detector;
		if (key == 'y') use_yolo = !use_yolo;
		if (key == 'e') use_equalize = !use_equalize;
		if (key == 'n') { scale++; }
		if (key == '1') { if (gain > 10)gain -= 10; else gain = 0; if (use_experts_cam) cam.setGain(gain); else cap.set(cv::CAP_PROP_BRIGHTNESS, gain); }
		if (key == '2') { if (gain < 245)gain += 10; else gain = 255; if (use_experts_cam) cam.setGain(gain); else cap.set(cv::CAP_PROP_BRIGHTNESS, gain); }
		if (key == 'm') { if(scale>1) scale--; }
		if (key == ' ') { 
			cv::Mat frame_lit;
			if(use_experts_cam) cam.setGain(230); else cap.set(cv::CAP_PROP_BRIGHTNESS, 230);
			if (use_experts_cam)
				for (int i = 0; i < 10; i++)cam.Grab((char*)frame.data);
			else
				cap.read(frame);
			cv::resize(frame, frame_lit, cv::Size(1458, 1088));
			if (use_experts_cam) cam.setGain(gain); else cap.set(cv::CAP_PROP_BRIGHTNESS, 128);
			//if (frame_lit.channels() == 1) cv::cvtColor(frame_lit, frame_lit, cv::COLOR_GRAY2BGR);
			if (frame_lit.channels() == 1) cv::cvtColor(frame_lit, frame_lit, cv::COLOR_GRAY2BGR);
			
			cv::Mat comp;
			cv::addWeighted(frame_n, 0.5f, frame_lit, 0.5f, 0.0, comp);

			cv::Rect2d r = cv::selectROI("frame", comp);
			cv::Rect2d R = r;
			cv::Mat RM;
			std::vector<cv::Point2f> pts(4), pts2(4);
			cv::Size sz1 = frame_n.size();
			cv::Size sz2 = cv::Size(sz1.height, sz1.width);
			
			WriteRect_YOLO(R, sz1, f_cnt, frame_n, ROT_CODE::R0);
			f_cnt++;
			
			// lit frame
			WriteRect_YOLO(R, sz1, f_cnt, frame_lit, ROT_CODE::R0);
			f_cnt++;

			////////cv::rotate(frame_n, frame_n, cv::ROTATE_90_CLOCKWISE);
			////////r = R;
			////////pts[0] = cv::Point(r.x, r.y);
			////////pts[1] = cv::Point(r.x + r.width, r.y);
			////////pts[2] = cv::Point(r.x, r.y + r.height);
			////////pts[3] = cv::Point(r.x + r.width, r.y + r.height);
			////////RM = cv::getRotationMatrix2D(cv::Point2i(sz1.width, sz1.height)/2, 90, 1);
			////////cv::transform(pts, pts2, RM);
			////////R = cv::boundingRect(pts2);
			WriteRect_YOLO(R, sz2, f_cnt, frame_n, ROT_CODE::R90);
			//testRect(f_cnt, frame_n);//cv::rectangle(frame_n, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
			//imwrite(string_format("%i.png", f_cnt), frame_n);//3
			f_cnt++;

			// lit
			//cv::rotate(frame_lit, frame_lit, cv::ROTATE_90_CLOCKWISE);
			WriteRect_YOLO(R, sz2, f_cnt, frame_lit, ROT_CODE::R90);
			//testRect(f_cnt, frame_lit);//cv::rectangle(frame_lit, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
			//imwrite(string_format("%i.png", f_cnt), frame_lit);//4
			f_cnt++;

			////cv::rotate(frame_n, frame_n, cv::ROTATE_90_CLOCKWISE);
			////r = R;
			////pts[0] = cv::Point(r.x, r.y);
			////pts[1] = cv::Point(r.x + r.width, r.y);
			////pts[2] = cv::Point(r.x, r.y + r.height);
			////pts[3] = cv::Point(r.x + r.width, r.y + r.height);
			////RM = cv::getRotationMatrix2D(cv::Point2i(sz2.width, sz2.height)/2, 90, 1);
			////cv::transform(pts, pts2, RM);
			////R = cv::boundingRect(pts2);
			WriteRect_YOLO(R, sz1, f_cnt, frame_n, ROT_CODE::R180);
			//testRect(f_cnt, frame_n);//cv::rectangle(frame_n, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
			//imwrite(string_format("%i.png", f_cnt), frame_n);//5
			f_cnt++;

			// lit
			//cv::rotate(frame_lit, frame_lit, cv::ROTATE_90_CLOCKWISE);
			WriteRect_YOLO(R, sz1, f_cnt, frame_lit, ROT_CODE::R180);
			//testRect(f_cnt, frame_lit);//cv::rectangle(frame_lit, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
			//imwrite(string_format("%i.png", f_cnt), frame_lit);//6
			f_cnt++;

			////cv::rotate(frame_n, frame_n, cv::ROTATE_90_CLOCKWISE);
			////r = R;
			////pts[0] = cv::Point(r.x, r.y);
			////pts[1] = cv::Point(r.x + r.width, r.y);
			////pts[2] = cv::Point(r.x, r.y + r.height);
			////pts[3] = cv::Point(r.x + r.width, r.y + r.height);
			////RM = cv::getRotationMatrix2D(cv::Point2i(sz1.width, sz1.height)/2, 90, 1);
			////cv::transform(pts, pts2, RM);
			////R = cv::boundingRect(pts2);
			WriteRect_YOLO(R, sz2, f_cnt, frame_n, ROT_CODE::R270);
			//testRect(f_cnt, frame_n);//cv::rectangle(frame_n, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
			//imwrite(string_format("%i.png", f_cnt), frame_n);//7
			f_cnt++;

			// lit
			//cv::rotate(frame_lit, frame_lit, cv::ROTATE_90_CLOCKWISE);
			WriteRect_YOLO(R, sz2, f_cnt, frame_lit, ROT_CODE::R270);
			//testRect(f_cnt, frame_n);//cv::rectangle(frame_lit, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
			//imwrite(string_format("%i.png", f_cnt), frame_lit);//8
			f_cnt++;
		}
		if (key == 'p') { 
			recording = !recording; 
			if (recording) {
				frame_cnt = 0;
				//writer.open("capture.mp4", cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), 30, cv::Size(frame_n.rows, frame_n.cols));
				if (record_raw) {
					writer = new cv::VideoWriter(string_format("%s_raw.avi", get_time_name()).c_str(), writer->fourcc('M', 'P', 'E', 'G'), 30, cv::Size(h, w));
				} else {
					writer = new cv::VideoWriter(string_format("%s_proc.avi", get_time_name()).c_str(), writer->fourcc('M', 'P', 'E', 'G'), 30, cv::Size(h, w));
				}
			}
		}
    }
}

int main(int argc, char *argv[])
{
//    DBOUT(" args:" << argc);
//    for(int i=0;i<argc;i++){
//        DBOUT(" argv:" << argv[i]);
//    }
	//for (int i = 3; i < 33; i++) {
	//	cv::Rect2f r = load(i);
	//	cv::Mat img = cv::imread(string_format("g:\\labels\\%i.png", i));
	//	int w = img.cols;
	//	int h = img.rows;
	//	cv::Rect R = cv::Rect(r.x*w, r.y*h, r.width*w, r.height*h);
	//	cv::rectangle(img, R.tl(), R.br(), cv::Scalar(255, 0, 0), 2);
	//	cv::imwrite(string_format("g:\\labels\\2\\%i.png", i), img);
	//	//WriteRect_YOLO(r, cv::Size(1, 1), i);
	//	break;
	//}
	//exit(1);

    src = cv::imread("ship11.png");
    cv::cvtColor(src, src, cv::COLOR_BGRA2GRAY);

    barr = cv::imread("barrel11.png");
    cv::cvtColor(barr, barr, cv::COLOR_BGR2GRAY);
    cv::resize(barr, barr, cv::Size(barr.cols*0.5,barr.rows));// width shrink

    float mul = 3;
    cv::resize(barr, barr2, cv::Size(barr.cols*mul,barr.rows*mul));

    mul = 8;
    cv::resize(barr, barr3, cv::Size(barr.cols*mul,barr.rows*mul));

    mul = 22;
    cv::resize(barr, barr4, cv::Size(barr.cols*mul,barr.rows*mul));

    cam_loop();

//	//src = Rotate(src, 10);
//	cv::GaussianBlur(src, src, cv::Size(ks(2), ks(2)), 0);
	
//	cv::Mat c = src.clone();
//	Detect(c);

////	for (int i = -20; i <= 20; i++) {
////		//c = src.clone();
////		c = Rotate(src, i/10);
////		DBOUT("angle "<<i/10);
////		templateMatch(c);
////	}

//    c = src.clone();
//    templateMatch(c);

//    //cv::namedWindow("ship", cv::WINDOW_AUTOSIZE);
//    //cv::createTrackbar("thresh:", "ship", &thresh, 255, get_markers);
//    //get_markers(0,0);

//	cv::waitKey();
	return 0;
}
