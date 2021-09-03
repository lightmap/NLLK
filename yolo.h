#pragma once
#include <opencv2/dnn.hpp>
using namespace cv::dnn;

#include <fstream>

struct YOLO {

	//https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/?nowprocket=1
	// Initialize the parameters
	float confThreshold = 0.3; // Confidence threshold
	float nmsThreshold = 0.4;  // Non-maximum suppression threshold
	int inpWidth = 416;        // Width of network's input image
	int inpHeight = 416;       // Height of network's input image
	std::vector<std::string> classes;
	cv::Mat blobFromImg;
	Net network;
	int target_class_id;
	cv::Rect roi_1, roi_2, roi_3;

	// Get the names of the output layers
	std::vector<std::string> getOutputsNames(const Net& net)
	{
		static std::vector<std::string> names;
		if (names.empty()) {
			//Get the indices of the output layers, i.e. the layers with unconnected outputs
			std::vector<int> outLayers = net.getUnconnectedOutLayers();
			//get the names of all the layers in the network
			std::vector<std::string> layersNames = net.getLayerNames();
			// Get the names of the output layers in names
			names.resize(outLayers.size());
			for (size_t i = 0; i < outLayers.size(); ++i) {
				names[i] = layersNames[outLayers[i] - 1];
				std::cerr << names[i] << std::endl;
			}
		}
		return names;
	}

	// Draw the predicted bounding box
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame, bool isTarget)
	{
		//Draw a rectangle displaying the bounding box
		if(isTarget) cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), isTarget ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 178, 50), 3);

		//Get the label for the class name and its confidence
		std::string label = cv::format("%.2f", conf);
		if (!classes.empty()) {
			CV_Assert(classId < (int)classes.size());
			label = classes[classId] + ":" + label;
		}
		int w = right - left;
		int h = bottom  - top;
		if (isTarget) {
			if (h > w) {
				label = label + " front ";// +cv::format("%i", w) + "/" + cv::format("%i", h);
			} else {
				label = label + " side ";// +cv::format("%i", w) + "/" + cv::format("%i", h);
			}
		}

		//Display the label at the top of the bounding box
		int baseLine;
		cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = cv::max(top, labelSize.height);
		if (isTarget) {
			cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
			int bws = w/3;
			cv::line(frame, cv::Point(left + bws, top), cv::Point(left + bws, bottom), cv::Scalar(0, 128, 255), 1);
			cv::line(frame, cv::Point(right - bws, top), cv::Point(right - bws, bottom), cv::Scalar(0, 128, 255), 1);
			roi_1 = cv::Rect(left, top, bws, h);
			roi_2 = cv::Rect(left+bws, top, bws, h);
			roi_3 = cv::Rect(left+bws*2, top, bws, h);
			cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
		}
	}

	// Remove the bounding boxes with low confidence using non-maxima suppression
	void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
	{
		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;

		for (size_t i = 0; i < outs.size(); ++i) {
			// Scan through all the bounding boxes output from the network and keep only the
			// ones with high confidence scores. Assign the box's class label as the class
			// with the highest score for the box.
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				// Get the value and location of the maximum score
				cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold) {
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}

		// Perform non maximum suppression to eliminate redundant overlapping boxes with
		// lower confidences
		std::vector<int> indices;
		NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
		for (size_t i = 0; i < indices.size(); ++i) {
			int idx = indices[i];
			cv::Rect box = boxes[idx];
			int o_class = classIds[idx];
			//if(o_class == target_class_id)
			drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, o_class == target_class_id);
		}
	}

	void init(bool tiny = true, bool cuda = false) {
		// Load names of classes
		//std::string classesFile = "yolo/coco.names";
		std::string classesFile = "yolo/tracker.names";
		std::ifstream ifs(classesFile.c_str());
		std::string line;
		while (std::getline(ifs, line)) classes.push_back(line);
		
		for (int i = 0; i < classes.size(); i++) {
			//if (classes[i] == "boat") { target_class_id = i; break; }
			if (classes[i] == "tracker") { target_class_id = i; break; }
		}

		//std::string model = "yolo/yolov3.weights";// 330ms CL
		//std::string config = "yolo/yolov3.cfg";

		std::string model = "yolo/yolov4.weights";
		std::string config = "yolo/yolov4.cfg";
		//std::string model = "yolo/yolov3_tracker.weights";
		//std::string config = "yolo/yolov3_tracker.cfg";

		if (tiny) {
			model = "yolo/yolov3-tiny.weights";// 30ms CL
			config = "yolo/yolov3-tiny.cfg";
		}
		network = readNet(model, config, "Darknet");

		if (cuda) {
			network.setPreferableBackend(DNN_BACKEND_CUDA);
			network.setPreferableTarget(DNN_TARGET_CUDA);// DNN_TARGET_CUDA_FP16);//
		} else {
			network.setPreferableBackend(DNN_BACKEND_DEFAULT);
			network.setPreferableTarget(DNN_TARGET_OPENCL);// cpu~1000ms, CL~330ms, CU~3ms, CU16=lag
		}
	}

	void detect(cv::Mat &frame) {
		
		bool swapRB = true;
		blobFromImage(frame, blobFromImg, 1, cv::Size(inpWidth, inpHeight), cv::Scalar(), swapRB, false, CV_8U);// CV_32F);

		float scale = 1.0 / 255.0;
		cv::Scalar mean = 0;
		network.setInput(blobFromImg, "", scale, mean);

		std::vector<cv::Mat> outMat;
		network.forward(outMat, getOutputsNames(network));

		postprocess(frame, outMat);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		std::vector<double> layersTimes;
		double freq = cv::getTickFrequency() / 1000;
		double t = network.getPerfProfile(layersTimes) / freq;
		std::string label = cv::format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

		//imshow("ship_det", frame);
	}

};