#include <fstream>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include "task_thread_pool.hpp"


using namespace std;
using namespace cv;

std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs("../cocos.names");
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

void load_net(cv::dnn::Net &net, bool is_cuda) {

        auto result = cv::dnn::readNet("../best.onnx");
    if (is_cuda) {
        std::cout << "Attempt to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
                                        cv::Scalar(255, 0, 0)};

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void
detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {
    cv::Mat blob;

    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true,
                           false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());


    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
        yolov8 = true;
        rows = outputs[0].size[2];
        dimensions = outputs[0].size[1];

        outputs[0] = outputs[0].reshape(1, dimensions);
        cv::transpose(outputs[0], outputs[0]);
    }
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *) outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        if (yolov8)
        {
            float *classes_scores = data+4;

            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > SCORE_THRESHOLD)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);

                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // yolov5
        {
            float confidence = data[4];

            if (confidence >= CONFIDENCE_THRESHOLD)
            {
                float *classes_scores = data+5;

                cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > SCORE_THRESHOLD)
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

vector<Ptr<Tracker>> trackers;
std::vector<std::string> class_list = load_class_list();
std::vector<Detection> detectedObjects;
bool hasNewDetection=false;
std::vector<Detection> newDetections;
task_thread_pool::task_thread_pool pool;

int frameId=-1;
Mat frame;

void threadNewDetection(){
    while(true){
        detectedObjects.clear();
//        detect(frame, net, detectedObjects, class_list);
    }
}

Rect updateTrackerAndReturnRect(const Ptr<Tracker>& tracker){
    Rect box;
    tracker->update(frame,box);
    return box;
}

int main(int argc, char **argv) {

    VideoWriter videoWriter;


    cv::VideoCapture cap("../../assets/8.MP4");
    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    Mat originalFrame;

    cap >> originalFrame;
    if(originalFrame.empty()) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    resize(originalFrame,frame,Size(1920,1080));


//    videoWriter.open("appsrc ! x264x tune=zerolatency ! rtph264pay aggregate-mode=zero-latency ! udpsink host=127.0.0.1 port 8123",CAP_GSTREAMER,0,(double)30,Size(frame.cols,frame.rows),true);

//    videoWriter.open("appsrc ! autovideoconvert ! video/x-raw,format=I420 ! x264x tune=zerolatency ! rtph264pay aggregate-mode=zero-latency ! udpsink host=127.0.0.1 port 8123",CAP_GSTREAMER,0,(double)30,Size(frame.cols,frame.rows),true);
   // videoWriter.open("appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=12345",CAP_GSTREAMER,0,(double)30,Size(1920,1080),true);
   cout<<frame.cols << frame.rows;
    videoWriter.open("appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw,format=RGBA ! autovideoconvert ! omxh264enc ! matroskamux ! filesink location=video.mkv sync=false",CAP_GSTREAMER,0,(double)30,Size(frame.cols,frame.rows),true);

//videoWriter.open("video.avi",0,29,Size(frame.cols,frame.rows),true);

    videoWriter.open("appsrc ! queue ! videoconvert ! video/x-raw ! omxh264enc ! video/x-h264 ! h264parse ! rtph264pay ! udpsink host=192.168.0.2 port=5000 sync=false",0,25.0,Size(1920,1080));

//    appsrc ! autovideoconvert ! videoscale ! video/x-raw,format=I420,width=1280,height=720,framerate=30/1 ! jpegenc ! rtpjpegpay ! udpsink host=127.0.0.1 port=5001
    if(!videoWriter.isOpened()) {
        cout << "Error opening output video stream" << endl;
        return -1;
    }

    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    cv::dnn::Net net;
    load_net(net, is_cuda);

    auto start = std::chrono::high_resolution_clock::now();


    Rect detectionBox(Point(0,0),Point(frame.cols,frame.rows));
    while(cap.isOpened()){
        double timer = (double)getTickCount();
        // Capture frame-by-frame
        cap >> originalFrame;
        resize(originalFrame,frame,Size(1920,1080));
        frameId++;

        // If the frame is empty, break immediately
        if (frame.empty()) {
            break;
        }

//kond
      //  if(frameId % 100 == 0) {
//Tond
        if(frameId % 100 == 0 || trackers.size()==0) {

            detectedObjects.clear();
            Mat croppedFrame=frame(detectionBox);
            detect(croppedFrame, net, detectedObjects, class_list);

            trackers.clear();
            if(!detectedObjects.empty()) {

                Detection largestDetection = detectedObjects[0];
                for (int i = 1; i < detectedObjects.size(); i++) {
                    if(largestDetection.box.width < detectedObjects[i].box.width)
                        largestDetection=detectedObjects[i];
                }

                largestDetection.box.x+=detectionBox.x;
                largestDetection.box.y+=detectionBox.y;

                Ptr<Tracker> tracker=TrackerMIL::create();

                Point center(largestDetection.box.x+largestDetection.box.width/2,largestDetection.box.y+largestDetection.box.height/2);
                detectionBox=Rect(Point(min(frame.cols,max(0,center.x-largestDetection.box.width)),min(frame.rows,max(0,center.y-largestDetection.box.height))),Point(min(frame.cols,max(0,center.x+largestDetection.box.width)),min(frame.rows,max(0,center.y+largestDetection.box.height))));

                tracker->init(frame,largestDetection.box);
                trackers.push_back(tracker);
            }
            else{
                Point center(detectionBox.x+detectionBox.width/2,detectionBox.y+detectionBox.height/2);
                detectionBox=Rect(Point(min(frame.cols,max(0,center.x-detectionBox.width)),min(frame.rows,max(0,center.y-detectionBox.height))),Point(min(frame.cols,max(0,center.x+detectionBox.width)),min(frame.rows,max(0,center.y+detectionBox.height))));

            }
        }



        vector<future<Rect>> futures;
        for(int i=0;i<trackers.size();i++) {
            futures.push_back(pool.submit([i] { return updateTrackerAndReturnRect(trackers[i]); }));

        }
        for(int i=0;i<trackers.size();i++){
            Rect box=futures[i].get();

            auto detection = detectedObjects[i];
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 3);

            Point center(box.x+box.width/2,box.y+box.height/2);
            detectionBox=Rect(Point(min(frame.cols,max(0,center.x-box.width)),min(frame.rows,max(0,center.y-box.height))),Point(min(frame.cols,max(0,center.x+box.width)),min(frame.rows,max(0,center.y+box.height))));


            cv::rectangle(frame, cv::Point(box.x, box.y - 40), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(frame, to_string(i)+" : "+class_list[classId] + " , "+to_string(detection.confidence), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0));
            cv::putText(frame, "X: "+to_string(center.x)+" Y: "+to_string(center.y), cv::Point(box.x, box.y - 25), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 0));
        }

        cv::rectangle(frame, detectionBox, colors[0], 1);


        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        putText(frame, "FPS : " + to_string(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255), 2);


        cv::imshow("output", frame);
        videoWriter.write(frame);




        // Press  ESC on keyboard to exit
        char c=(char) cv::waitKey(25);
        if(c==27)
            break;

    }


    // When everything done, release the video capture object
    cap.release();
    videoWriter.release();
    // Closes all the frames
    cv::destroyAllWindows();

    return 0;
}
