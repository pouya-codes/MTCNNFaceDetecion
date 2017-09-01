#ifndef MTCNNFACEDETECTION_H
#define MTCNNFACEDETECTION_H


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
using namespace caffe;
using std::string;
enum {
    NMS_UNION = 1 ,
    NMS_MIN = 2
};
class MTCNNFaceDetection {

public:
    MTCNNFaceDetection(const string& PNet_prototxt,const string& PNet_weight, const string& RNet_prototxt,const string& RNet_weight,const string& ONet_prototxt,const string& ONet_weight);
    cv::vector<cv::Mat> detecte_faces(cv::Mat img_org);
    cv::vector<cv::Rect> getFaceBoxRect(cv::Mat boxes) ;
    cv::vector<cv::vector<cv::Point2i>> getFacePoints(cv::Mat points) ;

    ~MTCNNFaceDetection();
private :
    double threshold [3] = {0.6, 0.7, 0.7};
    double factor = 0.709 ;
    int minsize = 20 ;
    shared_ptr<Net<float> > PNet,RNet,ONet;
    void Preprocess(shared_ptr<Net<float>>& net_,const cv::Mat& img);
    void Preprocess(shared_ptr<Net<float>>& net_,const cv::vector<cv::Mat>& imgs);
    cv::Mat generateBoundingBox(cv::Mat prob,cv::vector<cv::Mat> conv4_2,double scale,double threshold);
    cv::Mat nms(cv::Mat boxes,double threshold, int nms_type);
    void rerec (cv::Mat& bboxA);
    void bbreg(cv::Mat& boundingbox,cv::Mat reg);
    cv::vector<cv::Mat> pad(cv::Mat boxesA,int w,int h);
    int init_nets(const string& PNet_prototxt,const string& PNet_weight, const string& RNet_prototxt,const string& RNet_weight,const string& ONet_prototxt,const string& ONet_weight);
    void drawBoxes(cv::Mat &img,cv::Mat boxes);

};



#endif // MTCNNFACEDETECTION_H
