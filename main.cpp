#include <mtcnnfacedetection.h>

#define PNETPROTOTXT "Model/det1.prototxt"
#define PNETWEIGHT "Model/det1.caffemodel"

#define RNETPROTOTXT "Model/det2.prototxt"
#define RNETWEIGHT "Model/det2.caffemodel"

#define ONETPROTOTXT "Model/det3.prototxt"
#define ONETWEIGHT "Model/det3.caffemodel"


int main(int argc, char *argv[])
{
    cv::VideoCapture cap ;
    if (!cap.open(0)){
        return 0 ;
    }


    MTCNNFaceDetection mtcnn(PNETPROTOTXT,PNETWEIGHT,RNETPROTOTXT,RNETWEIGHT,ONETPROTOTXT,ONETWEIGHT) ;

    for(;;)
    {
          cv::Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          cv::vector<cv::Mat> res =mtcnn.detecte_faces(frame);
          cv::vector<cv::Rect> rects = mtcnn.getFaceBoxRect(res[0]) ;
          cv::vector<cv::vector<cv::Point2i>>  points = mtcnn.getFacePoints(res[1]) ;

          for (cv::Rect rect : rects) {
              cv::rectangle(frame,rect,cv::Scalar(0,255,0));
          }

          for (cv::vector<cv::Point2i> tmp : points) {
              for (cv::Point2i point : tmp) {
                  cv::circle(frame,point,1,cv::Scalar(0,0,255),-1);
              }
          }
          cv::imshow("this is you, smile! :)", frame);
          if( cv::waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }


    return 0;
}

