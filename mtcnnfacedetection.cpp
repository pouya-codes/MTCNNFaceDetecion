#include "mtcnnfacedetection.h"


MTCNNFaceDetection::MTCNNFaceDetection(const string& PNet_prototxt,const string& PNet_weight, const string& RNet_prototxt,const string& RNet_weight,const string& ONet_prototxt,const string& ONet_weight) {
    init_nets(PNet_prototxt,PNet_weight,RNet_prototxt,RNet_weight,ONet_prototxt,ONet_weight) ;
}

MTCNNFaceDetection::~MTCNNFaceDetection(){

}

void MTCNNFaceDetection::Preprocess(shared_ptr<Net<float>>& net_,const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    cv::vector<cv::Mat> input_channels ;


    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    cv::split(img, input_channels);
}

void MTCNNFaceDetection::Preprocess(shared_ptr<Net<float>>& net_,const cv::vector<cv::Mat>& imgs) {


    cv::vector<cv::Mat> input_channels ;

    Blob<float>* input_layer = net_->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels()* input_layer->num(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }
    for (int i = 0; i < input_layer->num(); i++) {
        cv::vector<cv::Mat> channels;
        cv::split(imgs[i], channels);
        for (int j = 0; j < channels.size(); j++){
            channels[j].copyTo((input_channels)[i*input_layer->channels()+j]);
        }
    }


}


cv::Mat MTCNNFaceDetection::generateBoundingBox(cv::Mat prob,cv::vector<cv::Mat> conv4_2,double scale,double threshold){
    int stride = 2 ;
    int cellsize = 12 ;
    cv::Mat map ,dx1,dx2,dy1,dy2,mask;
    cv::transpose(prob,map);

    cv::transpose(conv4_2[0],dx1);
    cv::transpose(conv4_2[1],dy1);
    cv::transpose(conv4_2[2],dx2);
    cv::transpose(conv4_2[3],dy2);


    mask = map>= threshold ;
    cv::vector<cv::Point> idx ;
    cv::Mat boundingbox ;
    //    std::cout << mask << std::endl ;

    if(cv::countNonZero(mask)>0)
        cv::findNonZero(mask,idx);



    cv::Mat score,reg;
    for (cv::Point p : idx) {
        score.push_back(map.at<float>(p));
        boundingbox.push_back(p.x);
        boundingbox.push_back(p.y);
        reg.push_back(dx1.at<float>(p));
        reg.push_back(dy1.at<float>(p));
        reg.push_back(dx2.at<float>(p));
        reg.push_back(dy2.at<float>(p));
    }
    reg = reg.reshape(1,idx.size());
    if (reg.rows == 0)
        return cv::Mat();


    boundingbox.convertTo(boundingbox,CV_32F);

    cv::MatIterator_<float> it, end;
    cv::Mat bb1 = (stride * (boundingbox)+ 1) / scale ;
    for( it = bb1.begin<float>(), end = bb1.end<float>(); it != end; ++it) *it = std::floor(*it) ;

    cv::Mat bb2 = (stride * (boundingbox) + cellsize ) / scale ;
    for( it = bb2.begin<float>(), end = bb2.end<float>(); it != end; ++it) *it = std::floor(*it);

    bb1 = bb1.reshape(1,idx.size());
    bb2 = bb2.reshape(1,idx.size());

    cv::Mat boundingbox_out ;
    cv::hconcat(bb1,bb2,boundingbox_out);
    boundingbox_out.convertTo(boundingbox_out,CV_32FC1);
    cv::hconcat(boundingbox_out,score,boundingbox_out);
    cv::hconcat(boundingbox_out,reg,boundingbox_out);


    return boundingbox_out ;

}

cv::Mat MTCNNFaceDetection::nms(cv::Mat boxes,double threshold, int nms_type) {
    if (boxes.rows == 0)
        return cv::Mat();

    cv::Mat x1 = boxes.colRange(0,1);
    cv::Mat y1 = boxes.colRange(1,2);
    cv::Mat x2 = boxes.colRange(2,3);
    cv::Mat y2 = boxes.colRange(3,4);
    cv::Mat s = boxes.colRange(4,5);
    cv::Mat area =(x2-x1 +1 ).mul((y2 - y1 + 1));
    cv::Mat idx ;
    cv::sortIdx(s,idx,CV_SORT_EVERY_COLUMN ) ;

    cv::Mat pick,xx1,yy1,xx2,yy2,w,h,inter;
    while (idx.rows>0) {

        cv::Mat temp; for (int var = 0; var < idx.rows -1 ; ++var) temp.push_back(x1.at<float>(idx.at<int>(var)));
        xx1 = idx.rows<=1 ? x1.row(idx.at<int>(0)) : cv::max( temp,x1.at<float>(idx.at<int>(idx.rows-1))) ;

        temp.release(); for (int var = 0; var < idx.rows -1; ++var) temp.push_back(y1.at<float>(idx.at<int>(var)));
        yy1 = idx.rows<=1 ? y1.row(idx.at<int>(0)) : cv::max( temp,y1.at<float>(idx.at<int>(idx.rows-1)))  ;

        temp.release(); for (int var = 0; var < idx.rows -1; ++var) temp.push_back(x2.at<float>(idx.at<int>(var)));
        xx2 = idx.rows<=1 ? x2.row(idx.at<int>(0)) : cv::min( temp,x2.at<float>(idx.at<int>(idx.rows-1)));

        temp.release(); for (int var = 0; var < idx.rows -1; ++var) temp.push_back(y2.at<float>(idx.at<int>(var)));
        yy2 = idx.rows<=1 ? y2.row(idx.at<int>(0)) : cv::min( temp,y2.at<float>(idx.at<int>(idx.rows-1)))  ;

        w = cv::max(0.0,xx2-xx1+1);
        h = cv::max(0.0,yy2-yy1+1);
        inter = w.mul(h) ;

        if (idx.rows<=1) temp = area.row(idx.at<int>(0)) ;
        else {
            temp.release();
            for (int var = 0; var < idx.rows -1 ; ++var) temp.push_back(area.at<float>(idx.at<int>(var)));
        }


        cv::Mat o ;
        if (nms_type==NMS_MIN){
            o =  inter /  cv::min( temp,area.at<float>(idx.at<int>(idx.rows-1)))  ;

        }
        else {
            o = inter /  (temp+area.at<float>(idx.at<int>(idx.rows-1))-inter) ;
        }
        pick.push_back(idx.at<int>(idx.rows-1)) ;

        cv::Mat mask = o<=threshold ;
        cv::vector<cv::Point> thridx ;
        if(cv::countNonZero(mask)>0)
            cv::findNonZero(mask,thridx);



        idx.copyTo(temp);
        idx.release();
        for (cv::Point p : thridx) idx.push_back(temp.at<int>(p.y));
    }

    return pick;

}

void MTCNNFaceDetection::rerec (cv::Mat& bboxA){

    cv::Mat w = bboxA.col(2)-bboxA.col(0);
    cv::Mat h = bboxA.col(3)-bboxA.col(1);
    cv::Mat l = cv::max(w,h);
    bboxA.col(0) =bboxA.col(0)+ (w * 0.5) - (l * 0.5) ;
    bboxA.col(1) =bboxA.col(1)+ (h * 0.5) - (l * 0.5) ;
    bboxA.col(2) =bboxA.col(0)+ l ;
    bboxA.col(3) =bboxA.col(1)+ l ;
    cv::Mat temp = bboxA.colRange(0,4);
    for(cv::MatIterator_<float> it = temp.begin<float>(), end = temp.end<float>(); it != end; ++it) *it = std::floor(*it);
    bboxA.colRange(0,4) = temp;

}


void MTCNNFaceDetection::bbreg(cv::Mat& boundingbox,cv::Mat reg){

    cv::Mat w = boundingbox.col(2)-boundingbox.col(0) + 1 ;
    cv::Mat h = boundingbox.col(3)-boundingbox.col(1) + 1 ;


    boundingbox.col(0) = boundingbox.col(0) + reg.col(0).mul(w) ;
    boundingbox.col(1) = boundingbox.col(1) + reg.col(1).mul(h) ;
    boundingbox.col(2) = boundingbox.col(2) + reg.col(2).mul(w) ;
    boundingbox.col(3) = boundingbox.col(3) + reg.col(3).mul(h) ;


}

cv::vector<cv::Mat> MTCNNFaceDetection::pad(cv::Mat boxesA,int w,int h){



    cv::Mat tmph = boxesA.col(3) - boxesA.col(1) + 1 ;
    cv::Mat tmpw = boxesA.col(2)- boxesA.col(0) + 1 ;
    int numbox = boxesA.rows ;


    cv::Mat dx = cv::Mat::ones(numbox,1,CV_8U);
    cv::Mat dy = cv::Mat::ones(numbox,1,CV_8U);

    cv::Mat edx , edy;
    tmpw.copyTo(edx);
    tmph.copyTo(edy);

    cv::Mat x = boxesA.col(0) ;
    cv::Mat y = boxesA.col(1) ;
    cv::Mat ex = boxesA.col(2) ;
    cv::Mat ey = boxesA.col(3) ;

    cv::Mat temp = ex> w ;
    cv::vector<cv::Point> idx ;
    if(cv::countNonZero(temp)>0)
        cv::findNonZero(temp,idx);


    if(idx.size() != 0) {
        for (cv::Point p : idx) {
            edx.at<float>(p)= -ex.at<float>(p) + w - 1 + tmpw.at<float>(p) ;
            ex.at<float>(p) = w - 1 ;
        }
    }

    temp = ey > h ;

    if(cv::countNonZero(temp)>0)
        cv::findNonZero(temp,idx);

    if(idx.size() != 0) {
        for (cv::Point p : idx) {
            edy.at<float>(p)= -ey.at<float>(p) + h - 1 + tmph.at<float>(p) ;
            ey.at<float>(p) = h - 1 ;
        }
    }


    temp = x < 1 ;
    if(cv::countNonZero(temp)>0)
        cv::findNonZero(temp,idx);

    if(idx.size() != 0) {
        for (cv::Point p : idx) {
            dx.at<float>(p)=  2 - x.at<float>(p) ;
            x.at<float>(p) = 1;
        }
    }

    temp = y < 1 ;



    if(cv::countNonZero(temp)>0)
        cv::findNonZero(temp,idx);


    if(idx.size() != 0) {
        for (cv::Point p : idx) {
            dy.at<float>(p)=  2 - y.at<float>(p) ;
            y.at<float>(p) = 1;
        }
    }

    dy = cv::max(0, dy - 1) ;
    dx = cv::max(0, dx - 1) ;
    y = cv::max(0, y - 1) ;
    x = cv::max(0, x - 1) ;
    edy = cv::max(0, edy - 1) ;
    edx = cv::max(0, edx - 1) ;
    ey = cv::max(0, ey - 1) ;
    ex = cv::max(0, ex - 1) ;
    cv::vector<cv::Mat> out ;
    out.push_back(y);
    out.push_back(ey);
    out.push_back(x);
    out.push_back(ex);
    return out ;

}



int MTCNNFaceDetection::init_nets(const string& PNet_prototxt,const string& PNet_weight, const string& RNet_prototxt,const string& RNet_weight,const string& ONet_prototxt,const string& ONet_weight) {


    Caffe::set_mode(Caffe::GPU);

    PNet.reset(new Net<float>(PNet_prototxt, TEST));
    PNet->CopyTrainedLayersFrom(PNet_weight);

    RNet.reset(new Net<float>(RNet_prototxt, TEST));
    RNet->CopyTrainedLayersFrom(RNet_weight);

    ONet.reset(new Net<float>(ONet_prototxt, TEST));
    ONet->CopyTrainedLayersFrom(ONet_weight);

    return 0 ;

}

cv::vector<cv::Mat> MTCNNFaceDetection::detecte_faces(cv::Mat img_org){
    cv::Mat img ;
    cv::cvtColor(img_org,img,cv::COLOR_RGB2BGR) ;
    int factor_count = 0 ;
    cv::Mat total_boxes,points;

    int width = img.cols ;
    int height = img.rows;
    double minl = std::min(width,height);
    double m = 12.0 / minsize ;
    minl = minl * m ;

    cv::vector<cv::Mat> pad_out;

    //    create scale pyramid

    std::vector<float> scales ;
    while (minl >= 12){
        scales.push_back(m * pow(factor, factor_count));
        minl *= factor ;
        factor_count += 1 ;
    }
    for ( auto & scale : scales ) {


        int hs = (int) ceil(height * scale );
        int ws = (int) ceil(width * scale );

        cv::Mat im_data ;
        img.convertTo(im_data, CV_32FC3);
        cv::resize(im_data,im_data,cv::Size(ws, hs)) ;

        im_data = (im_data-127.5)* 0.0078125  ;
        cv::transpose(im_data, im_data);


        PNet->input_blobs()[0]->Reshape(1, 3, ws,hs);
        PNet->Reshape();
        Preprocess(PNet,im_data);
        PNet->Forward();
        const boost::shared_ptr<caffe::Blob<float> > blob_conv4_2 = PNet->blob_by_name("conv4-2");
        const boost::shared_ptr<caffe::Blob<float> > blob_prob1 = PNet->blob_by_name("prob1");

        const float* blob_conv4_2_data = blob_conv4_2->cpu_data()  + blob_conv4_2->offset(0);
        cv::Mat conv4_2_mat (1, blob_conv4_2->count(), CV_32FC1,(float*) blob_conv4_2_data);

        std::vector<cv::Mat> conv4_2;
        for (int i = 0; i < blob_conv4_2->channels(); ++i) {
            cv::Mat channel = conv4_2_mat.colRange(i * blob_conv4_2->shape()[2]*blob_conv4_2->shape()[3],(i+1) * blob_conv4_2->shape()[2]*blob_conv4_2->shape()[3]) ;
            conv4_2.push_back(channel.reshape(1,blob_conv4_2->shape()[2]));
        }


        const float* blob_prob1_data = blob_prob1->cpu_data()  + blob_prob1->offset(0);
        cv::Mat prob1_mat (1, blob_prob1->count(), CV_32FC1,(float*) blob_prob1_data);
        prob1_mat = prob1_mat.colRange(blob_prob1->shape()[2]*blob_prob1->shape()[3],blob_prob1->count());
        prob1_mat=prob1_mat.reshape(1,blob_prob1->shape()[2]);

        cv::Mat boxes = generateBoundingBox( prob1_mat, conv4_2, scale, threshold[0]) ;


        if(boxes.rows!=0){
            cv::Mat pick = nms(boxes,0.5,NMS_UNION) ;

            if (pick.rows> 0) {
                cv::Mat temp ;
                for (int var = 0; var <= pick.rows-1; ++var) temp.push_back(boxes.row(pick.at<int>(var)));
                temp.copyTo(boxes);

            }
        }
        if(boxes.rows!=0){
            total_boxes.rows == 0 ? boxes.copyTo(total_boxes) : cv::vconcat(total_boxes,boxes,total_boxes) ;
        }

    }
    int numbox = total_boxes.rows;
    if(numbox>0) {
        cv::Mat pick = nms(total_boxes,0.7,NMS_UNION);
        cv::Mat temp ;
        for (int var = 0; var <= pick.rows-1; ++var) temp.push_back(total_boxes.row(pick.at<int>(var)));
        temp.copyTo(total_boxes);

        cv::Mat regh = total_boxes.col(3) - total_boxes.col(1) ;
        cv::Mat regw = total_boxes.col(2) - total_boxes.col(0) ;
        cv::Mat t1 = total_boxes.col(0) + total_boxes.col(5).mul(regw) ;
        cv::Mat t2 = total_boxes.col(1) + total_boxes.col(6).mul(regh) ;
        cv::Mat t3 = total_boxes.col(2) + total_boxes.col(7).mul(regw) ;
        cv::Mat t4 = total_boxes.col(3) + total_boxes.col(8).mul(regh) ;
        cv::Mat t5 = total_boxes.col(4) ;
        cv::hconcat(t1,t2,total_boxes);
        cv::hconcat(total_boxes,t3,total_boxes);
        cv::hconcat(total_boxes,t4,total_boxes);
        cv::hconcat(total_boxes,t5,total_boxes);

        rerec(total_boxes);
        cv::Mat y ;

        pad_out = pad(total_boxes,width,height);
    }
    numbox = total_boxes.rows;

    if(numbox>0) {

        cv::Mat y =  pad_out.at(0) ;
        cv::Mat ey =  pad_out.at(1) ;
        cv::Mat x =  pad_out.at(2) ;
        cv::Mat ex =  pad_out.at(3) ;

        cv::vector<cv::Mat> tempimg ;

        for (int k = 0; k < numbox; ++k) {
            cv::Mat roi ;
            roi = img(cv::Rect(int(x.at<float>(k)),int(y.at<float>(k)) ,int(ex.at<float>(k)) + 1-int(x.at<float>(k)),int(ey.at<float>(k)) + 1 - int(y.at<float>(k))));
            roi.convertTo(roi,CV_32F);
            cv::resize(roi,roi,cv::Size(24,24));
            cv::transpose(roi,roi);
            tempimg.push_back((roi-127.5) * 0.0078125 );

        }

        RNet->input_blobs()[0]->Reshape(numbox, 3, 24,24);
        RNet->Reshape();
        Preprocess(RNet,tempimg);
        RNet->Forward();

        const boost::shared_ptr<caffe::Blob<float> > blob_conv5_2 = RNet->blob_by_name("conv5-2");
        const float* blob_conv5_2_data = blob_conv5_2->cpu_data()  + blob_conv5_2->offset(0);
        cv::Mat conv_5_2_mat (1, blob_conv5_2->count(), CV_32FC1,(float*) blob_conv5_2_data);
        conv_5_2_mat = conv_5_2_mat.reshape(0,blob_conv5_2->shape()[0]);

        const boost::shared_ptr<caffe::Blob<float> > blob_prob = RNet->blob_by_name("prob1");
        const float* blob_prob_data = blob_prob->cpu_data()  + blob_prob->offset(0);
        cv::Mat prob_mat (1, blob_prob->count(), CV_32FC1,(float*) blob_prob_data);
        prob_mat = prob_mat.reshape(0,blob_prob->shape()[0]);
        cv::Mat score = prob_mat.col(1) ;
        cv::Mat pass_t = score > threshold[1] ;
        cv::vector<cv::Point> passidx ;
        if(cv::countNonZero(pass_t)>0)
            cv::findNonZero(pass_t,passidx);


        cv::Mat temp,mv ;
        for (cv::Point p : passidx) {
            temp.push_back(total_boxes.col(0).at<float>(p));
            temp.push_back(total_boxes.col(1).at<float>(p));
            temp.push_back(total_boxes.col(2).at<float>(p));
            temp.push_back(total_boxes.col(3).at<float>(p));
            temp.push_back(score.at<float>(p));
            mv.push_back(conv_5_2_mat.row(p.y));
        }
        if (mv.rows > 0) {
            mv = mv.reshape(0,passidx.size());
            cv::transpose(mv,mv);
            total_boxes=temp.reshape(0,passidx.size());
            if (total_boxes.rows>0) {
                cv::Mat pick = nms(total_boxes, 0.7, NMS_UNION) ;


                if (pick.rows>0){
                    temp.release();
                    cv::Mat tempmv ;
                    for( cv::MatIterator_<int> it = pick.begin<int>(), end = pick.end<int>(); it != end; ++it)
                    {
                        temp.push_back(total_boxes.row(*it));
                        tempmv.push_back(mv.col(*it));
                    }
                    mv= tempmv.reshape(0,pick.rows) ;
                    temp.copyTo(total_boxes);
                    bbreg(total_boxes,mv);
                    rerec(total_boxes);
                }


            }


            numbox = total_boxes.rows;

            if(numbox>0) {
                pad_out = pad(total_boxes,width,height);
                cv::Mat y =  pad_out.at(0) ;
                cv::Mat ey =  pad_out.at(1) ;
                cv::Mat x =  pad_out.at(2) ;
                cv::Mat ex =  pad_out.at(3) ;

                cv::vector<cv::Mat> tempimg ;

                for (int k = 0; k < numbox; ++k) {
                    cv::Mat roi ;
                    roi = img(cv::Rect(int(x.at<float>(k)),int(y.at<float>(k)) ,int(ex.at<float>(k)) + 1-int(x.at<float>(k)),int(ey.at<float>(k)) + 1 - int(y.at<float>(k))));
                    roi.convertTo(roi,CV_32F);
                    cv::resize(roi,roi,cv::Size(48,48));
                    cv::transpose(roi,roi);
                    tempimg.push_back((roi-127.5) * 0.0078125 );

                }

                ONet->input_blobs()[0]->Reshape(numbox, 3, 48,48);
                ONet->Reshape();
                Preprocess(ONet,tempimg);
                ONet->Forward();


                const boost::shared_ptr<caffe::Blob<float> > blob_conv6_3 = ONet->blob_by_name("conv6-3");
                const float* blob_conv6_3_data = blob_conv6_3->cpu_data()  + blob_conv6_3->offset(0);
                cv::Mat conv_6_3_mat (1, blob_conv6_3->count(), CV_32FC1,(float*) blob_conv6_3_data);
                conv_6_3_mat = conv_6_3_mat.reshape(0,blob_conv6_3->shape()[0]);

                const boost::shared_ptr<caffe::Blob<float> > blob_prob = ONet->blob_by_name("prob1");
                const float* blob_prob_data = blob_prob->cpu_data()  + blob_prob->offset(0);
                cv::Mat prob_mat (1, blob_prob->count(), CV_32FC1,(float*) blob_prob_data);
                prob_mat = prob_mat.reshape(0,blob_prob->shape()[0]);
                cv::Mat score = prob_mat.col(1) ;

                const boost::shared_ptr<caffe::Blob<float> > blob_conv6_2 = ONet->blob_by_name("conv6-2");
                const float* blob_conv6_2_data = blob_conv6_2->cpu_data()  + blob_conv6_2->offset(0);
                cv::Mat conv6_2_mat (1, blob_conv6_2->count(), CV_32FC1,(float*) blob_conv6_2_data);
                conv6_2_mat = conv6_2_mat.reshape(0,blob_conv6_2->shape()[0]);



                cv::Mat pass_t = score > threshold[2] ;
                cv::vector<cv::Point> passidx ;
                if(cv::countNonZero(pass_t)>0)
                    cv::findNonZero(pass_t,passidx);



                cv::Mat temp,mv ;
                for (cv::Point p : passidx) {
                    points.push_back(conv_6_3_mat.row(p.y));
                    temp.push_back(total_boxes.col(0).at<float>(p));
                    temp.push_back(total_boxes.col(1).at<float>(p));
                    temp.push_back(total_boxes.col(2).at<float>(p));
                    temp.push_back(total_boxes.col(3).at<float>(p));
                    temp.push_back(score.at<float>(p));
                    mv.push_back(conv6_2_mat.row(p.y));
                }
                total_boxes=temp.reshape(0,passidx.size());
                if (total_boxes.rows>0) {

                    mv = mv.reshape(0,passidx.size());
                    cv::transpose(mv,mv);
                    cv::Mat w = total_boxes.col(3) - total_boxes.col(1) + 1 ;
                    cv::Mat h = total_boxes.col(2) - total_boxes.col(0) + 1 ;

                    cv::Mat ww ; w.copyTo(ww); cv::hconcat(w,w,w) ; cv::hconcat(w,w,w) ;cv::hconcat(w,ww,w) ;
                    cv::Mat tb1 ; tb1 = total_boxes.col(0); cv::hconcat(tb1,tb1,tb1) ; cv::hconcat(tb1,tb1,tb1) ;cv::hconcat(tb1,total_boxes.col(0),tb1) ;

                    cv::Mat hh ; h.copyTo(hh); cv::hconcat(h,h,h) ; cv::hconcat(h,h,h) ;cv::hconcat(h,hh,h) ;
                    cv::Mat tb2 ; tb2 = total_boxes.col(1); cv::hconcat(tb2,tb2,tb2) ; cv::hconcat(tb2,tb2,tb2) ;cv::hconcat(tb2,total_boxes.col(1),tb2) ;


                    points.colRange(0,5)  = w.mul(points.colRange(0,5)) + tb1 -1 ;
                    points.colRange(5,10) = h.mul(points.colRange(5,10)) + tb2 -1 ;

                    if (total_boxes.rows>0){
                        cv::transpose(mv,mv);
                        bbreg(total_boxes, mv) ;
                        cv::Mat pick = nms(total_boxes, 0.7, NMS_MIN);
                        if (pick.rows > 0) {
                            temp.release();
                            cv::Mat point ;

                            for( cv::MatIterator_<int> it = pick.begin<int>(), end = pick.end<int>(); it != end; ++it)
                            {

                                temp.push_back(total_boxes.row(*it));
                                point.push_back(points.row(*it));
                            }
                            point.copyTo(points);
                            temp.copyTo(total_boxes);
                        }
                    }
                }
            }
        }
        else {
            total_boxes.release();
        }
    }
    cv::vector<cv::Mat> ret ;
    ret.push_back(total_boxes);
    ret.push_back(points);

    return ret ;
}

void MTCNNFaceDetection::drawBoxes(cv::Mat &img,cv::Mat boxes) {
    if(boxes.rows>0) {
        cv::Mat x1 = boxes.col(0) ;
        cv::Mat y1 = boxes.col(1) ;
        cv::Mat x2 = boxes.col(2) ;
        cv::Mat y2 = boxes.col(3) ;
        for (int i = 0; i < x1.rows; ++i) {
            cv::rectangle(img,cv::Rect(int(x1.at<float>(i)),int(y1.at<float>(i)),x2.at<float>(i) - int(x1.at<float>(i)),y2.at<float>(i)-int(y1.at<float>(i))),cv::Scalar(0,255,0)) ;
        }
    }


}
cv::vector<cv::Rect> MTCNNFaceDetection::getFaceBoxRect(cv::Mat boxes  ) {
    cv::vector<cv::Rect> rects ;
    if(boxes.rows>0) {
        cv::Mat x1 = boxes.col(0) ;
        cv::Mat y1 = boxes.col(1) ;
        cv::Mat x2 = boxes.col(2) ;
        cv::Mat y2 = boxes.col(3) ;
        for (int i = 0; i < x1.rows; ++i) {
            rects.push_back(cv::Rect(int(x1.at<float>(i)),int(y1.at<float>(i)),x2.at<float>(i) - int(x1.at<float>(i)),y2.at<float>(i)-int(y1.at<float>(i)))) ;
        }
    }
    return rects ;

}

cv::vector<cv::vector<cv::Point2i>> MTCNNFaceDetection::getFacePoints(cv::Mat pointsMat)  {
    cv::vector<cv::vector<cv::Point2i>> points ;
    for(int i=0 ; i<pointsMat.rows ; ++i) {
        cv::vector<cv::Point2i> tmp ;
        tmp.push_back(cv::Point(int(pointsMat.at<float>(i,0)),int(pointsMat.at<float>(i,5)))) ;
        for (int j = 0; j < (pointsMat.cols/2); ++j) {
            tmp.push_back(cv::Point(int(pointsMat.at<float>(i,j)),int(pointsMat.at<float>(i,5+j)))) ;
        }
        points.push_back(tmp);
    }
    return points ;
}






