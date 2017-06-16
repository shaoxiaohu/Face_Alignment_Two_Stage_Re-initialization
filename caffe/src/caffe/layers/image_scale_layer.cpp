#include "caffe/layers/image_scale_layer.hpp"  
#include "caffe/util/math_functions.hpp" 
#include <opencv2/opencv.hpp>  
namespace caffe {  
template <typename Dtype>  
void ImageScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  // get parameters  
  const ImageScaleParameter& param = this->layer_param_.image_scale_param();  
  // get the output size  
  out_height_ = param.out_height();  
  out_width_ = param.out_width();  
  visualize_ = param.visualize();  
   
  // get the input size  
  num_images_ = bottom[0]->num();  
  height_ = bottom[0]->height();  
  width_ = bottom[0]->width();  
  num_channels_ = bottom[0]->channels();  
  // check the channels must be images  
  // channel must be 1 or 3, gray image or color image  
  CHECK_EQ( (num_channels_==3) || (num_channels_ == 1), true);  
  // check the output size  
  CHECK_GT(out_height_, 0);  
  CHECK_GT(out_height_, 0);  
   
}  
template <typename Dtype>  
void ImageScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top) {  
  // reshape the outputs  
  top[0]->Reshape(num_images_, num_channels_, out_height_, out_width_);  
}  
template <typename Dtype>  
void ImageScaleLayer<Dtype>::Forward_cpu(  
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
  const Dtype* bottom_data = bottom[0]->cpu_data();  
  Dtype * top_data = top[0]->mutable_cpu_data();  
  cv::Mat srcimage, dstimage;  
   
  // precompurte the index  
  const int srcimagesize = width_ * height_;  
  const int dstimagesize = out_width_ *  out_height_;  
  const int srcchimagesize = srcimagesize * num_channels_;  
  const int dstchimagesize = dstimagesize * num_channels_;  
  for  ( int idx_img = 0; idx_img < num_images_; idx_img++ )  
  {  
        // zeros source images and scaled images  
        srcimage = cv::Mat::zeros(height_, width_, CV_32FC1);  
        dstimage = cv::Mat::zeros(out_height_, out_width_, CV_32FC1);  
        // read from bottom[0]  
        for  ( int idx_ch = 0; idx_ch < num_channels_; idx_ch++ )  
        {  
                for  (int i = 0; i < height_; i++)  
                {  
                        for ( int j=0; j < width_; j++ )  
                        {  
                                int image_idx = idx_img * srcchimagesize + srcimagesize * idx_ch + height_ *i + j;  
                                srcimage.at<float>(i,j) = (float)bottom_data[image_idx];  
                        }  
                }  
        }  
        // resize to specified size  
        // here we use linear interpolation  
        cv::resize(srcimage, dstimage, dstimage.size());  
        // store the resized image to top[0]  
        for (int idx_ch = 0; idx_ch < num_channels_; idx_ch++)  
        {  
                for (int i = 0; i < out_height_; i++)  
                {  
                        for (int j = 0; j < out_width_; j++)  
                        {  
                                int image_idx = idx_img * dstchimagesize + dstimagesize * idx_ch + out_height_*i + j;  
                                top_data[image_idx] = dstimage.at<float>(i,j);  
                        }  
                }  
        }  
        if (visualize_)  
        {  
                cv::namedWindow("src image", CV_WINDOW_AUTOSIZE);  
                cv::namedWindow("dst image", CV_WINDOW_AUTOSIZE);  
                cv::imshow("src image", srcimage);  
                cv::imshow("dst image", dstimage);  
                cv::waitKey(0);  
        }  
  }  
}  
#ifdef CPU_ONLY  
STUB_GPU(ImageScaleLayer);  
#endif  
INSTANTIATE_CLASS(ImageScaleLayer);  
REGISTER_LAYER_CLASS(ImageScale);  
}  // namespace caffe  
