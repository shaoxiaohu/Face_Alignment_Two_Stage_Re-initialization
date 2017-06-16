#ifndef CAFFE_IMAGE_SCALE_LAYER_HPP_  
#define CAFFE_IMAGE_SCALE_LAYER_HPP_  
#include "caffe/blob.hpp"  
#include "caffe/layer.hpp"  
#include "caffe/proto/caffe.pb.h"  
#include "caffe/layer.hpp"  
namespace caffe {  
// written by xizero00 2016/9/13  
template <typename Dtype>  
class ImageScaleLayer : public Layer<Dtype> {  
 public:  
  explicit ImageScaleLayer(const LayerParameter& param)  
      : Layer<Dtype>(param) {}  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  virtual inline const char* type() const { return "ImageScale"; }  
  virtual inline int ExactNumBottomBlobs() const { return 1; }  
  virtual inline int ExactNumTopBlobs() const { return 1; }  
 protected:  
  /// @copydoc ImageScaleLayer  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {};  
  int out_height_;  
  int out_width_;  
  int height_;  
  int width_;  
  bool visualize_;  
  int num_images_;  
  int num_channels_;  
};  
}  // namespace caffe  
#endif  // CAFFE_IMAGE_SCALE_LAYER_HPP_  
