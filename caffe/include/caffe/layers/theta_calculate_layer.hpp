#ifndef CAFFE_THETA_CALCULATE_LAYERHPP_
#define CAFFE_THETA_CALCULATE_LAYERHPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define ALIGN_POINT_18x  -0.78f
#define ALIGN_POINT_18y  -0.36f
#define ALIGN_POINT_22x  0.75f
#define ALIGN_POINT_22y  -0.53f
#define ALIGN_POINT_23x  -0.84f
#define ALIGN_POINT_23y  -0.53f
#define ALIGN_POINT_27x  0.69f
#define ALIGN_POINT_27y  -0.42f
#define ALIGN_POINT_28x  0.02f
#define ALIGN_POINT_28y  -0.73f
#define ALIGN_POINT_32x  -0.43f
#define ALIGN_POINT_32y  0.57f
#define ALIGN_POINT_36x  0.43f
#define ALIGN_POINT_36y  0.58f
#define ALIGN_POINT_49x  -0.77f
#define ALIGN_POINT_49y  0.03f
#define ALIGN_POINT_55x  0.75f
#define ALIGN_POINT_55y  0.03f
//#define ALIGN_POINT_31x  0.02f
//#define ALIGN_POINT_31y  -0.78f
#define ALIGN_POINT_58x  0.0154f
#define ALIGN_POINT_58y  0.40f

#define ALIGN_LEFT_EYE_x  0.08f
#define ALIGN_LEFT_EYE_y  -0.04f
#define ALIGN_RIGHT_EYE_x  -0.18f
#define ALIGN_RIGHT_EYE_y  -0.05f

namespace caffe {


template <typename Dtype>
class ThetaCalculateLayer : public Layer<Dtype> {
public:
  explicit ThetaCalculateLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ThetaCalculate Layer"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //   const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  int point_num_;
};

}  // namespace caffe

#endif  //CAFFE_POINT_TRANSFORMER_LAYERHPP_
