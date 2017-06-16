#ifndef CAFFE_POINT_TRANSFORMER_LAYERHPP_
#define CAFFE_POINT_TRANSFORMER_LAYERHPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/* PointTransformerLayer - Points Trasnformation layer
   Transform points/landmarks for input points/landmarks based input spatial transformation parameters
   input: [x1, y1, ..., xn, yn]
   output: [x1', y1', ..., xn', yn']

   20160705
*/
template <typename Dtype>
class PointTransformerLayer : public Layer<Dtype> {
public:
  explicit PointTransformerLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PointTransformer Layer"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
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
  
  bool inv_trans;
  int point_num_;
};

}  // namespace caffe

#endif  //CAFFE_POINT_TRANSFORMER_LAYERHPP_
