#include <vector>

#include "caffe/layers/hard_sample_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename T>
bool SortScorePairDescend(const pair<T, int>& pair1,
                          const pair<T, int>& pair2) {
  return pair1.first > pair2.first;
}

template <typename Dtype>
void HardSampleEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";


  const HardSampleEuclideanLayerParameter& hard_sample_param =
      this->layer_param_.hard_sample_param();
  hard_ratio = hard_sample_param.hard_ratio();

  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HardSampleEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  vector<pair<Dtype, int> > scores_indices;
  const Dtype* diff_data = diff_.cpu_data();
  const int sample_dim = bottom[0]->count(1);
  for (int n = 0; n < bottom[0]->num(); ++n) {
    Dtype sample_dot = caffe_cpu_dot(sample_dim, diff_data, diff_data);
    scores_indices.push_back(std::make_pair(sample_dot, n));
    diff_data += diff_.offset(1);
  }
  std::sort(scores_indices.begin(), scores_indices.end(), SortScorePairDescend<Dtype>);

  Dtype* set_diff_data = diff_.mutable_cpu_data();
  const Dtype alpha = 0;
  for (int n = bottom[0]->num()*hard_ratio; n < bottom[0]->num(); ++n) {
    int s_idx = scores_indices[n].second;
    Dtype* s_data = set_diff_data + s_idx * diff_.offset(1);
    caffe_set(sample_dim, alpha ,s_data);
  }

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2) / hard_ratio;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HardSampleEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / hard_ratio;
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HardSampleEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(HardSampleEuclideanLossLayer);
REGISTER_LAYER_CLASS(HardSampleEuclideanLoss);

}  // namespace caffe
