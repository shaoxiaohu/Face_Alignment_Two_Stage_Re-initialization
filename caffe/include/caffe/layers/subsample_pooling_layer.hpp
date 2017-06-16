#ifndef CAFFE_SUBSAMPLE_POOLING_LAYER_HPP_
#define CAFFE_SUBSAMPLE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Pools the input image by taking the max, average, etc. within regions.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
/* SubsamplePoolingLayer - Region of Interest Pooling Layer
*/
template <typename Dtype>
class SubsamplePoolingLayer : public Layer<Dtype> {
public:
	explicit SubsamplePoolingLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ROIPooling"; }

	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MaxBottomBlobs() const { return 1; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int channels_;
	int input_H_;
	int input_W_;
	int output_H_;
	int output_W_;
};

}  // namespace caffe

#endif  // CAFFE_SUBSAMPLE_POOLING_LAYER_HPP_
