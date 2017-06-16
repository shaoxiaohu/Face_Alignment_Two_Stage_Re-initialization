// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/subsample_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

#if _MSC_VER < 1800
inline double round(double x) {
	return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}
#endif

namespace caffe {

	template <typename Dtype>
	void SubsamplePoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			SubsamplePoolingParameter subsample_pool_param = this->layer_param_.subsample_pooling_param();
			
			output_H_ = subsample_pool_param.output_h();
			output_W_ = subsample_pool_param.output_h();

			channels_ = bottom[0]->channels();
			input_H_ = bottom[0]->height();
			input_W_ = bottom[0]->width();
	}

	template <typename Dtype>
	void SubsamplePoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			top[0]->Reshape(bottom[0]->num(), channels_, output_H_,
				output_W_);
	}

	template <typename Dtype>
	void SubsamplePoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			int top_count = top[0]->count();
			Dtype* top_data = top[0]->mutable_cpu_data();
			caffe_set(top_count, Dtype(0), top_data);

			const Dtype bin_size_h = static_cast<Dtype>(input_H_)
				/ static_cast<Dtype>(output_H_);
			const Dtype bin_size_w = static_cast<Dtype>(input_W_)
				/ static_cast<Dtype>(output_W_);
			for (int n = 0; n < bottom[0]->num(); ++n) {
				for (int c = 0; c < channels_; ++c) {
					for (int ph = 0; ph < output_H_; ++ph) {
						for (int pw = 0; pw < output_W_; ++pw) {
							// Compute pooling region for this output unit:
							int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
								* bin_size_h));
							int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
								* bin_size_w));
							int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
								* bin_size_h));
							int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
								* bin_size_w));

							hstart = min(hstart, input_H_);
							hend = min(hend, input_H_);
							wstart = min(wstart, input_W_);
							wend = min(wend, input_W_);
							int pool_size = (hend - hstart) * (wend - wstart);

							bool is_empty = (hend <= hstart) || (wend <= wstart);

							const int pool_index = ph * output_W_ + pw;
							if (is_empty) {
								top_data[pool_index] = 0;
							}

							for (int h = hstart; h < hend; ++h) {
								for (int w = wstart; w < wend; ++w) {
									top_data[pool_index] += bottom_data[h * input_W_ + w];
								}
							}
							top_data[pool_index] /= pool_size;
						}
					}
					// Increment all data pointers by one channel
					bottom_data += bottom[0]->offset(0, 1);
					top_data += top[0]->offset(0, 1);
				}
			}
	}

	template <typename Dtype>
	void SubsamplePoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			//NOT_IMPLEMENTED;
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	}


#ifdef CPU_ONLY
	STUB_GPU(SubsamplePoolingLayer);
#endif

INSTANTIATE_CLASS(SubsamplePoolingLayer);
REGISTER_LAYER_CLASS(SubsamplePooling);

}  // namespace caffe
