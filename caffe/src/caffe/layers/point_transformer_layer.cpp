

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/point_transformer_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

#if _MSC_VER < 1800
inline double round(double x) {
	return (x > 0.0) ? floor(x + 0.5) : ceil(x - 0.5);
}
#endif

//#define DEBUGINFO 

namespace caffe {

	template <typename Dtype>
	void PointTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			PointTransformerParameter point_trans_param = this->layer_param_.pt_param();
			inv_trans = point_trans_param.inv_trans();
			
	}

	template <typename Dtype>
	void PointTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			point_num_ = bottom[0]->channels()/2;
			//top[0]->Reshape(bottom[0]->num(), point_num_*2, 1, 1);
			top[0]->ReshapeLike(*bottom[0]);
	}

	template <typename Dtype>
	void PointTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_shape = bottom[0]->cpu_data();
			const Dtype* bottom_param = bottom[1]->cpu_data();

			int batch_size = bottom[0]->num();
			
			int top_count = top[0]->count();
			Dtype* top_data = top[0]->mutable_cpu_data();
			caffe_set(top_count, Dtype(0), top_data);

			switch (this->layer_param_.pt_param().transform_type()) {
  			case PointTransformerParameter_TransformType_CROP:
				for (int n = 0; n < batch_size; ++n) {
					const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
					const Dtype* batch_param = bottom_param + bottom[1]->offset(n);

					for (int p = 0; p < point_num_; p++) {	
						//Dtype out_pt_x = (batch_shape[p] - batch_param[3])/(batch_param[2] + 0.00001);
						//Dtype out_pt_y = (batch_shape[p + point_num_] - batch_param[1])/(batch_param[0] + 0.00001);
						Dtype out_pt_x, out_pt_y;
						if (inv_trans) {
							out_pt_x = batch_shape[2*p] * batch_param[2] + batch_param[3];
							out_pt_y = batch_shape[2*p + 1] * batch_param[0] + batch_param[1];
						} else {
							out_pt_x = (batch_shape[2*p] - batch_param[3])/(batch_param[2]); // + 0.00001
							out_pt_y = (batch_shape[2*p + 1] - batch_param[1])/(batch_param[0]); // + 0.00001
						}

						top_data[2*p] = out_pt_x;
						top_data[2*p + 1] = out_pt_y;					
					}
					
					top_data += top[0]->offset(1);
				}
				break;
			case PointTransformerParameter_TransformType_AFFINE:
				for (int n = 0; n < batch_size; ++n) {
					const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
					const Dtype* batch_param = bottom_param + bottom[1]->offset(n);

					for (int p = 0; p < point_num_; p++) {	
						//Dtype out_pt_x = (batch_shape[p] - batch_param[3])/(batch_param[2] + 0.00001);
						//Dtype out_pt_y = (batch_shape[p + point_num_] - batch_param[1])/(batch_param[0] + 0.00001);
						Dtype out_pt_x, out_pt_y;
						if (inv_trans) {
							out_pt_x = batch_shape[2*p + 1] * batch_param[3] + batch_shape[2*p] * batch_param[4] + batch_param[5];
							out_pt_y = batch_shape[2*p + 1] * batch_param[0] + batch_shape[2*p] * batch_param[1] + batch_param[2];
						} else {
							out_pt_x = (batch_shape[2*p] * batch_param[0] - batch_shape[2*p + 1] * batch_param[3] - batch_param[0] * batch_param[5] + batch_param[2] * batch_param[3]) / (batch_param[0] * batch_param[4] - batch_param[1] * batch_param[3]);// + 0.00001
							out_pt_y = (batch_shape[2*p] * batch_param[1] - batch_shape[2*p + 1] * batch_param[4] - batch_param[1] * batch_param[5] + batch_param[2] * batch_param[4]) / (batch_param[1] * batch_param[3] - batch_param[0] * batch_param[4]); // + 0.00001
						}

						top_data[2*p] = out_pt_x;
						top_data[2*p + 1] = out_pt_y;					
					}
					
					top_data += top[0]->offset(1);
				}
				break;
			case PointTransformerParameter_TransformType_SIMILAR:
				for (int n = 0; n < batch_size; ++n) {
					const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
					const Dtype* batch_param = bottom_param + bottom[1]->offset(n);

					for (int p = 0; p < point_num_; p++) {	
						//Dtype out_pt_x = (batch_shape[p] - batch_param[3])/(batch_param[2] + 0.00001);
						//Dtype out_pt_y = (batch_shape[p + point_num_] - batch_param[1])/(batch_param[0] + 0.00001);
						Dtype out_pt_x, out_pt_y;
						if (inv_trans) {
							out_pt_x = -batch_shape[2*p + 1] * batch_param[1] + batch_shape[2*p] * batch_param[0] + batch_param[3];
							out_pt_y = batch_shape[2*p + 1] * batch_param[0] + batch_shape[2*p] * batch_param[1] + batch_param[2];
						} else {
							out_pt_x = (batch_shape[2*p] * batch_param[0] + batch_shape[2*p + 1] * batch_param[1] - batch_param[0] * batch_param[3] - batch_param[2] * batch_param[1]) / (batch_param[0] * batch_param[0] + batch_param[1] * batch_param[1]);// + 0.00001
							out_pt_y = (batch_shape[2*p] * batch_param[1] - batch_shape[2*p + 1] * batch_param[0] - batch_param[1] * batch_param[3] + batch_param[2] * batch_param[0]) / (-batch_param[1] * batch_param[1] - batch_param[0] * batch_param[0]); // + 0.00001
						}

						top_data[2*p] = out_pt_x;
						top_data[2*p + 1] = out_pt_y;					
					}
					
					top_data += top[0]->offset(1);
				}
				break;
			default:
    			LOG(FATAL) << "Unknown transform type.";
        }
	}

	template <typename Dtype>
	void PointTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			//NOT_IMPLEMENTED;
			Dtype* bottom_diff1 = bottom[0]->mutable_cpu_diff();
			Dtype* bottom_diff2 = bottom[1]->mutable_cpu_diff();
	}


#ifdef CPU_ONLY
	STUB_GPU(PointTransformerLayer);
#endif

INSTANTIATE_CLASS(PointTransformerLayer);
REGISTER_LAYER_CLASS(PointTransformer);

}  // namespace caffe
