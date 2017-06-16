

#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/theta_calculate_layer.hpp"
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
	void ThetaCalculateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			ThetaCalculateParameter part_param = this->layer_param_.part_param();
			
	}

	template <typename Dtype>
	void ThetaCalculateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			point_num_ = bottom[0]->channels()/2;
      if (this->layer_param_.part_param().transform_type()== ThetaCalculateParameter_TransformType_AFFINE)
			  top[0]->Reshape(bottom[0]->num(), 6, 1, 1);
      else
        top[0]->Reshape(bottom[0]->num(), 4, 1, 1);
	}

	template <typename Dtype>
	void ThetaCalculateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_shape = bottom[0]->cpu_data();
			int batch_size = bottom[0]->num();
			
			int top_count = top[0]->count();
			Dtype* top_data = top[0]->mutable_cpu_data();
			caffe_set(top_count, Dtype(0), top_data);

			cv::Point2f srcKeyPoint[3];
            cv::Point2f dstKeyPoint[3];

      if (this->layer_param_.part_param().transform_type()== ThetaCalculateParameter_TransformType_AFFINE)
      {
    			switch (this->layer_param_.part_param().part_type()) {
      			case ThetaCalculateParameter_PartType_LEFTEYE:
      			    dstKeyPoint[0].y = ALIGN_POINT_18x;
                    dstKeyPoint[0].x = ALIGN_POINT_18y;
                    dstKeyPoint[1].y = ALIGN_POINT_22x;
                    dstKeyPoint[1].x = ALIGN_POINT_22y;
                    dstKeyPoint[2].y = ALIGN_LEFT_EYE_x;
                    dstKeyPoint[2].x = ALIGN_LEFT_EYE_y;
    				for (int n = 0; n < batch_size; ++n) {
    					const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
    					srcKeyPoint[0].y = batch_shape[2*17];
                        srcKeyPoint[0].x = batch_shape[2*17+1];
                        srcKeyPoint[1].y = batch_shape[2*21];
                        srcKeyPoint[1].x = batch_shape[2*21+1];

                        cv::Point2f mpt;
                        cv::Mat map_matrix;
           				mpt.x = 0;
            			mpt.y = 0;
    					for (int p = 36; p < 42; p++) {	
    						mpt.x += batch_shape[2*p]; 
              				mpt.y += batch_shape[2*p+1];				
    					}
    					srcKeyPoint[2].y = mpt.x/6;
            			srcKeyPoint[2].x = mpt.y/6;
            			map_matrix = cv::getAffineTransform(dstKeyPoint, srcKeyPoint);

      					for (int m=0; m<map_matrix.rows; m++)
        					for(int n=0; n<map_matrix.cols; n++) {
          						top_data[m*map_matrix.cols+n] = (Dtype)map_matrix.at<double>(m,n);
        					}
    					top_data += top[0]->offset(1);
    				}
    				break;
    			case ThetaCalculateParameter_PartType_RIGHTEYE:
      			    dstKeyPoint[0].y = ALIGN_POINT_23x;
                    dstKeyPoint[0].x = ALIGN_POINT_23y;
                    dstKeyPoint[1].y = ALIGN_POINT_27x;
                    dstKeyPoint[1].x = ALIGN_POINT_27y;
                    dstKeyPoint[2].y = ALIGN_RIGHT_EYE_x;
                    dstKeyPoint[2].x = ALIGN_RIGHT_EYE_y;
    				for (int n = 0; n < batch_size; ++n) {
    					const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
    					srcKeyPoint[0].y = batch_shape[2*22];
                        srcKeyPoint[0].x = batch_shape[2*22+1];
                        srcKeyPoint[1].y = batch_shape[2*26];
                        srcKeyPoint[1].x = batch_shape[2*26+1];

                        cv::Point2f mpt;
                        cv::Mat map_matrix;
           				mpt.x = 0;
            			mpt.y = 0;
    					for (int p = 42; p < 48; p++) {	
    						mpt.x += batch_shape[2*p]; 
              				mpt.y += batch_shape[2*p+1];				
    					}
    					srcKeyPoint[2].y = mpt.x/6;
            			srcKeyPoint[2].x = mpt.y/6;
            			map_matrix = cv::getAffineTransform(dstKeyPoint, srcKeyPoint);

      					for (int m=0; m<map_matrix.rows; m++)
        					for(int n=0; n<map_matrix.cols; n++) {
          						top_data[m*map_matrix.cols+n] = (Dtype)map_matrix.at<double>(m,n);
        					}
    					top_data += top[0]->offset(1);
    				}
    				break;
    			case ThetaCalculateParameter_PartType_NOSE:
      			    dstKeyPoint[0].y = ALIGN_POINT_28x;
                    dstKeyPoint[0].x = ALIGN_POINT_28y;
                    dstKeyPoint[1].y = ALIGN_POINT_32x;
                    dstKeyPoint[1].x = ALIGN_POINT_32y;
                    dstKeyPoint[2].y = ALIGN_POINT_36x;
                    dstKeyPoint[2].x = ALIGN_POINT_36y;
    				for (int n = 0; n < batch_size; ++n) {
    					const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
    					cv::Mat map_matrix;
    					srcKeyPoint[0].y = batch_shape[2*27];
                        srcKeyPoint[0].x = batch_shape[2*27+1];
                        srcKeyPoint[1].y = batch_shape[2*31];
                        srcKeyPoint[1].x = batch_shape[2*31+1];
                        srcKeyPoint[2].y = batch_shape[2*35];
            			srcKeyPoint[2].x = batch_shape[2*35+1];               
           				
            			map_matrix = cv::getAffineTransform(dstKeyPoint, srcKeyPoint);

      					for (int m=0; m<map_matrix.rows; m++)
        					for(int n=0; n<map_matrix.cols; n++) {
          						top_data[m*map_matrix.cols+n] = (Dtype)map_matrix.at<double>(m,n);
        					}
    					top_data += top[0]->offset(1);
    				}
    				break;
    			case ThetaCalculateParameter_PartType_MOUTH:
      			    dstKeyPoint[0].y = ALIGN_POINT_58x;
                    dstKeyPoint[0].x = ALIGN_POINT_58y;
                    dstKeyPoint[1].y = ALIGN_POINT_49x;
                    dstKeyPoint[1].x = ALIGN_POINT_49y;
                    dstKeyPoint[2].y = ALIGN_POINT_55x;
                    dstKeyPoint[2].x = ALIGN_POINT_55y;
    				for (int n = 0; n < batch_size; ++n) {
    					const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
    					cv::Mat map_matrix;
    					srcKeyPoint[0].y = batch_shape[2*57];
                        srcKeyPoint[0].x = batch_shape[2*57+1];
                        srcKeyPoint[1].y = batch_shape[2*48];
                        srcKeyPoint[1].x = batch_shape[2*48+1];
                        srcKeyPoint[2].y = batch_shape[2*54];
            			srcKeyPoint[2].x = batch_shape[2*54+1];               
           				
            			map_matrix = cv::getAffineTransform(dstKeyPoint, srcKeyPoint);

      					for (int m=0; m<map_matrix.rows; m++)
        					for(int n=0; n<map_matrix.cols; n++) {
          						top_data[m*map_matrix.cols+n] = (Dtype)map_matrix.at<double>(m,n);
        					}
    					top_data += top[0]->offset(1);
    				}
    				break;
    			default:
        			LOG(FATAL) << "Unknown transform type.";
            }
        }
        else
        {
         
         switch (this->layer_param_.part_param().part_type()) {
          case ThetaCalculateParameter_PartType_LEFTEYE:
                
            for (int n = 0; n < batch_size; ++n) {
              const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
              float LT_x=FLT_MAX, LT_y=FLT_MAX, RB_x=0, RB_y=0;
              for (int i = 17; i < 22; ++i)
              {
                if(LT_x > batch_shape[2*i+1])
                  LT_x = batch_shape[2*i+1];
                if(LT_y > batch_shape[2*i])
                  LT_y = batch_shape[2*i];
                if(RB_x < batch_shape[2*i+1])
                  RB_x = batch_shape[2*i+1];
                if(RB_y < batch_shape[2*i])
                  RB_y = batch_shape[2*i];
              }
              for (int i = 36; i < 42; ++i)
              {
                if(LT_x > batch_shape[2*i+1])
                  LT_x = batch_shape[2*i+1];
                if(LT_y > batch_shape[2*i])
                  LT_y = batch_shape[2*i];
                if(RB_x < batch_shape[2*i+1])
                  RB_x = batch_shape[2*i+1];
                if(RB_y < batch_shape[2*i])
                  RB_y = batch_shape[2*i];
              }
              float width = RB_x - LT_x, height = RB_y - LT_y;
              float ext_scale = 0.1;
              LT_x = LT_x - width * ext_scale/2;
              LT_y = LT_x - height * ext_scale/2;
              RB_x = RB_x + width * ext_scale/2;
              RB_y = RB_y + height * ext_scale/2;
              top_data[0] = (Dtype)(RB_y - LT_y) / 2;
              top_data[1] = (Dtype)(RB_y + LT_y) / 2;
              top_data[2] = (Dtype)(RB_x - LT_x) / 2;
              top_data[3] = (Dtype)(RB_x + LT_x) / 2;

              top_data += top[0]->offset(1);
            }
            break;
          case ThetaCalculateParameter_PartType_RIGHTEYE:
              for (int n = 0; n < batch_size; ++n) {
                const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
                float LT_x=FLT_MAX, LT_y=FLT_MAX, RB_x=0, RB_y=0;
                for (int i = 22; i < 27; ++i)
                {
                  if(LT_x > batch_shape[2*i+1])
                    LT_x = batch_shape[2*i+1];
                  if(LT_y > batch_shape[2*i])
                    LT_y = batch_shape[2*i];
                  if(RB_x < batch_shape[2*i+1])
                    RB_x = batch_shape[2*i+1];
                  if(RB_y < batch_shape[2*i])
                    RB_y = batch_shape[2*i];
                }
                for (int i = 42; i < 48; ++i)
                {
                  if(LT_x > batch_shape[2*i+1])
                    LT_x = batch_shape[2*i+1];
                  if(LT_y > batch_shape[2*i])
                    LT_y = batch_shape[2*i];
                  if(RB_x < batch_shape[2*i+1])
                    RB_x = batch_shape[2*i+1];
                  if(RB_y < batch_shape[2*i])
                    RB_y = batch_shape[2*i];
                }
                float width = RB_x - LT_x, height = RB_y - LT_y;
                float ext_scale = 0.1;
                LT_x = LT_x - width * ext_scale/2;
                LT_y = LT_x - height * ext_scale/2;
                RB_x = RB_x + width * ext_scale/2;
                RB_y = RB_y + height * ext_scale/2;
                top_data[0] = (Dtype)(RB_y - LT_y) / 2;
                top_data[1] = (Dtype)(RB_y + LT_y) / 2;
                top_data[2] = (Dtype)(RB_x - LT_x) / 2;
                top_data[3] = (Dtype)(RB_x + LT_x) / 2;

                top_data += top[0]->offset(1);
            }
            break;
          case ThetaCalculateParameter_PartType_NOSE:
              for (int n = 0; n < batch_size; ++n) {
                const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
                float LT_x=FLT_MAX, LT_y=FLT_MAX, RB_x=0, RB_y=0;
                for (int i = 27; i < 36; ++i)
                {
                  if(LT_x > batch_shape[2*i+1])
                    LT_x = batch_shape[2*i+1];
                  if(LT_y > batch_shape[2*i])
                    LT_y = batch_shape[2*i];
                  if(RB_x < batch_shape[2*i+1])
                    RB_x = batch_shape[2*i+1];
                  if(RB_y < batch_shape[2*i])
                    RB_y = batch_shape[2*i];
                }
                
                float width = RB_x - LT_x, height = RB_y - LT_y;
                float ext_scale = 0.1;
                LT_x = LT_x - width * ext_scale/2;
                LT_y = LT_x - height * ext_scale/2;
                RB_x = RB_x + width * ext_scale/2;
                RB_y = RB_y + height * ext_scale/2;
                top_data[0] = (Dtype)(RB_y - LT_y) / 2;
                top_data[1] = (Dtype)(RB_y + LT_y) / 2;
                top_data[2] = (Dtype)(RB_x - LT_x) / 2;
                top_data[3] = (Dtype)(RB_x + LT_x) / 2;

                top_data += top[0]->offset(1);
            }
            break;
          case ThetaCalculateParameter_PartType_MOUTH:
              for (int n = 0; n < batch_size; ++n) {
                const Dtype* batch_shape = bottom_shape + bottom[0]->offset(n);
                float LT_x=FLT_MAX, LT_y=FLT_MAX, RB_x=0, RB_y=0;
                for (int i = 48; i < 68; ++i)
                {
                  if(LT_x > batch_shape[2*i+1])
                    LT_x = batch_shape[2*i+1];
                  if(LT_y > batch_shape[2*i])
                    LT_y = batch_shape[2*i];
                  if(RB_x < batch_shape[2*i+1])
                    RB_x = batch_shape[2*i+1];
                  if(RB_y < batch_shape[2*i])
                    RB_y = batch_shape[2*i];
                }
                
                float width = RB_x - LT_x, height = RB_y - LT_y;
                float ext_scale = 0.1;
                LT_x = LT_x - width * ext_scale/2;
                LT_y = LT_x - height * ext_scale/2;
                RB_x = RB_x + width * ext_scale/2;
                RB_y = RB_y + height * ext_scale/2;
                top_data[0] = (Dtype)(RB_y - LT_y) / 2;
                top_data[1] = (Dtype)(RB_y + LT_y) / 2;
                top_data[2] = (Dtype)(RB_x - LT_x) / 2;
                top_data[3] = (Dtype)(RB_x + LT_x) / 2;

                top_data += top[0]->offset(1);
            }
            break;
          default:
              LOG(FATAL) << "Unknown transform type.";
            }
        }    
	}

	template <typename Dtype>
	void ThetaCalculateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			//NOT_IMPLEMENTED;
			Dtype* bottom_diff1 = bottom[0]->mutable_cpu_diff();
			//Dtype* bottom_diff2 = bottom[1]->mutable_cpu_diff();
	}


#ifdef CPU_ONLY
	STUB_GPU(ThetaCalculateLayer);
#endif

INSTANTIATE_CLASS(ThetaCalculateLayer);
REGISTER_LAYER_CLASS(ThetaCalculate);

}  // namespace caffe
