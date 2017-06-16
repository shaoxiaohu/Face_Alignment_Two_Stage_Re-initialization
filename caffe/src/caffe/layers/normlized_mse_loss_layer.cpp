#include <vector>

#include "caffe/layers/normlized_mse_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

//#define DEBUGINFO
#ifdef DEBUGINFO 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "stdio.h"
using namespace std;
#endif

namespace caffe {

template <typename Dtype>
void NormlizedMSELossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NormlizedMSELossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NormlizedMSEParameter norm_mse_param = this->layer_param_.normlized_mse_param();
  int point_num = norm_mse_param.point_num();
  CHECK_GE(point_num, 0)
      << "Number of landmarks must be a positive integer.";
  int left_eye_idx = norm_mse_param.left_eye_idx();
  CHECK_LT(left_eye_idx, point_num)
      << "Index of left eye must be smaller than number of landmarks.";
  int right_eye_idx = norm_mse_param.right_eye_idx();
  CHECK_LT(left_eye_idx, point_num)
      << "Index of right eye must be smaller than number of landmarks.";

  int count = bottom[0]->count();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

#ifdef DEBUGINFO 
  Dtype gt_pt_x, gt_pt_y, pr_pt_x, pr_pt_y;
  Dtype diff_x, diff_y;
  float u_rnd;
  int idx, n_rnd;

  caffe_rng_uniform<float>(1, 0, 1, &u_rnd);
  n_rnd = cvRound(u_rnd * num) % num;

  idx = n_rnd * 2 * point_num; 
  pr_pt_x = bottom[0]->cpu_data()[idx];
  pr_pt_y = bottom[0]->cpu_data()[idx + point_num];

  gt_pt_x = bottom[1]->cpu_data()[idx];
  gt_pt_y = bottom[1]->cpu_data()[idx + point_num];

  diff_x = diff_.cpu_data()[idx];
  diff_y = diff_.cpu_data()[idx + point_num];

  cout << "n = " << n_rnd << ": " <<endl;
  cout << "predict point: (" << pr_pt_x << ", " << pr_pt_y << ");" <<endl;
  cout << "ground truth point: (" << gt_pt_x << ", " << gt_pt_y << ");" <<endl;
  cout << "diff of points: (" << diff_x << ", " << diff_y << ");" << endl;
#endif

  // normalized mse dividey by inter-ocular distances
  Blob<Dtype> interOcular;
  interOcular.ReshapeLike(*bottom[0]);
  for (int i = 0; i < num; i++)
  {
      float delX, delY, interOc;
      /*delX = bottom[1]->cpu_data()[i * channels + left_eye_idx] - 
              bottom[1]->cpu_data()[i * channels + right_eye_idx];
      delY = bottom[1]->cpu_data()[i * channels + left_eye_idx + point_num] - 
              bottom[1]->cpu_data()[i * channels + right_eye_idx + point_num];*/

      delX = bottom[1]->cpu_data()[i * channels + 2*left_eye_idx] - 
              bottom[1]->cpu_data()[i * channels + 2*right_eye_idx];
      delY = bottom[1]->cpu_data()[i * channels + 2*left_eye_idx + 1] - 
              bottom[1]->cpu_data()[i * channels + 2*right_eye_idx + 1];

      interOc = sqrt(delX*delX + delY*delY + 0.0001);      

      // caffe_cpu_axpby(
      //     channels,                                 // count
      //     Dtype(1/interOc),                         // alpha
      //     &(diff_.cpu_data()[i * channels]),           // a
      //     Dtype(0),                                 // beta
      //     &(diff_.mutable_cpu_data()[i * channels]));  // b 

      for (int j = 0; j < channels; j++)
      {
          interOcular.mutable_cpu_data()[i * channels + j] = Dtype(1/interOc);
          //interOcular.mutable_cpu_data()[i * channels + j] = Dtype(1.0);
      }
  }

  caffe_mul(
      count, 
      diff_.cpu_data(),
      interOcular.cpu_data(), 
      diff_.mutable_cpu_data());

#ifdef DEBUGINFO 
      Dtype diff_x_n, diff_y_n, interOc;

      diff_x_n = diff_.cpu_data()[idx];
      diff_y_n = diff_.cpu_data()[idx + point_num];

      interOc = interOcular.cpu_data()[idx];

      cout << "Distance of inter-ocular:" << interOc << endl; 
      cout << "normalized diff of points: (" << diff_x_n << ", " << diff_y_n << ");" << endl;

      /*cv::Mat m = cv::Mat::zeros(256, 256, cv::CV_8UC3);
      cv::imshow("Back Ground", m);
      cv::waitKey(); */
      int j = 0;
      cin >> j; 
#endif

  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void NormlizedMSELossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
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
STUB_GPU(NormlizedMSELossLayer);
#endif

INSTANTIATE_CLASS(NormlizedMSELossLayer);
REGISTER_LAYER_CLASS(NormlizedMSELoss);

}  // namespace caffe
