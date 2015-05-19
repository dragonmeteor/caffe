#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[2]->num())
      << "The data and weight should have the same number.";
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1))
      << "Inputs must have the same dimension.";    
  diff_.ReshapeLike(*bottom[0]);
  weightedDiff_.ReshapeLike(*bottom[0]);
  diffSquared_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_mul(
    count,
    bottom[2]->cpu_data(),
    diff_.cpu_data(),
    weightedDiff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, weightedDiff_.cpu_data(), diff_.cpu_data());    
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),                 // count
          alpha,                              // alpha
          weightedDiff_.cpu_data(),           // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  } 
  if (propagate_down[2]) {    
    caffe_mul(
      bottom[2]->count(),
      diff_.cpu_data(),
      diff_.cpu_data(),
      diffSquared_.mutable_cpu_data());
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[2]->num() / 2;
    caffe_cpu_axpby(
      bottom[2]->count(),                 // count
      alpha,                              // alpha
      diffSquared_.cpu_data(),            // a
      Dtype(0),                           // beta
      bottom[2]->mutable_cpu_diff());  // b
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);
REGISTER_LAYER_CLASS(WeightedEuclideanLoss);

}  // namespace caffe
