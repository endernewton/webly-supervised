#include <algorithm>
#include <cfloat>
#include <vector>

#include <iostream>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::ifstream;
// using std::ofstream;
using std::ios;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
  // X.C.
  const string source(this->layer_param_.softmax_param().relation());
  if (source.empty())
    return;
  LOG(INFO) << "Reading in relationships from: " << source;
  ifstream fi(source.c_str(), ios::in|ios::binary);
  if (!fi.is_open()) {
    LOG(FATAL) << "No relationships file!";
  }
  fi.read((char*)&classnumber, sizeof(classnumber));
  LOG(INFO) << classnumber;
  // relationships.resize(classnumber * classnumber);
  Dtype f;
  int flag = 0;
  for (int i=0; i<classnumber; i++) {
    for (int j=0; j<classnumber; j++) {
      fi.read((char*)&f, sizeof(Dtype));
      relationships.push_back(f);
      if (flag < 10) {
        LOG(INFO) << relationships[i * classnumber + j];
        flag++;
      }
    }
  }
  fi.close();
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
  // should also consider the spatial dimension
  lost_inst.resize((bottom[1])->count());
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  int index = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      // should be bigger than zero
      lost_inst[index] = -log(std::max(prob_data[i * dim +
          static_cast<int>(label[index]) * spatial_dim + j],
                           Dtype(FLT_MIN)));
      loss += lost_inst[index];
      // loss -= log(std::max(prob_data[i * dim +
      //     static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
      //                      Dtype(FLT_MIN)));
      index ++;
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype lambda = this->layer_param_.softmax_param().lambda();
    if (lambda < 0.0) {
      Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      caffe_copy(prob_.count(), prob_data, bottom_diff);
      const Dtype* label = (*bottom)[1]->cpu_data();
      int num = prob_.num();
      int dim = prob_.count() / num;
      int spatial_dim = prob_.height() * prob_.width();
      for (int i = 0; i < num; ++i) {
        if (!relationships.empty()) {
          for (int j = 0; j < spatial_dim; ++j) {
            int thislabel = static_cast<int>(label[i * spatial_dim + j]);
            // LOG(INFO) << thislabel;
            int st = thislabel * classnumber;
            // LOG(INFO) << relationships[st];
            for (int k = 0; k < classnumber; k++) {
              bottom_diff[i * dim + k * spatial_dim + j] -= relationships[st + k];
            }
          }
        } else {
          for (int j = 0; j < spatial_dim; ++j) {
            bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
                * spatial_dim + j] -= 1;
          }
        }
      }
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
    } else {
      Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
      const Dtype* prob_data = prob_.cpu_data();
      // first clear
      caffe_set(prob_.count(), 0, bottom_diff);
      const Dtype* label = (*bottom)[1]->cpu_data();
      int num = prob_.num();
      int dim = prob_.count() / num;
      int spatial_dim = prob_.height() * prob_.width();
      int soft_dim = dim / spatial_dim;
      // only the ones below lambda is added
      int index = 0;
      int count = 0;
      if (relationships.empty()) {
        for (int i = 0; i < num; ++i) {
          for (int j = 0; j < spatial_dim; ++j) {
            if (lost_inst[index] < lambda) {
              for (int k = 0; k < soft_dim; k++) {
                int thisind = i * dim + k * spatial_dim + j;
                bottom_diff[thisind] = prob_data[thisind];
              }
              bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
                  * spatial_dim + j] -= 1;
              count ++;
            }
            index ++;
          }
        }
      } else {
        for (int i = 0; i < num; ++i) {
          for (int j = 0; j < spatial_dim; ++j) {
            if (lost_inst[index] < lambda) {
              int thislabel = static_cast<int>(label[i * spatial_dim + j]);
              int st = thislabel * soft_dim;
              for (int k = 0; k < soft_dim; k++) {
                int thisind = i * dim + k * spatial_dim + j;
                bottom_diff[thisind] = prob_data[thisind] - relationships[st + k];
              }
              count ++;
            }
            index ++;
          }
        }
      }
      LOG(INFO) << count;
      // Scale gradient
      const Dtype loss_weight = top[0]->cpu_diff()[0];
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
