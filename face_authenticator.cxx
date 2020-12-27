#include "face_authenticator.h"
#include <dlib/image_io.h>

using namespace std;
using namespace dlib;

Authenticator::Authenticator(int32_t number_jitter) {
    this->number_jitter = number_jitter;
}


void Authenticator::Init(const string &model1, const string &model2) {
    this->detector = get_frontal_face_detector();
    deserialize(model1) >> this->shape_predictor;
    deserialize(model2) >> this->neural_net;
}

rectangle Authenticator::DetectFace(const matrix<rgb_pixel> &img) {
    for (rectangle face : detector(img)) {
        // return first face
        return face;
    }
    return {0, 0, 0, 0};
}

matrix<rgb_pixel> Authenticator::ExtractFace(const matrix<rgb_pixel> &img, rectangle &face_pos) {
    auto shape = this->shape_predictor(img, face_pos);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
    matrix<rgb_pixel> face = move(face_chip);
    draw_rectangle(img, face_pos, dlib::rgb_pixel(255, 0, 0), 1);
    return face;
}


matrix<float, 0, 1> Authenticator::GenerateEmbeddings(const matrix<rgb_pixel> &face) {
    matrix<float, 0, 1> face_descriptors = mean(mat(this->neural_net(jitter_image(face))));
    return face_descriptors;
}

double Authenticator::ComputeDistance(const matrix<float, 0, 1> &face1, const matrix<float, 0, 1> &face2) {
    return length(face1 - face2);
}

std::vector<matrix<rgb_pixel>> Authenticator::jitter_image(const matrix<rgb_pixel> &img) {
    thread_local dlib::rand rnd;
    std::vector<matrix<rgb_pixel>> crops;
    for (int i = 0; i < this->number_jitter; ++i)
        crops.push_back(::jitter_image(img, rnd));

    return crops;
}
