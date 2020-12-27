#include "face_authenticator.h"

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

Rectangle Authenticator::DetectFace(const Image &image) {
    for (rectangle face : detector(image.img)) {
        // return first face
        return {face.left(), face.top(), face.right(), face.bottom()};
    }

    return {0, 0, 0, 0};
}

Image Authenticator::ExtractFace(const Image &image, Rectangle &face_rect) {
    dlib::rectangle face_pos = {face_rect.left, face_rect.top, face_rect.right, face_rect.bottom};
    auto shape = this->shape_predictor(image.img, face_pos);
    matrix<rgb_pixel> face_chip;
    extract_image_chip(image.img, get_face_chip_details(shape, 150, 0.25), face_chip);
    matrix<rgb_pixel> face = move(face_chip);
    draw_rectangle(image.img, face_pos, dlib::rgb_pixel(255, 0, 0), 1);
    Image face_img;
    face_img.img = face;
    return face_img;
}


matrix<float, 0, 1> Authenticator::GenerateEmbeddings(const Image &image) {
    matrix<float, 0, 1> face_descriptors = mean(mat(this->neural_net(jitter_image(image.img))));
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
