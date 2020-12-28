#include "face_authenticator.h"

using namespace dlib;

Authenticator::Authenticator(int32_t number_jitter) {
    this->number_jitter = number_jitter;
}

void Authenticator::Init(const std::string &model1, const std::string &model2) {
    this->detector = get_frontal_face_detector();
    deserialize(model1) >> this->shape_predictor;
    deserialize(model2) >> this->neural_net;
}

Rectangle Authenticator::DetectFace(const Image &image) {
    std::lock_guard<std::mutex> lock(this->detector_mutex);
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
    matrix<rgb_pixel> face = std::move(face_chip);
    draw_rectangle(image.img, face_pos, dlib::rgb_pixel(255, 0, 0), 1);
    Image face_img;
    face_img.img = face;
    return face_img;
}


matrix<float, 0, 1> Authenticator::GenerateEmbeddings(const Image &image) {
    std::lock_guard<std::mutex> lock(this->net_mutex);
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

long serialize_embeddings(const dlib::matrix<float, 0, 1> &embeddings, float *array) {
    for (auto it = embeddings.begin(); it != embeddings.end(); ++it) {
        auto i = std::distance(embeddings.begin(), it);
        array[i] = *it;
    }
    return embeddings.size();
}

matrix<float, 0, 1> deserialize_embeddings(const float *array) {
    dlib::array<float> float_array = dlib::array<float>();
    for (long i = 0; i < 128; ++i) {
        float tmp = array[i];
        float_array.push_back(tmp);
    }
    auto float_mat = mat(float_array);
    auto float_matrix = matrix<float, 0, 1>(float_mat);
    return float_matrix;
}
