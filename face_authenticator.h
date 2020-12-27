#ifndef FACE_AUTHENTICATOR_H
#define FACE_AUTHENTICATOR_H

#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <string>
#include "neural_network.h"

struct Rectangle {
    long left;
    long top;
    long right;
    long bottom;

};

class Authenticator {
public:

    explicit Authenticator(int32_t number_jitter);

    void Init(const std::string &, const std::string &);

    Rectangle DetectFace(const dlib::matrix<dlib::rgb_pixel> &);

    dlib::matrix<dlib::rgb_pixel> ExtractFace(const dlib::matrix<dlib::rgb_pixel> &, Rectangle &);

    dlib::matrix<float, 0, 1> GenerateEmbeddings(const dlib::matrix<dlib::rgb_pixel> &);


    static double ComputeDistance(const dlib::matrix<float, 0, 1> &, const dlib::matrix<float, 0, 1> &);


private:
    anet_type neural_net;
    dlib::shape_predictor shape_predictor;
    dlib::frontal_face_detector detector;
    int32_t number_jitter;

    std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel> &img);
};


#endif //FACE_AUTHENTICATOR_H
