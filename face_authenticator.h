#ifndef FACE_AUTHENTICATOR_H
#define FACE_AUTHENTICATOR_H

#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <string>
#include "neural_network.h"
#include "load_mem_jpeg.h"

#define EMBEDDINGS_SIZE 128

typedef struct {
    long left;
    long top;
    long right;
    long bottom;

} Rectangle;

class Authenticator {
public:

    explicit Authenticator(int32_t number_jitter);

    void Init(const std::string &, const std::string &);

    Rectangle DetectFace(const Image &);

    Image ExtractFace(const Image &, Rectangle &);

    dlib::matrix<float, 0, 1> GenerateEmbeddings(const Image &);


    double ComputeDistance(const dlib::matrix<float, 0, 1> &, const dlib::matrix<float, 0, 1> &);


private:
    std::mutex detector_mutex;
    std::mutex net_mutex;
    anet_type neural_net;
    dlib::shape_predictor shape_predictor;
    dlib::frontal_face_detector detector;
    int32_t number_jitter;

    std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel> &img);
};

long serialize_embeddings(const dlib::matrix<float, 0, 1> &, float *);
dlib::matrix<float, 0, 1> deserialize_embeddings(const float *);

#endif //FACE_AUTHENTICATOR_H
